import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("drone_m_1280.pt")

# Initialize variables
zoom_region = None  # The area to zoom into
selected_bbox = None
selected_object_id = None  # Track which detected object is selected
tracker = None
zoom_factor = 2.0
is_tracking = False
drawing = False
start_point = None
playback_speed = 30
mode = "select_zoom"  # Modes: "select_zoom" or "detect_objects"
detected_objects = []  # Store detected objects in zoomed region
use_manual_selection = False  # Toggle between YOLO and manual selection
tracking_lost_frames = 0  # Counter for consecutive lost frames
MAX_LOST_FRAMES = 5  # Stop tracking after this many lost frames
input_video_path = "followcar.mp4"

# --- New zoom sequence state ---
zoom_steps = [6.0,5.5,5.0,4.5,4.0,3.5,3.0,2.5, 1.8, 1.2, 1.0]  # Step 1, 2, 3
zoom_step_index = 0
zoom_sequence_active = False
last_zoom_time = None
paused = False
attack_mode = False
# smoothing variables (new)
current_zoom_factor = 1.0   # actual zoom used for rendering (will approach zoom_factor)
zoom_smooth_alpha = 0.12    # interpolation speed: 0.02 = very slow, 0.3 = snappy

tick_freq = cv2.getTickFrequency()


def draw_corner_bbox(frame, bbox, color, label, thickness=2):
    x1, y1, x2, y2 = map(int, bbox)

    # Corner length
    corner_length = min((x2 - x1), (y2 - y1)) // 10
    corner_thickness = thickness

    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)

    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)

    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)

    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)

    # Draw label
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def crop_and_zoom(frame, bbox, zoom_factor):
    """
    Crop around bbox with zoom_factor and return:
    - zoomed_frame
    - crop coordinates (new_x1, new_y1, new_x2, new_y2) in original frame
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Calculate center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Calculate new dimensions
    width = int((x2 - x1) * zoom_factor)
    height = int((y2 - y1) * zoom_factor)

    # Calculate new coordinates
    new_x1 = max(0, center_x - width // 2)
    new_y1 = max(0, center_y - height // 2)
    new_x2 = min(frame.shape[1], new_x1 + width)
    new_y2 = min(frame.shape[0], new_y1 + height)

    # Crop the image
    zoomed_frame = frame[new_y1:new_y2, new_x1:new_x2]

    return zoomed_frame, (new_x1, new_y1, new_x2, new_y2)


def embed_zoomed_view(main_frame, zoomed_frame, position='top_right'):
    # No longer used (we now do full-screen zoom)
    return main_frame


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def find_matching_detection(tracked_bbox, detected_objects, iou_threshold=0.3):
    """Find which detected object matches the tracked bbox"""
    if not detected_objects:
        return None

    x, y, w, h = tracked_bbox
    tracked_box = (x, y, x + w, y + h)

    best_match = None
    best_iou = iou_threshold

    for idx, obj in enumerate(detected_objects):
        x1, y1, w1, h1 = obj['bbox_original']
        det_box = (x1, y1, x1 + w1, y1 + h1)

        iou = calculate_iou(tracked_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_match = idx

    return best_match


def start_zoom_sequence():
    """Initialize the 3-step zoom sequence when user selects an object."""
    global zoom_sequence_active, zoom_step_index, zoom_factor, last_zoom_time, paused
    zoom_sequence_active = True
    zoom_step_index = 0
    zoom_factor = zoom_steps[zoom_step_index]
    last_zoom_time = cv2.getTickCount() / tick_freq
    paused = False
    print(f"Zoom sequence started. Step 1, zoom_factor={zoom_factor}")


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, selected_bbox, tracker, is_tracking
    global zoom_region, mode, zoomed_frame_display, detected_objects, zoomed_frame_for_tracking
    global use_manual_selection, selected_object_id, tracking_lost_frames
    global zoom_sequence_active, zoom_step_index, zoom_factor, last_zoom_time, paused

    if mode == "select_zoom":
        # First mode: manually draw zoom region
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_frame = current_frame.copy()
                cv2.rectangle(temp_frame, start_point, (x, y), (255, 255, 0), 2)
                cv2.imshow("Manual Object Tracking", temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)

            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])

            if x2 - x1 > 10 and y2 - y1 > 10:
                zoom_region = (x1, y1, x2, y2)
                mode = "detect_objects"
                print(f"Zoom region selected: {zoom_region}")
                if use_manual_selection:
                    print("MANUAL mode: Draw a box around the target object")
                else:
                    print("YOLO mode: Click on any detected object to track it (or press 'm' for manual)")

    elif mode == "detect_objects":
        if use_manual_selection:
            # Manual mode: draw bounding box
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    temp_frame = zoomed_frame_display.copy()
                    cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 255), 2)
                    cv2.imshow("Manual Object Tracking", temp_frame)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)

                x1 = min(start_point[0], end_point[0])
                y1 = min(start_point[1], end_point[1])
                x2 = max(start_point[0], end_point[0])
                y2 = max(start_point[1], end_point[1])

                # Convert scaled coordinates back to original zoomed frame coordinates
                scale_x = zoomed_frame_for_tracking.shape[1] / zoomed_frame_display.shape[1]
                scale_y = zoomed_frame_for_tracking.shape[0] / zoomed_frame_display.shape[0]

                x1_original = int(x1 * scale_x)
                y1_original = int(y1 * scale_y)
                x2_original = int(x2 * scale_x)
                y2_original = int(y2 * scale_y)

                # Validate coordinates
                frame_h, frame_w = zoomed_frame_for_tracking.shape[:2]
                x1_original = max(0, min(x1_original, frame_w - 1))
                y1_original = max(0, min(y1_original, frame_h - 1))
                x2_original = max(0, min(x2_original, frame_w))
                y2_original = max(0, min(y2_original, frame_h))

                width = x2_original - x1_original
                height = y2_original - y1_original

                if width > 5 and height > 5:
                    selected_bbox = (x1_original, y1_original, width, height)
                    selected_object_id = None

                    # Initialize tracker
                    tracker = cv2.TrackerMIL_create()
                    tracker.init(zoomed_frame_for_tracking, selected_bbox)
                    is_tracking = True
                    tracking_lost_frames = 0
                    print(f"Manual tracking started - bbox: {selected_bbox}")

                    # Start zoom sequence on manual selection
                    start_zoom_sequence()
                else:
                    print("Selection too small. Please try again.")
        else:
            # YOLO mode: click on detected objects
            if event == cv2.EVENT_LBUTTONDOWN:
                for idx, obj in enumerate(detected_objects):
                    x1, y1, x2, y2 = obj['bbox_scaled']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected_bbox = obj['bbox_original']
                        selected_object_id = idx

                        # Initialize tracker
                        tracker = cv2.TrackerMIL_create()
                        tracker.init(zoomed_frame_for_tracking, selected_bbox)
                        is_tracking = True
                        tracking_lost_frames = 0
                        print(f"Tracking started on {obj['class_name']} - bbox: {selected_bbox}")

                        # Start zoom sequence when user clicks object
                        start_zoom_sequence()
                        return


# Open video capture
cap = cv2.VideoCapture(input_video_path)

# Create window
cv2.namedWindow("Manual Object Tracking")
cv2.setMouseCallback("Manual Object Tracking", mouse_callback)

# Read first frame
ret, current_frame = cap.read()
if not ret:
    print("Error: Cannot read video file")
    exit()

print("Instructions:")
print("Step 1: Click and drag to select the ZOOM REGION")
print("Step 2a: YOLO will detect objects - CLICK on any detected object to start tracking + 3-step zoom")
print("Step 2b: OR press 'm' to switch to MANUAL mode and draw box around target")
print("- Zoom steps: 1.2x -> 1.8x -> 2.5x, every 2 seconds, then video PAUSES")
print("- Press 'p' to RESUME after final pause")
print("- Press 'm' to toggle between YOLO and Manual mode")
print("- Press 'r' to reset and start over")
print("- Press '+' to manually increase zoom (only when sequence is not active)")
print("- Press '-' to manually decrease zoom")
print("- Press 'f' for faster playback")
print("- Press 's' for slower playback")
print("- Press 'q' to quit")

zoomed_frame_display = current_frame.copy()
zoomed_frame_for_tracking = current_frame.copy()

while True:
    # Handle pause: when paused, do NOT read a new frame
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed.")
            break
        current_frame = frame.copy()
    else:
        # Re-use last frame when paused
        frame = current_frame.copy()

    # If zoom region is selected, extract that region
    if zoom_region is not None:
        x1, y1, x2, y2 = zoom_region
        zoomed_frame_for_tracking = frame[y1:y2, x1:x2].copy()

        # Resize zoom-region to full screen for base view
        h, w = frame.shape[:2]
        zoomed_frame_display = cv2.resize(zoomed_frame_for_tracking, (w, h))

                # --- SMOOTH THE ZOOM FACTOR (NEW) ---
        current_zoom_factor += (zoom_factor - current_zoom_factor) * zoom_smooth_alpha

        # Small clamp to avoid tiny jitter
        if abs(current_zoom_factor - zoom_factor) < 0.001:
            current_zoom_factor = zoom_factor


        display_frame = zoomed_frame_display.copy()

        # Always run YOLO detection on the zoomed frame (if in YOLO mode)
        if not use_manual_selection:
            results = model(zoomed_frame_for_tracking, verbose=False)
            detected_objects = []

            if len(results[0].boxes) > 0:
                for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                    x_obj1, y_obj1, x_obj2, y_obj2 = box.cpu().numpy()
                    class_id = int(cls.cpu().numpy())
                    class_name = model.names[class_id]
                    confidence = float(conf.cpu().numpy())

                    # Store original bbox (in zoomed frame coordinates)
                    width = x_obj2 - x_obj1
                    height = y_obj2 - y_obj1
                    bbox_original = (int(x_obj1), int(y_obj1), int(width), int(height))

                    # Scale coordinates to display frame
                    scale_x = display_frame.shape[1] / zoomed_frame_for_tracking.shape[1]
                    scale_y = display_frame.shape[0] / zoomed_frame_for_tracking.shape[0]

                    x_scaled1 = int(x_obj1 * scale_x)
                    y_scaled1 = int(y_obj1 * scale_y)
                    x_scaled2 = int(x_obj2 * scale_x)
                    y_scaled2 = int(y_obj2 * scale_y)

                    bbox_scaled = (x_scaled1, y_scaled1, x_scaled2, y_scaled2)

                    detected_objects.append({
                        'bbox_original': bbox_original,
                        'bbox_scaled': bbox_scaled,
                        'class_name': class_name,
                        'confidence': confidence
                    })

        # --- Handle 3-step zoom timings ---
        if zoom_sequence_active and is_tracking and not paused:
            now = cv2.getTickCount() / tick_freq
            if last_zoom_time is None:
                last_zoom_time = now
            elif now - last_zoom_time >= 2.0:
                # Move to next zoom step
                if zoom_step_index < len(zoom_steps) - 1:
                    zoom_step_index += 1
                    zoom_factor = zoom_steps[zoom_step_index]
                    last_zoom_time = now
                    print(f"Zoom step {zoom_step_index + 1}/{len(zoom_steps)} - zoom_factor={zoom_factor}")

                    # If this is the last step, pause the video
                    if zoom_step_index == len(zoom_steps) - 1:
                        paused = True
                        zoom_sequence_active = False
                        attack_mode = True
                        print("Final zoom step reached. ATTACK MODE. Press 'p' to resume.")


        # If tracking is active, update tracker
        if is_tracking and tracker is not None:
            success, bbox = tracker.update(zoomed_frame_for_tracking)

            if success:
                x, y, w_box, h_box = map(int, bbox)

                # In YOLO mode, check if tracked object still matches a detection
                if use_manual_selection:
                    matched_id = "manual"  # Special case for manual tracking
                else:
                    matched_id = find_matching_detection((x, y, w_box, h_box), detected_objects)

                if matched_id is not None:
                    # Update based on mode
                    if use_manual_selection:
                        # Manual mode - just use tracker bbox
                        selected_bbox = (x, y, w_box, h_box)
                        tracking_lost_frames = 0
                        class_label = "TRACKING (Manual)"
                    else:
                        # YOLO mode - object is still detected by YOLO, use YOLO bbox
                        selected_object_id = matched_id
                        selected_bbox = detected_objects[matched_id]['bbox_original']
                        class_label = f"TRACKING: {detected_objects[matched_id]['class_name']}"

                        # Re-initialize tracker with YOLO detection
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(zoomed_frame_for_tracking, selected_bbox)
                        tracking_lost_frames = 0

                    # --- Full-screen zoom on the object using current zoom_factor ---
                    x_obj, y_obj, w_obj, h_obj = selected_bbox
                    obj_zoom_coords = (x_obj, y_obj, x_obj + w_obj, y_obj + h_obj)

                    object_zoomed_view, crop_coords = crop_and_zoom(
                        zoomed_frame_for_tracking,
                        obj_zoom_coords,
                        current_zoom_factor
                    )


                    if object_zoomed_view is not None and object_zoomed_view.size > 0:
                        # Resize object view to full window
                        h_disp, w_disp = display_frame.shape[:2]
                        object_zoomed_view_resized = cv2.resize(object_zoomed_view, (w_disp, h_disp))

                        # Compute bbox position inside the cropped view
                        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                        bx1 = x_obj - crop_x1
                        by1 = y_obj - crop_y1
                        bx2 = bx1 + w_obj
                        by2 = by1 + h_obj

                        # Scale bbox coords to full-screen
                        scale_x_zoom = w_disp / object_zoomed_view.shape[1]
                        scale_y_zoom = h_disp / object_zoomed_view.shape[0]

                        sx1 = int(bx1 * scale_x_zoom)
                        sy1 = int(by1 * scale_y_zoom)
                        sx2 = int(bx2 * scale_x_zoom)
                        sy2 = int(by2 * scale_y_zoom)

                        display_frame = object_zoomed_view_resized

                        margin = 40  # How close to edge before stopping

                        if (sx1 <= margin or sy1 <= margin or
                            sx2 >= display_frame.shape[1] - margin or
                            sy2 >= display_frame.shape[0] - margin):

                            print("Object near boundary → Forcing final zoom + pause")

                            # Force final zoom step (2.5x)
                            zoom_sequence_active = False
                            zoom_step_index = len(zoom_steps) - 1
                            zoom_factor = zoom_steps[zoom_step_index]

                            # Pause the video
                            paused = True
                            attack_mode = True
                        # ----- DRAW TEXT IN CENTER OF BBOX -----

                        label = "ATTACK" if attack_mode else class_label

                        # Center of bounding box
                        cx = (sx1 + sx2) // 2
                        cy = (sy1 + sy2) // 2

                        # Get text size
                        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                        # Text position (centered)
                        text_x = cx - text_w // 2
                        text_y = cy + text_h // 2

                        # Draw bbox corners
                        draw_corner_bbox(display_frame, (sx1, sy1, sx2, sy2), (0, 0, 255), "", thickness=3)

                        # Draw centered text
                        cv2.putText(display_frame, label, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                else:
                    # Object not found in YOLO detections (only applies in YOLO mode)
                    tracking_lost_frames += 1

                    if tracking_lost_frames >= MAX_LOST_FRAMES:
                        # Stop tracking - object is gone
                        is_tracking = False
                        tracker = None
                        selected_bbox = None
                        selected_object_id = None
                        tracking_lost_frames = 0
                        zoom_sequence_active = False
                        print("Object disappeared from detections. Click to select new target.")
                        cv2.putText(display_frame, "Object Lost - Click to select new target",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        # Still trying to track (uncertain state)
                        selected_bbox = (x, y, w_box, h_box)
                        scale_x = display_frame.shape[1] / zoomed_frame_for_tracking.shape[1]
                        scale_y = display_frame.shape[0] / zoomed_frame_for_tracking.shape[0]

                        x_scaled = int(x * scale_x)
                        y_scaled = int(y * scale_y)
                        w_scaled = int(w_box * scale_x)
                        h_scaled = int(h_box * scale_y)

                        bbox_coords = (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)

                        # Draw in YELLOW to show uncertain tracking
                        draw_corner_bbox(display_frame, bbox_coords, (0, 255, 255),
                                         f"UNCERTAIN ({MAX_LOST_FRAMES - tracking_lost_frames})", thickness=3)

                        cv2.putText(display_frame,
                                    f"Warning: Object not detected ({tracking_lost_frames}/{MAX_LOST_FRAMES})",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Tracker failed
                tracking_lost_frames += 1

                if tracking_lost_frames >= MAX_LOST_FRAMES:
                    is_tracking = False
                    tracker = None
                    selected_bbox = None
                    selected_object_id = None
                    tracking_lost_frames = 0
                    zoom_sequence_active = False
                    print("Tracking failed. Click to select new target.")

        # Draw all YOLO detections in GREEN (except tracked one) when not zooming on it
        if not use_manual_selection and not is_tracking:
            for idx, obj in enumerate(detected_objects):
                draw_corner_bbox(display_frame, obj['bbox_scaled'], (0, 255, 0),
                                 f"{obj['class_name']} {obj['confidence']:.2f}")

        # Show status message
        if not is_tracking:
            if use_manual_selection:
                cv2.putText(display_frame, "MANUAL MODE: Draw a box around target object",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if len(detected_objects) > 0:
                    cv2.putText(display_frame, f"YOLO: {len(detected_objects)} objects - Click to track",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "YOLO: No objects detected - Press 'm' for manual mode",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        # else:
        #     status_text = f"Tracking | Objects: {len(detected_objects)}"
        #     cv2.putText(display_frame, status_text, (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Show original frame with instruction
        display_frame = current_frame.copy()
        cv2.putText(display_frame, "Click and drag to select ZOOM REGION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display frame
    cv2.imshow("Manual Object Tracking", display_frame)

    # Handle key presses
    key = cv2.waitKey(playback_speed if not paused else 30) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("v"):
        # Reset everything
        is_tracking = False
        tracker = None
        selected_bbox = None
        selected_object_id = None
        zoom_region = None
        mode = "select_zoom"
        detected_objects = []
        use_manual_selection = False
        tracking_lost_frames = 0
        zoom_sequence_active = False
        zoom_step_index = 0
        zoom_factor = 2.0
        last_zoom_time = None
        paused = False
        print("Reset complete. Select zoom region again.")

    elif key == ord("r"):
        print("Resetting label to O and stopping current tracking.")

        # Stop tracking but KEEP zoom region
        is_tracking = False
        tracker = None
        selected_bbox = None
        selected_object_id = None

        # Reset zoom sequence
        zoom_sequence_active = False
        zoom_step_index = 0
        last_zoom_time = None
        paused = False

        # Reset attack mode + label
        attack_mode = False
        class_label = "O"

        # Keep region & mode exactly the same
        tracking_lost_frames = 0

        print("Ready for new target inside same selected region.")


    elif key == ord("m"):
        # Toggle between YOLO and manual mode
        if zoom_region is not None:
            use_manual_selection = not use_manual_selection
            detected_objects = []
            mode_name = "MANUAL" if use_manual_selection else "YOLO"
            print(f"Switched to {mode_name} mode")
            # Stop tracking when switching modes
            if is_tracking:
                is_tracking = False
                tracker = None
                selected_bbox = None
                selected_object_id = None
                tracking_lost_frames = 0
                zoom_sequence_active = False
                zoom_step_index = 0
                last_zoom_time = None
                print("Tracking stopped. Select a new target.")

    elif key == ord("+"):
        if not zoom_sequence_active:
            zoom_factor += 0.5
            print(f"Manual zoom factor: {zoom_factor}")
        else:
            print("Zoom sequence active - manual zoom disabled until it finishes.")

    elif key == ord("-"):
        if not zoom_sequence_active:
            zoom_factor = max(1.0, zoom_factor - 0.5)
            print(f"Manual zoom factor: {zoom_factor}")
        else:
            print("Zoom sequence active - manual zoom disabled until it finishes.")

    elif key == ord("f"):
        # Faster playback
        playback_speed = max(1, playback_speed - 10)
        print(f"Playback speed: {playback_speed}ms")

    elif key == ord("s"):
        # Slower playback
        playback_speed = min(500, playback_speed + 10)
        print(f"Playback speed: {playback_speed}ms")

    elif key == ord("p"):
        # Resume after auto-pause
        if paused:
            paused = False
            print("Playback resumed after pause.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
