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
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)

    # Draw label
    cv2.putText(frame, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def crop_and_zoom(frame, bbox, zoom_factor):
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
    
    return zoomed_frame

def embed_zoomed_view(main_frame, zoomed_frame, position='top_right'):
    if zoomed_frame.size == 0:
        return main_frame
    
    # Resize zoomed frame to a fixed size
    zoomed_frame_resized = cv2.resize(zoomed_frame, (250, 250))
    
    # Determine placement
    if position == 'top_right':
        x_offset = main_frame.shape[1] - zoomed_frame_resized.shape[1] - 10
        y_offset = 10
    
    # Add a border to the zoomed frame
    bordered_zoomed = cv2.copyMakeBorder(
        zoomed_frame_resized, 
        2, 2, 2, 2, 
        cv2.BORDER_CONSTANT, 
        value=(0, 255, 0)  # Green border
    )
    
    # Region of interest in main frame
    roi = main_frame[y_offset:y_offset+bordered_zoomed.shape[0], 
                     x_offset:x_offset+bordered_zoomed.shape[1]]
    
    # Create a mask of the zoomed frame and its inverse
    zoomed_gray = cv2.cvtColor(bordered_zoomed, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(zoomed_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Black-out the area of zoomed frame in ROI
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # Take only region of zoomed frame
    zoomed_fg = cv2.bitwise_and(bordered_zoomed, bordered_zoomed, mask=mask)
    
    # Put zoomed frame in ROI and modify the main frame
    dst = cv2.add(roi_bg, zoomed_fg)
    main_frame[y_offset:y_offset+dst.shape[0], 
               x_offset:x_offset+dst.shape[1]] = dst
    
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

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, selected_bbox, tracker, is_tracking
    global zoom_region, mode, zoomed_frame_display, detected_objects, zoomed_frame_for_tracking
    global use_manual_selection, selected_object_id, tracking_lost_frames
    
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
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(zoomed_frame_for_tracking, selected_bbox)
                    is_tracking = True
                    tracking_lost_frames = 0
                    print(f"Manual tracking started - bbox: {selected_bbox}")
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
print("Step 2a: YOLO will detect objects - CLICK on any detected object to track")
print("Step 2b: OR press 'm' to switch to MANUAL mode and draw box around target")
print("- Press 'm' to toggle between YOLO and Manual mode")
print("- Press 'r' to reset and start over")
print("- Press '+' to increase object zoom")
print("- Press '-' to decrease object zoom")
print("- Press 'f' for faster playback")
print("- Press 's' for slower playback")
print("- Press 'q' to quit")

zoomed_frame_display = current_frame.copy()
zoomed_frame_for_tracking = current_frame.copy()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video processing completed.")
        break
    
    current_frame = frame.copy()
    
    # If zoom region is selected, extract that region
    if zoom_region is not None:
        x1, y1, x2, y2 = zoom_region
        zoomed_frame_for_tracking = frame[y1:y2, x1:x2].copy()
        
        # Resize to full screen for better visibility
        h, w = frame.shape[:2]
        zoomed_frame_display = cv2.resize(zoomed_frame_for_tracking, (w, h))
        
        display_frame = zoomed_frame_display.copy()
        
        # Always run YOLO detection on the zoomed frame
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
        
        # If tracking is active, update tracker
        if is_tracking and tracker is not None:
            success, bbox = tracker.update(zoomed_frame_for_tracking)
            
            if success:
                x, y, w_box, h_box = map(int, bbox)
                
                # In YOLO mode, check if tracked object still matches a detection
                # In manual mode, just use the tracker result
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
                    else:
                        # YOLO mode - object is still detected by YOLO, use YOLO bbox
                        selected_object_id = matched_id
                        selected_bbox = detected_objects[matched_id]['bbox_original']
                        
                        # Re-initialize tracker with YOLO detection
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(zoomed_frame_for_tracking, selected_bbox)
                        tracking_lost_frames = 0
                    
                    # Scale for display
                    x_obj, y_obj, w_obj, h_obj = selected_bbox
                    scale_x = display_frame.shape[1] / zoomed_frame_for_tracking.shape[1]
                    scale_y = display_frame.shape[0] / zoomed_frame_for_tracking.shape[0]
                    
                    x_scaled = int(x_obj * scale_x)
                    y_scaled = int(y_obj * scale_y)
                    w_scaled = int(w_obj * scale_x)
                    h_scaled = int(h_obj * scale_y)
                    
                    bbox_coords = (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)
                    
                    # Draw tracked object in RED
                    if use_manual_selection:
                        draw_corner_bbox(display_frame, bbox_coords, (0, 0, 255), 
                                       "TRACKING (Manual)", thickness=3)
                    else:
                        draw_corner_bbox(display_frame, bbox_coords, (0, 0, 255), 
                                       f"TRACKING: {detected_objects[matched_id]['class_name']}", thickness=3)
                    
                    # Create zoomed view
                    obj_zoom_coords = (x_obj, y_obj, x_obj + w_obj, y_obj + h_obj)
                    object_zoomed_view = crop_and_zoom(zoomed_frame_for_tracking, obj_zoom_coords, zoom_factor)
                    
                    if object_zoomed_view is not None and object_zoomed_view.size > 0:
                        display_frame = embed_zoomed_view(display_frame, object_zoomed_view)
                else:
                    # Object not found in YOLO detections (only applies in YOLO mode)
                    # In manual mode, this branch won't be reached
                    tracking_lost_frames += 1
                    
                    if tracking_lost_frames >= MAX_LOST_FRAMES:
                        # Stop tracking - object is gone
                        is_tracking = False
                        tracker = None
                        selected_bbox = None
                        selected_object_id = None
                        tracking_lost_frames = 0
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
                        
                        cv2.putText(display_frame, f"Warning: Object not detected ({tracking_lost_frames}/{MAX_LOST_FRAMES})", 
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
                    print("Tracking failed. Click to select new target.")
        
        # Draw all YOLO detections in GREEN (except tracked one)
        if not use_manual_selection:
            for idx, obj in enumerate(detected_objects):
                if idx == selected_object_id and is_tracking:
                    continue  # Skip drawing - already drawn in RED above
                
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
        else:
            status_text = f"Tracking | Objects: {len(detected_objects)}"
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Show original frame with instruction
        display_frame = current_frame.copy()
        cv2.putText(display_frame, "Click and drag to select ZOOM REGION", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Display frame
    cv2.imshow("Manual Object Tracking", display_frame)
    
    # Handle key presses
    key = cv2.waitKey(playback_speed) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
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
        print("Reset complete. Select zoom region again.")
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
                print("Tracking stopped. Select a new target.")
    elif key == ord("+"):
        # Increase zoom on tracked object
        zoom_factor += 0.5
        print(f"Object zoom factor: {zoom_factor}")
    elif key == ord("-"):
        # Decrease zoom (minimum 1.0)
        zoom_factor = max(1.0, zoom_factor - 0.5)
        print(f"Object zoom factor: {zoom_factor}")
    elif key == ord("f"):
        # Faster playback
        playback_speed = max(1, playback_speed - 10)
        print(f"Playback speed: {playback_speed}ms")
    elif key == ord("s"):
        # Slower playback
        playback_speed = min(500, playback_speed + 10)
        print(f"Playback speed: {playback_speed}ms")

# Cleanup
cap.release()
cv2.destroyAllWindows()