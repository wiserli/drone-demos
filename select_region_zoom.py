# =============================================================== manual + yoo ==================================================

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("drone_m_1280.pt")

# Initialize variables
zoom_region = None  # The area to zoom into
selected_bbox = None
tracker = None
zoom_factor = 2.0
is_tracking = False
drawing = False
start_point = None
playback_speed = 30
mode = "select_zoom"  # Modes: "select_zoom" or "detect_objects"
detected_objects = []  # Store detected objects in zoomed region
use_manual_selection = False  # Toggle between YOLO and manual selection

def draw_corner_bbox(frame, bbox, color, label):
    x1, y1, x2, y2 = map(int, bbox)
    
    # Corner length
    corner_length = min((x2 - x1), (y2 - y1)) // 10
    corner_thickness = 2

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

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, selected_bbox, tracker, is_tracking
    global zoom_region, mode, zoomed_frame_display, detected_objects, zoomed_frame_for_tracking
    global use_manual_selection
    
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
                    
                    # Initialize tracker
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(zoomed_frame_for_tracking, selected_bbox)
                    is_tracking = True
                    print(f"Manual tracking started - bbox: {selected_bbox}")
                else:
                    print("Selection too small. Please try again.")
        else:
            # YOLO mode: click on detected objects
            if event == cv2.EVENT_LBUTTONDOWN:
                for obj in detected_objects:
                    x1, y1, x2, y2 = obj['bbox_scaled']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected_bbox = obj['bbox_original']
                        
                        # Initialize tracker
                        tracker = cv2.TrackerMIL_create()
                        tracker.init(zoomed_frame_for_tracking, selected_bbox)
                        is_tracking = True
                        print(f"Tracking started on {obj['class_name']} - bbox: {selected_bbox}")
                        return

# Open video capture
cap = cv2.VideoCapture("12762043-hd_1280_720_60fps.mp4")

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
        
        # Run YOLO detection on the zoomed frame if not tracking yet
        if not is_tracking:
            if use_manual_selection:
                # Manual mode - show instruction
                cv2.putText(display_frame, "MANUAL MODE: Draw a box around target object", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # YOLO mode - run detection
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
                        
                        # Draw detection on display frame
                        draw_corner_bbox(display_frame, bbox_scaled, (0, 255, 0), 
                                       f"{class_name} {confidence:.2f}")
                    
                    cv2.putText(display_frame, f"YOLO: {len(detected_objects)} objects - Click to track or press 'm' for manual", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "YOLO: No objects detected - Press 'm' to switch to manual mode", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # If tracking is active, update tracker
        if is_tracking and tracker is not None:
            success, bbox = tracker.update(zoomed_frame_for_tracking)
            
            if success:
                # Convert bbox to (x1, y1, x2, y2) for drawing
                x, y, w_box, h_box = map(int, bbox)
                selected_bbox = (x, y, w_box, h_box)
                
                # Scale coordinates to display frame
                scale_x = display_frame.shape[1] / zoomed_frame_for_tracking.shape[1]
                scale_y = display_frame.shape[0] / zoomed_frame_for_tracking.shape[0]
                
                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)
                w_scaled = int(w_box * scale_x)
                h_scaled = int(h_box * scale_y)
                
                bbox_coords = (x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled)
                
                # Draw corner bbox
                draw_corner_bbox(display_frame, bbox_coords, (0, 0, 255), "Target")
                
                # Create even more zoomed view of the tracked object
                obj_zoom_coords = (x, y, x + w_box, y + h_box)
                object_zoomed_view = crop_and_zoom(zoomed_frame_for_tracking, obj_zoom_coords, zoom_factor)
                
                # Embed zoomed view
                if object_zoomed_view is not None and object_zoomed_view.size > 0:
                    display_frame = embed_zoomed_view(display_frame, object_zoomed_view)
                
                # Show info
                cv2.putText(display_frame, f"Tracking Target", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Tracking failed
                cv2.putText(display_frame, "Tracking Lost - Press 'r' to Reset", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                is_tracking = False
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
        zoom_region = None
        mode = "select_zoom"
        detected_objects = []
        use_manual_selection = False
        print("Reset complete. Select zoom region again.")
    elif key == ord("m"):
        # Toggle between YOLO and manual mode
        if not is_tracking and zoom_region is not None:
            use_manual_selection = not use_manual_selection
            detected_objects = []
            mode_name = "MANUAL" if use_manual_selection else "YOLO"
            print(f"Switched to {mode_name} mode")
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