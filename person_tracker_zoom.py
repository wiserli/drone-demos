import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("drone_m_1280.pt")

# Initialize variables
selected_person = None
zoom_factor = 2.0  # How much to zoom in

def draw_corner_bbox(frame, bbox, color, label):
    x1, y1, x2, y2 = map(int, bbox)
    
    # Corner length (adjust as needed)
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
    global selected_person, detections
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is inside any bounding box
        for box, track_id, cls in zip(detections['boxes'], detections['track_ids'], detections['classes']):
            x1, y1, x2, y2 = map(int, box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Toggle selection
                selected_person = track_id if selected_person != track_id else None
                print(f"Selected person ID: {selected_person}")
                return

# Open video capture
cap = cv2.VideoCapture("12762043-hd_1280_720_60fps.mp4")

# Create window
cv2.namedWindow("Person Tracking")
cv2.setMouseCallback("Person Tracking", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video processing completed.")
        break
    
    # Track objects
    results = model.track(frame, persist=True)
    
    # Store detections for mouse callback
    detections = {
        'boxes': results[0].boxes.xyxy.cpu(),
        'track_ids': results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [],
        'classes': results[0].boxes.cls.int().cpu().tolist()
    }
    
    # Create a copy of the frame for drawing
    display_frame = frame.copy()
    
    # Process detections
    zoomed_view = None
    if detections['track_ids']:
        for box, track_id, cls in zip(detections['boxes'], detections['track_ids'], detections['classes']):
            # Get class name
            class_name = model.names[cls]
            
            # Determine box color based on selection
            bbox_color = (0, 0, 255) if track_id == selected_person else (0, 255, 0)
            
            # Draw custom corner bounding box with label
            draw_corner_bbox(display_frame, box, bbox_color, f"#{track_id} {class_name}")
            
            # If this is the selected person, create zoomed view
            if track_id == selected_person:
                zoomed_view = crop_and_zoom(frame, box, zoom_factor)
    
    # Embed zoomed view if a person is selected
    if selected_person is not None and zoomed_view is not None:
        display_frame = embed_zoomed_view(display_frame, zoomed_view)
    
    # Display frame
    cv2.imshow("Person Tracking", display_frame)
    
    # Exit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("+"):
        # Increase zoom
        zoom_factor += 0.5
    elif key == ord("-"):
        # Decrease zoom (minimum 1.0)
        zoom_factor = max(1.0, zoom_factor - 0.5)

# Cleanup
cap.release()
cv2.destroyAllWindows()

