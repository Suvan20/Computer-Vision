import os
import cv2
import numpy as np

def detect_and_track_balls(video_path, output_video_path, output_txt_path):
    # Ensuring that output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    # Initialize variables for tracking
    ball_info = {}
    ball_colors = {
        'red': ([0, 100, 100], [10, 255, 255]),   # HSV format
        'green': ([35, 100, 100], [85, 255, 255]),
        'blue': ([100, 150, 0], [140, 255, 255]),
        'yellow': ([25, 150, 150], [35, 255, 255]),
        'white': ([0, 0, 200], [180, 20, 255])
    }
    
    quadrant_boundaries = [(0, frame_width//2, 0, frame_height//2),
                           (frame_width//2, frame_width, 0, frame_height//2),
                           (0, frame_width//2, frame_height//2, frame_height),
                           (frame_width//2, frame_width, frame_height//2, frame_height)]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get current timestamp in seconds
        
        for color, (lower, upper) in ball_colors.items():
            lower_bound = np.array(lower, dtype="uint8")
            upper_bound = np.array(upper, dtype="uint8")
            
            # Create a mask for the color
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Use Gaussian Blur to reduce noise
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Apply morphological operations to improve mask
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if 500 < cv2.contourArea(contour) < 5000:  # Filter by contour area to detect balls
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cx, cy = x + w//2, y + h//2
                    
                    for i, (x1, x2, y1, y2) in enumerate(quadrant_boundaries):
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            ball_id = (color, cx, cy)
                            if ball_id not in ball_info:
                                event_type = "Entry"
                                ball_info[ball_id] = timestamp
                            else:
                                event_type = "Exit"
                                ball_info[ball_id] = timestamp
                            
                            record = f"{timestamp:.2f}, Quadrant {i+1}, {color}, {event_type}"
                            with open(output_txt_path, 'a') as f:
                                f.write(record + '\n')
                            cv2.putText(frame, f"{event_type} {timestamp:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            break
        
        out.write(frame)
    
    cap.release()
    out.release()

# Input and Output paths
video_path = r'C:\Users\Suvan\Desktop\SecqurAlse\AI Assignment video.mp4'  # Input video path
output_video_path = r'C:\Users\Suvan\Desktop\SecqurAlse\output_video.avi'  # Output video path
output_txt_path = r'C:\Users\Suvan\Desktop\SecqurAlse\Output_data.txt'  # Output txt file path
detect_and_track_balls(video_path, output_video_path, output_txt_path)
