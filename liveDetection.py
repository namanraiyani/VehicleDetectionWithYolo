import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

model = YOLO('best.pt')
cap = cv2.VideoCapture('test5.mp4')

object_paths = defaultdict(lambda: deque(maxlen=30)) 
smooth_paths = defaultdict(list)  
counted_objects = set() 

object_counts = {
    'Cyclist': 0,
    'Truck': 0,
    'Car': 0,
    'DontCare': 0,
    'Misc': 0,
    'Pedestrian': 0,
    'Tram': 0,
    'Van': 0,
    'Person_sitting': 0
}
colors = {
    'Cyclist': (0, 255, 0),
    'Truck': (255, 0, 0),
    'Car': (0, 0, 255),
    'DontCare': (128, 128, 128),
    'Misc': (0, 255, 255),
    'Pedestrian': (255, 0, 255),
    'Tram': (255, 255, 0),
    'Van': (128, 0, 0),
    'Person_sitting': (0, 128, 128)
}

def smooth_path(points, window_size=5):
    if len(points) < window_size:
        return points
    
    points = np.array(points)
    kernel = np.ones(window_size) / window_size
    x_smooth = np.convolve(points[:, 0], kernel, mode='valid')
    y_smooth = np.convolve(points[:, 1], kernel, mode='valid')
    
    return list(zip(x_smooth.astype(int), y_smooth.astype(int)))

fps_counter = 0
fps_start_time = cv2.getTickCount()
fps = 0 
cv2.namedWindow('Object Detection and Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection and Tracking', 800, 600)  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    fps_counter += 1
    time_elapsed = (cv2.getTickCount() - fps_start_time) / cv2.getTickFrequency()
    if time_elapsed >= 1.0:
        fps = fps_counter / time_elapsed  
        fps_counter = 0
        fps_start_time = cv2.getTickCount()  
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()
        for box, track_id, cl in zip(boxes, track_ids, cls):
            x1, y1, x2, y2 = box.astype(int)
            track_id = int(track_id)
            class_name = model.names[int(cl)]
            color = colors[class_name]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name} #{track_id}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            object_paths[track_id].append(center)
            object_key = (track_id, class_name)
            if object_key not in counted_objects:
                object_counts[class_name] += 1
                counted_objects.add(object_key)
            
            if len(object_paths[track_id]) > 1:
                smooth_path_points = smooth_path(list(object_paths[track_id]), window_size=10)
                if len(smooth_path_points) > 1:
                    for i in range(len(smooth_path_points) - 1):
                        pt1 = smooth_path_points[i]
                        pt2 = smooth_path_points[i + 1]
                        cv2.line(frame, pt1, pt2, color, 2)
        
        current_ids = set(track_ids)
        remove_ids = []
        for path_id in object_paths:
            if path_id not in current_ids:
                remove_ids.append(path_id)
        for path_id in remove_ids:
            del object_paths[path_id]
    
    y_offset = 30
    for class_name, count in object_counts.items():
        text = f'{class_name}: {count}'
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_name], 2)
        y_offset += 25

    fps_text = f"FPS: {fps:.2f}"  
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    frame_resized = cv2.resize(frame, (800, 600))  

    cv2.imshow('Object Detection and Tracking', frame_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()