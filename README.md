# YOLO Vehicle Detection on KITTI Dataset

This project is all about using the YOLO (You Only Look Once) v8 model to detect vehicles in images and videos from the KITTI dataset.

The exact dataset can be found at - https://www.kaggle.com/datasets/namanraiyani/kitti-vehicle-dataset

Setup: We start by importing some handy libraries like NumPy, Pandas, and Ultralytics YOLO. These help us with data handling and model training.

Dataset Paths: We define where our training and validation data is located and list the vehicle types we want to detect ('Cyclist', 'Truck', 'Car', 'DontCare', 'Misc', 'Pedestrian', 'Tram', 'Van', 'Person_sitting').

Temporary YAML File: We create a temporary YAML file that stores our dataset paths and class names. This is needed for training the model.

Training the Model: We initialize the YOLOv8 model and train it using our dataset. The model goes through 200 epochs .

Making Predictions: After training, we can use the model to detect vehicles in new images or videos. The results are saved automatically.

The script liveDetection.py processes a video file (e.g., `test2.mp4`) in real-time using **OpenCV**'s
`cv2.VideoCapture()` function. The video is read frame by frame, and object detection and tracking
occur in each frame as follows:
1. Video Input: The video is opened using `cv2.VideoCapture('test2.mp4')`, which loads the
video file for processing. Each frame is read sequentially inside the `while cap.isOpened()` loop.
2. Object Detection: In each frame, YOLO (`model.track(frame)`) detects objects. The model
predicts the location and class of each object, returning bounding boxes, IDs, and class labels.
3. Real-Time Processing: For each detected object, a bounding box is drawn on the frame, and
the object is tracked across frames using a unique ID. This tracking data is stored and updated in
realtime.
4. FPS Calculation: The script tracks the frames per second (FPS) to measure performance,
ensuring that processing speed is displayed on the video feed.
5. Visualization: The processed frame, with bounding boxes, object IDs, labels, smooth paths,
and FPS display, is shown in a window using `cv2.imshow()`. This allows the user to see the detection
and tracking results in real-time as the video plays.
6. Exit Mechanism: The video feed continues playing until the user presses the 'q' key, which
breaks the loop and terminates the program.
This workflow allows for **live object detection and tracking** on a pre-recorded video, simulating a
real-time system by processing each frame as it's read from the file.

![image](https://github.com/user-attachments/assets/36b1a640-6667-4622-8cef-0791f4ed889f)

![image](https://github.com/user-attachments/assets/d698c088-31e1-49db-9e0f-4519dc34c4bc)



