#YOLO Vehicle Detection on KITTI Dataset

Hey there! This project is all about using the YOLO (You Only Look Once) model to detect vehicles in images and videos from the KITTI dataset.
**
What’s Going On?**
Setup: We start by importing some handy libraries like NumPy, Pandas, and Ultralytics YOLO. These help us with data handling and model training.

Dataset Paths: We define where our training and validation data is located and list the vehicle types we want to detect ('Cyclist', 'Truck', 'Car', 'DontCare', 'Misc', 'Pedestrian', 'Tram', 'Van', 'Person_sitting').

Temporary YAML File: We create a temporary YAML file that stores our dataset paths and class names. This is needed for training the model.

Training the Model: We initialize the YOLOv8 model and train it using our dataset. The model goes through 200 epochs .

Making Predictions: After training, we can use the model to detect vehicles in new images or videos. The results are saved automatically.

![image](https://github.com/user-attachments/assets/4f80f98a-631d-456b-a23f-fc2a9804decb)

![000013](https://github.com/user-attachments/assets/934ec252-c37a-4c20-80cf-daf7064ffdf0)
