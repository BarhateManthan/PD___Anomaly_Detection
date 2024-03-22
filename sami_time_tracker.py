import cv2
from ultralytics import YOLO
import math
import cvzone
import time
import torch



# Initialize time variables
pr_time = 0
curr_time = 0

# Open the video file
cap=cv2.VideoCapture(0)
previous_time=time.time()

# Set the width and height for video frames
des_width = 1280
des_height = 720

# Set the resolution of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, des_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, des_height)

# Load YOLO model
model = YOLO("yolov8x.pt")
# print(torch.cuda.is_available())
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')
# model.to("cuda:0")
# # if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#     # Check for available GPUs
#     model.to("cuda:0")  # Move the model to the GPU

# COCO classes
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

while True:
    # Read a frame from the video
    ret, img = cap.read()

    # if the frame is read
    if not ret:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Perform object detection using YOLO
    results = model.predict(img, stream=True)

    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx, cy = x2 - (w / 2), y2 - (h / 2)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            class_index = int(box.cls[0])
            current_time=time.time()
            elapsed_time=current_time-previous_time


            # Check if the detected object is a suitcase
            if coco_classes[class_index] == "suitcase":
                # Highlight the bounding box

                cvzone.cornerRect(img, (x1, y1, w, h), l=int(w / 10), rt=1, colorR=(255, 0, 0))
                # Display class name
                cvzone.putTextRect(img, f'{coco_classes[class_index]}', (max(0, x1), max(35, y1)),
                                   scale=0.7, thickness=1, offset=10, colorR=(0, 0, 255))
                cvzone.putTextRect(img, format(elapsed_time), (max(0, x1), max(35, y1)),
                                   scale=0.7, thickness=1, offset=10, colorR=(0, 0, 255))

    # Calculate frames per second (FPS)
    curr_time = time.time()
    fps = 1 / (curr_time - pr_time)
    pr_time = curr_time

    # Display FPS on the image
    cv2.putText(img, f"{str(int(fps))}fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (225, 0, 0), 3)

    # Show the final image
    cv2.imshow("Object Detection", img)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture
# close all windows
cap.release()
cv2.destroyAllWindows()