import cv2
from ultralytics import YOLO
import math
import time
import torch
import numpy as np

# time var
pr_time = 0
curr_time = 0

# vid file
vid_path = r"C:\Users\manth\Downloads\Easy and cute ways to identify your luggage.mp4"
cap = cv2.VideoCapture(vid_path)

# width and height of vid frame
des_width = 1280
des_height = 720

# Set the resolution of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, des_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, des_height)

# Load YOLO model
model = YOLO("yolov8s.pt")
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

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

    # Apply background subtraction to get the foreground mask
    fgmask = fgbg.apply(img)

    # Perform object detection using YOLO on the original image
    results = model.predict(img, stream=True)

    # Create a black background
    black_bg = np.zeros_like(img)

    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx, cy = x2 - (w / 2), y2 - (h / 2)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            class_index = int(box.cls[0])

            # Check if the detected object is a suitcase
            if coco_classes[class_index] == "suitcase":
                # Draw a white rectangle on the black background
                cv2.rectangle(black_bg, (x1, y1), (x2, y2), (255, 255, 255), thickness=cv2.FILLED-3)

    # Apply the foreground mask to the black background
    img_with_mask = cv2.bitwise_and(black_bg, black_bg, mask=fgmask)

    # enhance visibility by morphology
    kernel = np.ones((5, 5), np.uint8)
    img_with_mask = cv2.morphologyEx(img_with_mask, cv2.MORPH_CLOSE, kernel)
    img_with_mask = cv2.morphologyEx(img_with_mask, cv2.MORPH_OPEN, kernel)

    # Display the original image with bounding boxes, the masked image, and the enhanced mask
    cv2.imshow("Object Detection", img)
    cv2.imshow("Foreground Mask", img_with_mask)

    # Calculate (FPS)
    curr_time = time.time()
    fps = 1 / (curr_time - pr_time)
    pr_time = curr_time

    # Display FPS on the image
    cv2.putText(img, f"{str(int(fps))}fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (225, 0, 0), 3)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
