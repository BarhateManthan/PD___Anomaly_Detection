import cv2
import cvzone
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)
previous_time = int(time.time())

# Initialize face detection
face_detect = mp.solutions.face_detection.FaceDetection()

# Define variables to store coordinates at 0th and 4th frames
coords_0th_frame = None
coords_4th_frame = None

def facedetect(face, x1, y1, track):

    for element in face:
        print(element)
        f_dict = {'x': x1, 'y': y1, 'elapsed_time': track}  # Store data for a single face
        print(f_dict)

    return f_dict

def suspicious(x11,x22):
    print("more than 4 sec")
    print(x11, x22)
    diff=abs(x22-x11)
    return diff

def detect(diff):
    if(diff>abs(100)):
        print("difference:",diff)
        print("non suspicious--------------------")


def ini_cor(x1):
    return x1

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Convert frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detect.process(rgb_frame)

    # Check if any faces are detected
    if (results.detections):
        print("Faces detected")

        # Iterate over each detected face
        for detection in results.detections:
            # Get bounding box coordinates
            ih, iw, _ = frame.shape
            bboxC = detection.location_data.relative_bounding_box
            x1 = bboxC.xmin * iw
            y1 = bboxC.ymin * ih
            print(x1, y1)

            bbox = int(x1), int(y1), int(bboxC.width * iw), int(bboxC.height * ih)
            elapsed_time = int(time.time()) - previous_time

            # Draw rectangle around the face
            cvzone.cornerRect(frame, bbox, int(30), int(5), 1, (0, 0, 0), (255, 255, 255))

            cvzone.putTextRect(frame, format(elapsed_time), (10, 10), 0.5, 1, (255, 255, 255), (255, 0, 0),
                               cv2.FONT_HERSHEY_PLAIN, 10, None, (0, 255, 0))
            cv2.circle(frame, (int(bboxC.xmin * iw), int(bboxC.ymin * ih)), int(5), (0, 255, 255), 5)

            # Call facedetect function to process face detection data
            facedetect(results.detections, x1, y1, elapsed_time)
            i = int(elapsed_time)


            if elapsed_time == 0:
                coords_0th_frame = ini_cor(x1)
            if elapsed_time== 4:
                coords_4th_frame = ini_cor(x1)
                k=suspicious(coords_4th_frame,coords_0th_frame)
                detect(k)


    else:
        print("No faces detected")

    # Display the frame with rectangles
    cv2.imshow("Video", frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
