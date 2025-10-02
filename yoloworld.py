# import cv2
# from ultralytics import YOLO

# model = YOLO("yolov8s-worldv2.pt")

# custom_classes = ["person", "car", "dog", "bottle"]

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("❌ Error: Could not open camera.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Error: Failed to capture frame.")
#         break

#     results = model.predict(frame, classes=custom_classes, conf=0.4, verbose=False)

#     annotated_frame = results[0].plot()

#     cv2.imshow("YOLO-World Real-Time Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

model = YOLO("yolov8s-worldv2.pt")

# Use COCO class IDs instead of names
custom_classes = [0, 2, 16, 39]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    results = model.predict(frame, classes=custom_classes, conf=0.4, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO-World Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
