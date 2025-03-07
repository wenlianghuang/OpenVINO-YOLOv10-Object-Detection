import cv2
#from yolov10 import YOLOv10, draw_detections
from yolov10.yolov10_openvino import YOLOv10_openvino
from yolov10.utils import draw_detections
# Initialize yolov10 object detector
model_path = "models/yolov10l.onnx"
detector = YOLOv10_openvino(model_path, conf_thres=0.6)

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect Objects
    class_ids, boxes, confidences = detector(frame)

    # Draw detections
    combined_img = draw_detections(frame, boxes, confidences, class_ids)

    # Display the resulting frame
    cv2.imshow("Detected Objects", combined_img)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()