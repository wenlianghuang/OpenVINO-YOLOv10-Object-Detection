import cv2
from imread_from_url import imread_from_url

#from yolov10 import YOLOv10, draw_detections
from yolov10.yolov10_openvino import YOLOv10_openvino
from yolov10.utils import draw_detections

#model_path = "models/yolov10l.onnx"
model_path = "models/yolov10n.xml"
#detector = YOLOv10(model_path, conf_thres=0.6)
detector = YOLOv10_openvino(model_path, conf_thres=0.6)
# Read image
#img_url = "https://github.com/ibaiGorordo/ONNX-YOLOv10-Object-Detection/blob/assets/assets/test.png?raw=true"
#img = imread_from_url(img_url)
# Read image from local file
img_path = './coco_bike.jpg'
img = cv2.imread(img_path)
# Detect Objects
class_ids, boxes, confidences = detector(img)

# Draw detections
combined_img = draw_detections(img, boxes, confidences, class_ids)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.waitKey(0)
