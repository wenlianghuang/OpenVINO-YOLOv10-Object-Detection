import os
import cv2
import numpy as np
import tqdm
import requests

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

available_models = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"]


def download_model(url: str, path: str):
    print(f"Downloading model from {url} to {path}")
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024 * 1024), total=total_length // (1024 * 1024),
                               bar_format='{l_bar}{bar:10}'):
            if chunk:
                f.write(chunk)
                f.flush()


def check_model(model_path: str):
    if os.path.exists(model_path):
        return

    model_name = os.path.basename(model_path).split('.')[0]
    if model_name not in available_models:
        raise ValueError(f"Invalid model name: {model_name}")
    url = f"https://github.com/THU-MIG/yolov10/releases/download/v1.1/{model_name}.onnx"
    download_model(url, model_path)


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    # Make a copy of the image to draw detections on
    det_img = image.copy()

    # Get the height and width of the image
    img_height, img_width = image.shape[:2]
    
    # Calculate font size and text thickness based on image dimensions
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw masks on the image
    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        # Get the color for the current class
        color = colors[class_id]

        # Draw the bounding box
        draw_box(det_img, box, color)

        # Create the label and caption for the detection
        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        
        # Draw the text label on the image
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    # Return the image with detections drawn
    return det_img


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    # Extract coordinates from the bounding box
    x1, y1, x2, y2 = box.astype(int)
    
    # Get the text size
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    
    # Adjust text height
    th = int(th * 1.2)

    # Draw a filled rectangle for the text background
    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

    # Put the text on the image
    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness,
                       cv2.LINE_AA)


def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    # Make a copy of the image to draw masks on
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        # Get the color for the current class
        color = colors[class_id]

        # Extract coordinates from the bounding box
        x1, y1, x2, y2 = box.astype(int)

        # Draw a filled rectangle in the mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    # Blend the mask image with the original image
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
