# Import necessary libraries
import time
import cv2
import numpy as np
from openvino.runtime import Core

# Import utility function to check the model
from .utils import check_model

# Define the YOLOv10_openvino class
class YOLOv10_openvino:

    # Initialize the class with model path and confidence threshold
    def __init__(self, path: str, conf_thres: float = 0.2):

        # Set confidence threshold
        self.conf_threshold = conf_thres

        # Check if the model path is valid
        check_model(path)

        # Initialize OpenVINO model
        core = Core()
        self.model = core.read_model(model=path)
        self.compiled_model = core.compile_model(model=self.model, device_name="NPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Get model input and output details
        self.get_input_details()
        self.get_output_details()

    # Define the call method to detect objects in an image
    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.detect_objects(image)

    # Define the method to detect objects in an image
    def detect_objects(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        return self.process_output(outputs)

    # Define the method to prepare input image for the model
    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        # Convert image from BGR to RGB
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    # Define the method to perform inference on the input tensor
    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.compiled_model([input_tensor])[self.output_layer]

        # Print inference time
        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    # Define the method to process the output of the model
    def process_output(self, output):
        output = output.squeeze()
        boxes = output[:, :-2]
        confidences = output[:, -2]
        class_ids = output[:, -1].astype(int)

        # Apply confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return class_ids, boxes, confidences

    # Define the method to rescale boxes to original image dimensions
    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    # Define the method to get input details of the model
    def get_input_details(self):
        input_shape = self.input_layer.shape
        self.input_height = input_shape[2] if type(input_shape[2]) == int else 640
        self.input_width = input_shape[3] if type(input_shape[3]) == int else 640

    # Define the method to get output details of the model
    def get_output_details(self):
        pass