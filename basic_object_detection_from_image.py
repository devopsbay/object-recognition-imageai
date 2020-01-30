from datetime import datetime
import os

import cv2

from imageai.Detection import ObjectDetection
from imageai.Detection.Custom import CustomObjectDetection

probability = 30


def detect_objects_resnet(filename):

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(os.getcwd(), "models/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=filename,
                                                 output_image_path=os.path.join('output/images/', datetime.now().strftime("%H_%M_%S")+"_resnet_detected.jpg"),
                                                 minimum_percentage_probability=probability)

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        print("--------------------------------")
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def detect_objects_yolo(filename):

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(os.getcwd(), "models/yolo.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=filename,
                                                 output_image_path=os.path.join('output/images/', datetime.now().strftime("%H_%M_%S")+"_yolo_detected.jpg"),
                                                 minimum_percentage_probability=probability,
                                                 extract_detected_objects=False)
    print("---------YOLO------------------")
    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        print("--------------------------------")
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def detect_objects_yolo_custom(filename):

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("examples/specjale/models/detection_model-ex-008--loss-0023.046.h5")
    detector.setJsonPath("examples/specjale/json/detection_config.json")
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=filename,
                                                 output_image_path=os.path.join('output/images/', datetime.now().strftime("%H_%M_%S")+"_yolo_detected.jpg"),
                                                 minimum_percentage_probability=probability,
                                                 extract_detected_objects=False)

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        print("--------------------------------")
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filename = "examples/kuchnia4.jpg"
    #detect_objects_resnet(filename)
    #detect_objects_yolo(filename)
    detect_objects_yolo_custom(filename)
