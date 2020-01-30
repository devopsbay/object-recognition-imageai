import os

import cv2
from matplotlib import pyplot as plt

from imageai.Detection import VideoObjectDetection

execution_path = os.getcwd()

color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow',
               'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold',
               'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry',
               'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt',
               'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock',
               'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey',
               'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen',
               'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki',
               'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood',
               'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown',
               'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver',
               'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink',
               'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple',
               'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue',
               'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock',
               'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream',
               'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew',
               'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow',
               'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}

resized = False


def forSecond(frame_number, output_arrays, count_arrays, average_count, returned_frame):
    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    counter = 0

    for eachItem in average_count:
        counter += 1
        labels.append(eachItem + " = " + str(average_count[eachItem]))
        sizes.append(average_count[eachItem])
        this_colors.append(color_index[eachItem])

    plt.subplot(1, 2, 1)
    plt.title("Second : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Analysis: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    plt.pause(0.01)


def detect_objects_yolo(filename, fps=30):
    # 0,1 is recording from the camera
    camera = cv2.VideoCapture(0)
    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(os.getcwd(), "models/yolo.h5"))
    detector.loadModel()
    plt.show()
    video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                                 output_file_path=filename,
                                                 frames_per_second=fps,
                                                 log_progress=True,
                                                 #per_second_function=forSecond,
                                                 return_detected_frame=True,
                                                 minimum_percentage_probability=40)

    camera.release()  # Close the window / Release webcam
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def detect_objects_resnet(filename, fps=30):
    # 0,1 is recording from the camera
    camera = cv2.VideoCapture(0)
    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(os.getcwd(), "models/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    plt.show()
    video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                                 output_file_path=filename,
                                                 frames_per_second=fps,
                                                 log_progress=True,
                                                 per_second_function=forSecond,
                                                 return_detected_frame=True,
                                                 minimum_percentage_probability=40)

    camera.release()  # Close the window / Release webcam
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filename = 'output/detekcja'
    detect_objects_resnet(filename)
