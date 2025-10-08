from ultralytics import YOLO


import ultralytics
ultralytics.checks()

from config import *
from file_utils import get_file_list

# Load a pretrained YOLO11n model
model = YOLO("yolo11l-obb.pt")  # load an official model


for f_image in get_file_list(RGB_TRAIN, 10):
    results = model(f_image)
    for result in results:
        xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
        xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
        names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
        confs = result.obb.conf  # confidence score of each box

        print(names)
        print(xywhr)