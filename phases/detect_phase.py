"""

Use YOLO to detect parcels bbox in ROI (RGB - 2D) -> list of detections

Output format (list of dict):
[
    {
        "image_path": str,
        "center": [u, v],
        "bbox_xyxy": [x1, y1, x2, y2],
        "polygon": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]],
        "conf": float,
        "label": str
    }, ...
]
"""

import os
import json
from typing import List, Dict, Any

from ultralytics import YOLO
from Code.config import *
from Code.file_utils import get_file_list
from Code.finetune_yolo.model_utils import detect_in_area_batch

YOLO_CKPT = "/Users/cybercs/Documents/Competition/Code/finetune_yolo/checkpoint/parcel_yolo11l/weights/best.pt"


def run_detection_on_dir(rgb_dir: str, model_path: str = "yolo11l.pt", max_images: int = -1) -> List[Dict[str, Any]]:
    """
    Run YOLO detection on images in directory using ROI-based detection.
    """
    model = YOLO(YOLO_CKPT)
    image_paths = get_file_list(rgb_dir, max_images)
    
    # Use the reusable function from model_utils
    detections = detect_in_area_batch(model, image_paths, target_class='parcel-box')
    
    return detections


def save_detections_json(dets: List[Dict[str, Any]], out_json: str):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(dets, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    dets = run_detection_on_dir(RGB_TRAIN, model_path="yolo11l.pt", max_images=50)
    detect_path = os.path.join(os.path.dirname(__file__), "exp", "detections_train.json")
    print("Saving detections to:", detect_path)
    save_detections_json(dets, detect_path)