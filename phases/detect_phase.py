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
from config import *
from file_utils import get_file_list


def run_detection_on_dir(rgb_dir: str, model_path: str = "yolo11l.pt", max_images: int = -1) -> List[Dict[str, Any]]:
    model = YOLO(model_path)
    detections: List[Dict[str, Any]] = []

    for f_image in get_file_list(rgb_dir, max_images):
        results = model(f_image)
        for result in results:
            boxes = result.boxes  # axis-aligned boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy  # [N, 4]
            confs = boxes.conf.tolist() if boxes.conf is not None else [1.0] * len(xyxy)
            clsi = boxes.cls.int() if boxes.cls is not None else []
            names = [result.names[c.item()] for c in clsi] if len(clsi) == len(xyxy) else ["object"] * len(xyxy)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                detections.append({
                    "image_path": f_image,
                    "center": [float(cx), float(cy)],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "polygon": [[float(px), float(py)] for px, py in poly],
                    "conf": float(confs[i]),
                    "label": names[i]
                })

    return detections


def save_detections_json(dets: List[Dict[str, Any]], out_json: str):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(dets, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    dets = run_detection_on_dir(RGB_TRAIN, model_path="yolo11l.pt", max_images=50)
    save_detections_json(dets, os.path.join("exp", "detections_train.json"))