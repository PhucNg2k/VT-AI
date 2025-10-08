"""

    From detected bboxes
    + identify bbox center
    + get only 'bbox CENTER' in ROI

    + get correspond depth value of each center

    + choose center that has lowest depth (means parcel is on topmost)
    + if different between depth value is 5 (depth must be in milimeters), they will be consider same height, then choose center farthest fro ROBOT_POS in euclide distance

    retur 1 bbox center coordinate in 2D

"""

import json
from typing import List, Dict, Any, Tuple
import os
import numpy as np

from config import ROI, HEIGHT_TOL_MM, ROBOT_POS
from coord_utils import in_roi, dist_from_robot, bbox_center_from_xywhr, bbox_center_from_polygon
from file_utils import get_depth_pixel


def select_topmost_detection(dets: List[Dict[str, Any]], split: str = 'train') -> Tuple[Tuple[float, float], float, Dict[str, Any]]:
    """
    Return (center_uv, depth_mm, det_dict) according to rules.
    - Filter by ROI using center.
    - Retrieve depth at center.
    - Choose min depth (top-most). If within HEIGHT_TOL_MM, pick farthest from ROBOT_POS.
    """
    candidates = []
    for d in dets:
        if "center" in d and d["center"] is not None:
            cx, cy = d["center"]
        elif "xywhr" in d:
            cx, cy = bbox_center_from_xywhr(d["xywhr"])  # fallback
        elif "polygon" in d:
            cx, cy = bbox_center_from_polygon(d["polygon"])  # fallback
        else:
            continue

        if not in_roi((cx, cy), ROI):
            continue

        image_path = d.get("image_path", "")
        depth_mm = get_depth_pixel(image_path, (cx, cy), set=split)
        if depth_mm is None or float(depth_mm) <= 0:
            continue

        candidates.append({
            "center": (cx, cy),
            "depth": float(depth_mm),
            "dist_robot": dist_from_robot((cx, cy), ROBOT_POS),
            "det": d
        })

    if not candidates:
        return None, None, None

    # sort by depth ascending (smaller depth => closer to camera => top-most)
    candidates.sort(key=lambda x: x["depth"])  # ascending
    best = candidates[0]

    # group ties within HEIGHT_TOL_MM
    top_depth = best["depth"]
    ties = [c for c in candidates if abs(c["depth"] - top_depth) <= float(HEIGHT_TOL_MM)]
    if len(ties) == 1:
        chosen = ties[0]
    else:
        # choose farthest from robot
        chosen = max(ties, key=lambda x: x["dist_robot"])

    return chosen["center"], chosen["depth"], chosen["det"]


def load_detections_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    dets_path = os.path.join("exp", "detections_train.json")
    dets = load_detections_json(dets_path)
    center, depth_mm, det = select_topmost_detection(dets, split='train')
    print("Selected center:", center, "depth(mm):", depth_mm)