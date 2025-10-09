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
import cv2

from Code.config import ROI, HEIGHT_TOL_MM, ROBOT_POS
from Code.coord_utils import in_roi, dist_from_robot, bbox_center_from_xywhr, bbox_center_from_polygon
from Code.file_utils import get_depth_pixel


def select_topmost_per_image(dets: List[Dict[str, Any]], split: str = 'train') -> List[Tuple[str, Tuple[float, float], float, Dict[str, Any]]]:
    """
    For each image, select one detection by rules and return a list of
    (image_path, center_uv, depth_mm, det_dict).
    Rules per image:
    - Filter by ROI using center.
    - Retrieve depth at center.
    - Choose min depth (top-most). If within HEIGHT_TOL_MM, pick farthest from ROBOT_POS.
    """
    # Group detections by image
    
    rx, ry, rw, rh = ROI
    fallback_center = (float(rx + rw * 0.5), float(ry + rh * 0.5))
    
    by_image: Dict[str, List[Dict[str, Any]]] = {}
    
    for d in dets:
        img = d.get("image_path", "")
        if not img:
            continue
        by_image.setdefault(img, []).append(d)

    selected: List[Tuple[str, Tuple[float, float], float, Dict[str, Any]]] = []
    print("By image:", len(by_image))
    
    for image_path, det_list in by_image.items():
        candidates = []
        print("Det list:", len(det_list))

        for d in det_list:
            if not isinstance(d.get("center_uv"), (list, tuple)) or len(d["center_uv"]) != 2:
                print("Center not found:", d)
                continue
            cx, cy = float(d["center_uv"][0]), float(d["center_uv"][1])
            if not in_roi((cx, cy), ROI):
                #print("Center not in ROI:", d)
                continue

            depth_mm = get_depth_pixel(image_path, (cx, cy), set=split)
            
            if depth_mm is None or float(depth_mm) <= 0:
                print("Depth mm not found:", depth_mm)
                continue
            
            cand = {
                "center_uv": (cx, cy),
                "depth": float(depth_mm),
                "dist_robot": dist_from_robot((cx, cy), ROBOT_POS),
                "det": d
            }
            candidates.append(cand)

        if not candidates:
            # Fallback: no valid detections in ROI for this image.
            selected.append((image_path, fallback_center, 1000.0, {}))
            continue

        # sort by depth ascending (smaller depth => closer to camera => top-most)
        candidates.sort(key=lambda x: x["depth"])  # ascending
        best = candidates[0]

        # group ties within HEIGHT_TOL_MM
        top_depth = best["depth"]
        ties = [c for c in candidates if abs(c["depth"] - top_depth) <= float(HEIGHT_TOL_MM)]
        if len(ties) == 1:
            chosen = ties[0]
        else:
            # prioritize lower u (x) then lower v (y)
            chosen = sorted(ties, key=lambda x: (x["center_uv"][0], x["center_uv"][1]))[0]

        # (image, chosen_point)
        selected.append((image_path, chosen["center_uv"], chosen["depth"], chosen["det"]))

    return selected


def load_detections_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    dets_path = os.path.join(os.path.dirname(__file__), "exp", "detections_train.json")
    print("Loading detections from:", dets_path)
    dets = load_detections_json(dets_path)
    print("Loaded detections:", len(dets))

    selections = select_topmost_per_image(dets, split='train')
    print("Num images with selection:", len(selections))

    # visualize each selected detection on its corresponding image
    os.makedirs("exp", exist_ok=True)
    for image_path, center, depth_mm, det in selections:
        if det is None or center is None:
            continue
        if not (isinstance(det.get("bbox_xyxy", []), list) and len(det["bbox_xyxy"]) == 4):
            continue
        if not os.path.isfile(image_path):
            print("Image path not found:", image_path)
            continue
        img = cv2.imread(image_path)
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cx, cy = int(center[0]), int(center[1])
        # draw bbox and center
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        label = f"depth={depth_mm:.1f}mm"
        cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(os.path.dirname(__file__), "exp", f"vis_selected_{base}.png")
        cv2.imwrite(out_path, img)
        print("Saved visualization to:", out_path)