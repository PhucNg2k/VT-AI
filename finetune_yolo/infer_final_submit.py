import os
import csv
import glob
from typing import List

import cv2

try:
    from Code.config import RGB_TEST, RGB_TRAIN
    from Code.file_utils import get_file_list, get_depth_pixel
    from Code.phases.filter_phase import select_topmost_per_image
    from Code.camera_config import unproject_pixel_to_cam, COLOR_INTRINSIC
except ImportError:
    from config import RGB_TEST, RGB_TRAIN
    from file_utils import get_file_list, get_depth_pixel
    from phases.filter_phase import select_topmost_per_image
    from camera_config import unproject_pixel_to_cam, COLOR_INTRINSIC

from ultralytics import YOLO


def find_latest_checkpoint(checkpoint_root: str) -> str:
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_root}")
    subdirs = [os.path.join(checkpoint_root, d) for d in os.listdir(checkpoint_root)
               if os.path.isdir(os.path.join(checkpoint_root, d))]
    if not subdirs:
        raise FileNotFoundError("No runs found under checkpoint directory")
    latest = max(subdirs, key=os.path.getmtime)
    for w in [os.path.join(latest, 'weights', 'best.pt'), os.path.join(latest, 'weights', 'last.pt')]:
        if os.path.isfile(w):
            return w
    cand = glob.glob(os.path.join(latest, '**', '*.pt'), recursive=True)
    if not cand:
        raise FileNotFoundError("No .pt weights found in latest checkpoint run")
    return cand[0]


def run_and_export(output_csv: str) -> None:
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)

    img_paths: List[str] = get_file_list(RGB_TRAIN, -1)
    if not img_paths:
        print('No test images found at:', RGB_TRAIN)
        return

    rows: List[dict] = []

    for p in img_paths:
        # run inference (no need to read image for drawing; YOLO accepts path)
        results = model(p)

        # Collect detections in our common dict format for current image
        dets_for_filter = []
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy
            confs = boxes.conf.tolist() if boxes.conf is not None else [1.0] * len(xyxy)
            clsi = boxes.cls.int() if boxes.cls is not None else []
            names = [res.names[c.item()] for c in clsi] if len(clsi) == len(xyxy) else ["object"] * len(xyxy)
            for i in range(len(xyxy)):
                if i < len(names) and names[i] != 'parcel-box':
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                dets_for_filter.append({
                    "image_path": p,
                    "center": [cx, cy],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "conf": float(confs[i]) if i < len(confs) else 1.0,
                    "label": names[i]
                })

        if not dets_for_filter:
            # Ensure we get ROI-center fallback from filter
            print("GOT DEFAULT BBOX IN ROI")
            dets_for_filter = [{"image_path": p, "center": []}]

        selections = select_topmost_per_image(dets_for_filter, split='train')
        
        # There will be exactly one selection for current image path
        for img_path, center_uv, depth_sel, det_sel in selections:
            if img_path != p:
                continue
        
            # Compute 3D camera coordinates from center_uv and depth
            x_cam = y_cam = z_cam = 0.0
            if depth_sel is not None and float(depth_sel) > 0:
                u_f, v_f = float(center_uv[0]), float(center_uv[1]) # float center 
                z_m = float(depth_sel) / 1000.0 # meter scale for z
                xyz = unproject_pixel_to_cam(u_f, v_f, z_m, COLOR_INTRINSIC)
                if xyz is not None:
                    x_cam, y_cam, z_cam = float(xyz[0]), float(xyz[1]), float(xyz[2])

            # Compose CSV row
            rows.append({
                "image_filename": f"image_{os.path.basename(p)}",
                "x": f"{x_cam:.3f}",
                "y": f"{y_cam:.3f}",
                "z": f"{z_cam:.3f}",
                "Rx": 0,
                "Ry": 0,
                "Rz": 1
            })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename","x","y","z","Rx","Ry","Rz"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {output_csv}")


if __name__ == "__main__":
    out_csv = os.path.join(os.path.dirname(__file__), "submit", "Submit_train.csv")
    print(f"Saving to {out_csv}")
    run_and_export(out_csv)

