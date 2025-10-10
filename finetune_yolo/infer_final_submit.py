import os
import csv
import glob
from typing import List
import numpy as np
import cv2

try:
    from Code.config import RGB_TEST, RGB_TRAIN
    from Code.file_utils import get_file_list, get_depth_pixel
    from Code.phases.filter_phase import select_topmost_per_image
    from Code.camera_config import get_cam_coord, COLOR_INTRINSIC, PLY_MATRIX
    from Code.finetune_yolo.model_utils import detect_in_area_batch, detect_on_original_image
except ImportError:
    from config import RGB_TEST, RGB_TRAIN
    from file_utils import get_file_list, get_depth_pixel
    from phases.filter_phase import select_topmost_per_image
    from camera_config import get_cam_coord, COLOR_INTRINSIC, PLY_MATRIX
    from finetune_yolo.model_utils import detect_in_area_batch, detect_on_original_image

from ultralytics import YOLO

T_MAT = np.asarray(PLY_MATRIX, dtype=float)

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


def run_and_export(output_csv: str, split) -> None:
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)
    
    print(f"Split: {split}")
    data_src = RGB_TRAIN if split=='train' else RGB_TEST

    img_paths: List[str] = get_file_list(data_src, -1)
    if not img_paths:
        print('No images found at:', data_src)
        return
    print("Data source: ", data_src)

    rows: List[dict] = []
        
    for image_path in img_paths:
        dets_for_filter = detect_on_original_image(model, image_path, target_class='parcel-box')
        if not dets_for_filter:
            dets_for_filter = [{"image_path": image_path, "center": []}]

        selections = select_topmost_per_image(dets_for_filter, split=split, direct=True)
        
        if len(selections) == 1:
            img_path, center_uv, depth_sel, det_sel = selections[0]
                
            if img_path != image_path:
                print("ERROR: Image path not matching")
                continue
            
            # keep float center for math; ints only for drawing
            cxf, cyf = float(center_uv[0]), float(center_uv[1])
            
            # get 3d camera coord from center (float) -> [x,y,z] in meters
            xyz_cam = None
            if depth_sel is not None and float(depth_sel) > 0:
                z_d = float(depth_sel) / 1000.0
                #xyz = unproject_pixel_to_cam(cxf, cyf, z_m, COLOR_INTRINSIC)

                x_cam, y_cam, _ = get_cam_coord(center_uv, COLOR_INTRINSIC)
                point_cloud = [x_cam, y_cam, z_d] @ T_MAT.T        
                
                x_cam, y_cam, z_cam = point_cloud
    
            # Compose CSV row
                rows.append({
                    "image_filename": f"image_{os.path.basename(image_path)}",
                    "x": f"{x_cam:.3f}",
                    "y": f"{y_cam:.3f}",
                    "z": f"{z_cam:.3f}",
                    "Rx": 0,
                    "Ry": 0,
                    "Rz": 1
                })
        else:
            print("INVALID SELECTED RESULT")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename","x","y","z","Rx","Ry","Rz"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {output_csv}")


if __name__ == "__main__":
    out_csv = os.path.join(os.path.dirname(__file__), "submit", "Submission_3D1.csv")
    print(f"Saving to {out_csv}")
    run_and_export(out_csv, split='test')

