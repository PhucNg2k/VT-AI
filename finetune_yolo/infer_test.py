import os
import glob
import cv2
from typing import List

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
    # choose most recently modified subdir
    subdirs = [os.path.join(checkpoint_root, d) for d in os.listdir(checkpoint_root)
               if os.path.isdir(os.path.join(checkpoint_root, d))]
    if not subdirs:
        raise FileNotFoundError("No runs found under checkpoint directory")
    latest = max(subdirs, key=os.path.getmtime)
    # prefer best.pt then last.pt
    for w in [os.path.join(latest, 'weights', 'best.pt'), os.path.join(latest, 'weights', 'last.pt')]:
        if os.path.isfile(w):
            return w
    # fallback: any *.pt under latest
    cand = glob.glob(os.path.join(latest, '**', '*.pt'), recursive=True)
    if not cand:
        raise FileNotFoundError("No .pt weights found in latest checkpoint run")
    return cand[0]


def draw_detections(img, result, image_path: str) -> None:
    split = 'test'
    print(f"Split: {split}")
    xyxy = result.boxes.xyxy if (result.boxes is not None and len(result.boxes) > 0) else []
    confs = result.boxes.conf.tolist() if (result.boxes is not None and result.boxes.conf is not None) else [1.0] * len(xyxy)
    clsi = result.boxes.cls.int() if (result.boxes is not None and result.boxes.cls is not None) else []
    names = [result.names[c.item()] for c in clsi] if len(clsi) == len(xyxy) else ["obj"] * len(xyxy)
    # Collect dets for filter-phase selection
    dets_for_filter = []
    for i in range(len(xyxy)):
        # visualize only 'parcel-box' class
        if i < len(names) and names[i] != 'parcel-box':
            continue
        x1f, y1f, x2f, y2f = [float(v) for v in xyxy[i].tolist()]
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        cx, cy = 0.5 * (x1f + x2f), 0.5 * (y1f + y2f)
        depth_mm = get_depth_pixel(image_path, (cx, cy), set=split)
        depth_txt = f"{depth_mm:.1f}mm" if depth_mm is not None and float(depth_mm) > 0 else "NA"
        label = f"{names[i]} {confs[i]:.2f} depth={depth_txt}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Build detection dict compatible with filter_phase
        dets_for_filter.append({
            "image_path": image_path,
            "center": [cx, cy],
            "bbox_xyxy": [x1f, y1f, x2f, y2f],
            "polygon": [[x1f, y1f], [x2f, y1f], [x2f, y2f], [x1f, y2f]],
            "conf": float(confs[i]) if i < len(confs) else 1.0,
            "label": names[i]
        })

    # Overlay filtered selection in RED (always, even if no YOLO boxes)
    if not dets_for_filter:
        # Ensure the filter receives this image to produce fallback ROI center
        dets_for_filter = [{"image_path": image_path, "center": []}]
    
    selections = select_topmost_per_image(dets_for_filter, split=split)
    
    for img_path, center_uv, depth_sel, det_sel in selections:
        if img_path != image_path:
            continue
        # keep float center for math; ints only for drawing
        cxf, cyf = float(center_uv[0]), float(center_uv[1])
        cxs, cys = int(round(cxf)), int(round(cyf))
        
        bbox = det_sel.get("bbox_xyxy") if isinstance(det_sel, dict) else None
        
        if isinstance(bbox, list) and len(bbox) == 4:
            x1s, y1s, x2s, y2s = [int(v) for v in bbox]
        else:
            # No bbox: draw a small box centered at the selected center
            half = 10
            x1s, y1s, x2s, y2s = cxs - half, cys - half, cxs + half, cys + half
        cv2.rectangle(img, (x1s, y1s), (x2s, y2s), (0, 0, 255), 2)
        cv2.circle(img, (cxs, cys), 5, (0, 0, 255), -1)

        # get 3d camera coord from center (float) -> [x,y,z] in meters
        xyz_cam = None
        if depth_sel is not None and float(depth_sel) > 0:
            z_m = float(depth_sel) / 1000.0
            xyz = unproject_pixel_to_cam(cxf, cyf, z_m, COLOR_INTRINSIC)
            if xyz is not None:
                xyz_cam = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

        # label depth and optionally 3D
       
        lbl_xyz = f"X={xyz_cam[0]:.3f}m Y={xyz_cam[1]:.3f}m Z={xyz_cam[2]:.3f}m"
        y_text = min(img.shape[0] - 5, y1s + 18)
        cv2.putText(img, lbl_xyz, (x1s, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)

    img_paths: List[str] = get_file_list(RGB_TEST, -1)
    if not img_paths:
        print('No test images found at:', RGB_TEST)
        return

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            print('Failed to read:', p)
            continue
        results = model(p)
        for res in results:
            draw_detections(img, res, p)
        cv2.imshow('inference', img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


