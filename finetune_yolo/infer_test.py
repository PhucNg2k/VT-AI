import os
import glob
import cv2
from typing import List

try:
    from Code.config import RGB_TEST
    from Code.file_utils import get_file_list, get_depth_pixel
    from Code.phases.filter_phase import select_topmost_per_image
except ImportError:
    from config import RGB_TEST
    from file_utils import get_file_list, get_depth_pixel
    from phases.filter_phase import select_topmost_per_image

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
    if result.boxes is None or len(result.boxes) == 0:
        return
    xyxy = result.boxes.xyxy
    confs = result.boxes.conf.tolist() if result.boxes.conf is not None else [1.0] * len(xyxy)
    clsi = result.boxes.cls.int() if result.boxes.cls is not None else []
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
        depth_mm = get_depth_pixel(image_path, (cx, cy), set='test')
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
    selections = select_topmost_per_image(dets_for_filter, split='test')
    for img_path, center_uv, depth_sel, det_sel in selections:
        if img_path != image_path:
            continue
        cxs, cys = int(center_uv[0]), int(center_uv[1])
        bbox = det_sel.get("bbox_xyxy") if isinstance(det_sel, dict) else None
        if isinstance(bbox, list) and len(bbox) == 4:
            x1s, y1s, x2s, y2s = [int(v) for v in bbox]
        else:
            # No bbox: draw a small box centered at the selected center
            half = 10
            x1s, y1s, x2s, y2s = cxs - half, cys - half, cxs + half, cys + half
        cv2.rectangle(img, (x1s, y1s), (x2s, y2s), (0, 0, 255), 2)
        cv2.circle(img, (cxs, cys), 5, (0, 0, 255), -1)
        lbl = f"depth={depth_sel:.1f}mm"
        cv2.putText(img, lbl, (x1s, max(0, y1s - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)

    img_paths: List[str] = get_file_list(RGB_TEST, 10)
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


