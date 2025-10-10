import os
import glob
import cv2
from typing import List

try:
    from Code.config import RGB_TEST, RGB_TRAIN
    from Code.file_utils import get_file_list, get_depth_pixel
    from Code.phases.filter_phase import select_topmost_per_image
    from Code.camera_config import unproject_pixel_to_cam, COLOR_INTRINSIC, get_cam_coord
    from Code.finetune_yolo.model_utils import detect_in_area, detect_on_original_image
except ImportError:
    from config import RGB_TEST, RGB_TRAIN
    from file_utils import get_file_list, get_depth_pixel
    from phases.filter_phase import select_topmost_per_image
    from camera_config import unproject_pixel_to_cam, COLOR_INTRINSIC, get_cam_coord
    from finetune_yolo.model_utils import detect_in_area, detect_on_original_image

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


def draw_detections(model, image_path: str, split) -> None:
    """
    Draw detections using ROI-based detection from model_utils.
    """
    
    # Use ROI-based detection from model_utils
    img = cv2.imread(image_path)
    if img is None:
        print('Failed to read:', image_path)
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    dets_for_filter = detect_on_original_image(model, image_path, target_class='parcel-box')
    
    # Draw all detections in GREEN
    for det in dets_for_filter:
        if not det.get("center_uv") or not det.get("bbox_xyxy"):
            continue
            
        x1f, y1f, x2f, y2f = det["bbox_xyxy"]
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        cx, cy = det["center_uv"]
        depth_mm = get_depth_pixel(image_path, (cx, cy), set=split)
        
        
        cam_x, cam_y, _ = get_cam_coord(det['center_uv'])
        
        
        depth_txt = f"{depth_mm:.1f}mm" if depth_mm is not None and float(depth_mm) > 0 else "NA"
        #label = f"{det['label']} {det['conf']:.2f} depth={depth_txt}"
        label = f"({cam_x:.3f}, {cam_y:.3f}) depth={depth_txt}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(cy)), 5, (255, 0, 0), -1)


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
        
        #cv2.rectangle(img, (x1s, y1s), (x2s, y2s), (0, 0, 255), 2)
        cv2.circle(img, (cxs, cys), 5, (0, 0, 255), -1)

        # get 3d camera coord from center (float) -> [x,y,z] in meters
        xyz_cam = None
        if depth_sel is not None and float(depth_sel) > 0:
            z_d = float(depth_sel) / 1000.0

            x_cam, y_cam, _ = get_cam_coord(center_uv, COLOR_INTRINSIC)
                    
            xyz_cam = tuple(map(float, (x_cam, y_cam, z_d)))

        # label depth and optionally 3D
        if xyz_cam is not None:
            lbl_xyz = f"X={xyz_cam[0]:.3f}m Y={xyz_cam[1]:.3f}m Z={xyz_cam[2]:.3f}m"
            y_text = min(img.shape[0] - 5, y1s + 18)
            cv2.putText(img, lbl_xyz, (x1s, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('inference', img)
    


def main():
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)

    split='test'

    print(f"Split: {split}")
    data_src = RGB_TRAIN if split=='train' else RGB_TEST

    img_paths: List[str] = get_file_list(data_src, -1)
    if not img_paths:
        print('No images found at:', data_src)
        return
    print("Data source: ", data_src)

    for image_path in img_paths:
        draw_detections(model, image_path, split)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            continue

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


