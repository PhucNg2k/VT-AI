import os
import glob
import cv2
from typing import List

try:
    from Code.config import RGB_TEST
    from Code.file_utils import get_file_list
except ImportError:
    from config import RGB_TEST
    from file_utils import get_file_list

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


def draw_detections(img, result) -> None:
    if result.boxes is None or len(result.boxes) == 0:
        return
    xyxy = result.boxes.xyxy
    confs = result.boxes.conf.tolist() if result.boxes.conf is not None else [1.0] * len(xyxy)
    clsi = result.boxes.cls.int() if result.boxes.cls is not None else []
    names = [result.names[c.item()] for c in clsi] if len(clsi) == len(xyxy) else ["obj"] * len(xyxy)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
        label = f"{names[i]} {confs[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)

    img_paths: List[str] = get_file_list(RGB_TEST, 5)
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
            draw_detections(img, res)
        cv2.imshow('inference', img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


