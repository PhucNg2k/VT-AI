import os
import argparse
import yaml
from ultralytics import YOLO


def check_data_yaml(data_yaml_path: str):
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Required keys
    for k in ['train', 'val', 'nc', 'names']:
        if k not in data:
            raise ValueError(f"data.yaml missing key: {k}")

    # Validate names length matches nc
    names = data['names']
    nc = int(data['nc'])
    if not isinstance(names, (list, tuple)) or len(names) != nc:
        raise ValueError("data.yaml names length must equal nc")

    # Resolve train/val directories relative to yaml location
    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    train_dir = os.path.normpath(os.path.join(base_dir, data['train']))
    val_dir = os.path.normpath(os.path.join(base_dir, data['val']))

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train images dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val images dir not found: {val_dir}")

    # Optional: sample a label file to ensure format 'class cx cy w h'
    # We will just check one provided by user if exists
    return train_dir, val_dir


def main():
    parser = argparse.ArgumentParser(description='Train YOLO on parcel-box dataset')
    parser.add_argument('--data', type=str, default=os.path.join(os.path.dirname(__file__), 'augmented_data', 'data.yaml'), help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo11l.pt', help='Base model weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', type=str, default='parcel_yolo11l')
    args = parser.parse_args()

    # Check data.yaml and paths
    train_dir, val_dir = check_data_yaml(args.data)

    # Ensure checkpoint directory
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoint'))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Train
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=ckpt_dir,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    print('Training complete. Checkpoints at:', os.path.join(ckpt_dir, args.name))


if __name__ == '__main__':
    main()


