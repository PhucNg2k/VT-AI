import os
import shutil
import random
import argparse
from typing import List, Tuple


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_data_yaml(out_yaml: str, root_dir: str):
    content = (
        f"path: {root_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['parcel-box']\n"
    )
    with open(out_yaml, 'w', encoding='utf-8') as f:
        f.write(content)


def collect_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return pairs
    img_files = sorted(os.listdir(images_dir))
    for f in img_files:
        name, ext = os.path.splitext(f)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        img_path = os.path.join(images_dir, f)
        label_path = os.path.join(labels_dir, name + '.txt')
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
    return pairs


def split_pairs(pairs: List[Tuple[str, str]], val_ratio: float = 0.1, seed: int = 42):
    random.Random(seed).shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val


def copy_pairs(pairs: List[Tuple[str, str]], out_img_dir: str, out_lbl_dir: str):
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))
        shutil.copy2(lbl_path, os.path.join(out_lbl_dir, os.path.basename(lbl_path)))


def prepare_dataset(
    images_dir: str,
    labels_dir: str,
    out_root: str = os.path.abspath(os.path.join('exp', 'yolo_dataset')),
    val_ratio: float = 0.1,
    seed: int = 42,
):

    pairs = collect_pairs(images_dir, labels_dir)
    if not pairs:
        print('No (image,label) pairs found. Make sure labels exist in YOLO format.')
        return

    train_pairs, val_pairs = split_pairs(pairs, val_ratio=val_ratio, seed=seed)

    images_train_dir = os.path.join(out_root, 'images', 'train')
    images_val_dir = os.path.join(out_root, 'images', 'val')
    labels_train_dir = os.path.join(out_root, 'labels', 'train')
    labels_val_dir = os.path.join(out_root, 'labels', 'val')

    copy_pairs(train_pairs, images_train_dir, labels_train_dir)
    copy_pairs(val_pairs, images_val_dir, labels_val_dir)

    write_data_yaml(os.path.join(out_root, 'data.yaml'), out_root)

    print('Prepared YOLO dataset at:', out_root)
    print('Train pairs:', len(train_pairs), 'Val pairs:', len(val_pairs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset from images and labels.')
    parser.add_argument('images_dir', type=str, help='Path to images folder')
    parser.add_argument('labels_dir', type=str, help='Path to labels folder (YOLO format)')
    parser.add_argument('--out_root', type=str, default=os.path.abspath(os.path.join('exp', 'yolo_dataset')), help='Output root directory')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    args = parser.parse_args()

    prepare_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        out_root=args.out_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

