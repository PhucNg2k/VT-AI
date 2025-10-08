Finetune YOLO – Parcel Box Dataset Pipeline

Overview
- Manual labeling → data augmentation → dataset prep for YOLO finetuning.

Prerequisites
- Python + OpenCV installed
- Images in RGB_TRAIN (see Code/config.py)

1) Label images (manual)
```bash
python -m Code.finetune_yolo.collect_label
```
- Draw boxes with mouse. Keys: n(next) p(prev) s(save) d(del last) c(clear) q(quit)
- Labels saved as YOLO txt to a sibling folder named labels

2) Augment data
```bash
python -m Code.finetune_yolo.augment_data /path/to/RGB_TRAIN /path/to/labels --out_dir exp/yolo_augmented --max_aug_per_image 2
```
- Creates augmented images/labels under exp/yolo_augmented/{images,labels}
- Augmentations: color jitter (global), in-place bbox patch flip/rotate

3) Prepare YOLO dataset (train/val + data.yaml)
```bash
python -m Code.finetune_yolo.prepate_dataloader exp/yolo_augmented/images exp/yolo_augmented/labels --out_root exp/yolo_dataset --val_ratio 0.1
```
- Outputs to exp/yolo_dataset/{images,labels}/{train,val} and exp/yolo_dataset/data.yaml

Notes
- Single class: parcel-box (id=0)
- Ensure image filenames match label filenames (name.txt)
- You can point step (3) directly to original images/labels if you skip augmentation

