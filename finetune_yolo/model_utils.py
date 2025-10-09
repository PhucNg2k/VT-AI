

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO


# ROI region defined by 4 corner points in clockwise order:
# Top-left -> Top-right -> Bottom-right -> Bottom-left
# [(274, 84), (919, 110), (932, 549), (272, 565)]
AREA_POINTS = np.array([[274, 84], [919, 110], [932, 549], [272, 565]], dtype=np.int32)


def crop_image_to_roi(image_path: str, roi_points: np.ndarray = None) -> Tuple[np.ndarray, int, int]:
    """
    Crop image to ROI polygon region and return cropped image and offset.
    
    Args:
        image_path: Path to the input image
        roi_points: ROI polygon points in clockwise order (default: AREA_POINTS)
                    Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    Order: Top-left -> Top-right -> Bottom-right -> Bottom-left
    
    Returns:
        Tuple of (cropped_image, offset_x, offset_y)
    """
    if roi_points is None:
        roi_points = AREA_POINTS
        
    img = cv2.imread(image_path)
    if img is None:
        return None, 0, 0
    
    # Create mask for ROI polygon
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    
    # Find bounding rectangle of ROI polygon
    x, y, w, h = cv2.boundingRect(roi_points)
    
    # Crop image to bounding rectangle
    cropped_img = img[y:y+h, x:x+w]
    
    # Adjust ROI polygon points to cropped coordinate system
    cropped_roi_points = roi_points - [x, y]
    
    # Create mask for cropped image using adjusted polygon
    cropped_mask = np.zeros(cropped_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(cropped_mask, [cropped_roi_points], 255)
    
    # Apply polygon mask to cropped image (zero out pixels outside polygon)
    cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
    
    return cropped_img, x, y


def detect_in_area(model: YOLO, image_path: str, roi_points: np.ndarray = None, 
                  target_class: str = 'parcel-box') -> List[Dict[str, Any]]:
    """
    Run YOLO detection within specified ROI polygon area.
    
    Args:
        model: YOLO model instance
        image_path: Path to the input image
        roi_points: ROI polygon points in clockwise order (default: AREA_POINTS)
                    Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    Order: Top-left -> Top-right -> Bottom-right -> Bottom-left
        target_class: Target class to filter detections (default: 'parcel-box')
    
    Returns:
        List of detection dictionaries with coordinates in original image space
    """
    if roi_points is None:
        roi_points = AREA_POINTS
        
    detections = []
    
    # Crop image to ROI region
    cropped_img, offset_x, offset_y = crop_image_to_roi(image_path, roi_points)
    if cropped_img is None:
        return detections
        
    # Run YOLO on cropped image
    results = model(cropped_img)
    
    for result in results:
        boxes = result.boxes
        
        # If there are no boxes at all, register an empty detection for this image
        if boxes is None or len(boxes) == 0:
            detections.append({
                "image_path": image_path,
                "center": [],
                "bbox_xyxy": [],
                "polygon": [],
                "conf": 0,
                "label": None
            })
            continue
        
        xyxy = boxes.xyxy  # [N, 4]
        confs = boxes.conf.tolist() if boxes.conf is not None else [1.0] * len(xyxy)
        clsi = boxes.cls.int() if boxes.cls is not None else []
        names = [result.names[c.item()] for c in clsi] if len(clsi) == len(xyxy) else ["object"] * len(xyxy)

        # Keep only detections with target class
        num_target_boxes = 0
        for i in range(len(xyxy)):
            if i < len(names) and names[i] != target_class:
                continue
                
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            
            # Convert coordinates back to original image space
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
            
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            detections.append({
                "image_path": image_path,
                "center": [float(cx), float(cy)],
                "bbox_xyxy": [x1, y1, x2, y2],
                "polygon": [[float(px), float(py)] for px, py in poly],
                "conf": float(confs[i]),
                "label": names[i]
            })
            num_target_boxes += 1

        # If there were boxes but none with target class, register an empty detection
        if num_target_boxes == 0:
            detections.append({
                "image_path": image_path,
                "center": [],
                "bbox_xyxy": [],
                "polygon": [],
                "conf": 0,
                "label": None
            })

    return detections


def detect_in_area_batch(model: YOLO, image_paths: List[str], roi_points: np.ndarray = None,
                        target_class: str = 'parcel-box') -> List[Dict[str, Any]]:
    """
    Run YOLO detection on multiple images within specified ROI polygon area.
    
    Args:
        model: YOLO model instance
        image_paths: List of image paths
        roi_points: ROI polygon points in clockwise order (default: AREA_POINTS)
                    Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    Order: Top-left -> Top-right -> Bottom-right -> Bottom-left
        target_class: Target class to filter detections (default: 'parcel-box')
    
    Returns:
        List of all detection dictionaries
    """
    all_detections = []
    
    for image_path in image_paths:
        detections = detect_in_area(model, image_path, roi_points, target_class)
        all_detections.extend(detections)
    
    return all_detections


def detect_on_original_image(model: YOLO, image_path: str, target_class: str = 'parcel-box') -> List[Dict[str, Any]]:
    """
    Run YOLO detection on the original full image without ROI cropping.
    
    Args:
        model: YOLO model instance
        image_path: Path to the input image
        target_class: Target class to filter detections (default: 'parcel-box')
    
    Returns:
        List of detection dictionaries with coordinates in original image space
    """
    detections = []
    
    # Run YOLO on original image (no cropping)
    results = model(image_path)
    
    for result in results:
        boxes = result.boxes
        
        # If there are no boxes at all, register an empty detection for this image
        if boxes is None or len(boxes) == 0:
            detections.append({
                "image_path": image_path,
                "center_uv": [],
                "bbox_xyxy": [],
                "polygon": [],
                "conf": 0,
                "label": None
            })
            continue
        
        xyxy = boxes.xyxy  # [N, 4]
        confs = boxes.conf.tolist() if boxes.conf is not None else [1.0] * len(xyxy)
        clsi = boxes.cls.int() if boxes.cls is not None else []
        names = [result.names[c.item()] for c in clsi] if len(clsi) == len(xyxy) else ["object"] * len(xyxy)

        # Keep only detections with target class
        num_target_boxes = 0
        for i in range(len(xyxy)):
            if i < len(names) and names[i] != target_class:
                continue
                
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            
            # Coordinates are already in original image space (no offset needed)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            detections.append({
                "image_path": image_path,
                "center_uv": [float(cx), float(cy)],
                "bbox_xyxy": [x1, y1, x2, y2],
                "polygon": [[float(px), float(py)] for px, py in poly],
                "conf": float(confs[i]),
                "label": names[i]
            })
            num_target_boxes += 1

        # If there were boxes but none with target class, register an empty detection
        if num_target_boxes == 0:
            detections.append({
                "image_path": image_path,
                "center_uv": [],
                "bbox_xyxy": [],
                "polygon": [],
                "conf": 0,
                "label": None
            })

    return detections


def detect_on_original_image_batch(model: YOLO, image_paths: List[str], target_class: str = 'parcel-box') -> List[Dict[str, Any]]:
    """
    Run YOLO detection on multiple original full images without ROI cropping.
    
    Args:
        model: YOLO model instance
        image_paths: List of image paths
        target_class: Target class to filter detections (default: 'parcel-box')
    
    Returns:
        List of all detection dictionaries
    """
    all_detections = []
    
    for image_path in image_paths:
        detections = detect_on_original_image(model, image_path, target_class)
        all_detections.extend(detections)
    
    return all_detections