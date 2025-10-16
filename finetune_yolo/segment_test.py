from ultralytics import YOLO
from Code.config import RGB_TEST, RGB_TRAIN
import cv2
import numpy as np
import os

from Code.file_utils import get_file_list
from Code.finetune_yolo.model_utils import detect_in_area_batch, detect_on_original_image
from Code.phases.filter_phase import select_topmost_per_image


weights = r'D:\ViettelAI\Code\finetune_yolo\checkpoint\parcel_yolo11l\weights\best.pt'
detect_model = YOLO(weights)

seg_model = YOLO("yolo11l-seg.pt")

def visualize_segmentation(image, det_sel, seg_results, bbox_coords=None):
    """
    Visualize detection and segmentation results on the image.
    
    Args:
        image: Original image
        det_sel: Selected detection result
        seg_results: Segmentation results from YOLO (on cropped region)
        bbox_coords: Bounding box coordinates (x1, y1, x2, y2) for mapping mask back
    """
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Get bounding box from detection
    bbox_xyxy = det_sel.get('bbox_xyxy', [])
    if len(bbox_xyxy) == 4:
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, 'Detection', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw segmentation masks if available
    if seg_results and len(seg_results) > 0 and bbox_coords:
        x1, y1, x2, y2 = bbox_coords
        
        for result in seg_results:
            if result.masks is not None:
                # Get the first mask (assuming single object)
                mask = result.masks.data[0].cpu().numpy()
                
                # Resize mask to match cropped region dimensions
                cropped_h, cropped_w = y2 - y1, x2 - x1
                mask_resized = cv2.resize(mask, (cropped_w, cropped_h))
                
                # Create colored mask overlay for the full image
                colored_mask = np.zeros_like(vis_image)
                
                # Map mask back to original image coordinates
                mask_full = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                mask_full[y1:y2, x1:x2] = mask_resized
                
                # Create colored overlay
                colored_mask[mask_full > 0.5] = [0, 0, 255]  # Red mask
                
                # Blend with original image
                vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
                
                # Draw mask contour
                mask_binary = (mask_full > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_image, contours, -1, (255, 0, 0), 2)
                
                cv2.putText(vis_image, 'Segmentation', (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return vis_image

def run_segmentation_test():
    """Run segmentation test on training images."""
    img_paths = get_file_list(RGB_TRAIN, -1)
    
    for i, image_path in enumerate(img_paths):
        print(f"Processing {i+1}/{len(img_paths)}: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Detection
        dets_for_filter = detect_on_original_image(detect_model, image_path, target_class='parcel-box')
        if not dets_for_filter:
            dets_for_filter = [{"image_path": image_path, "center_uv": []}]

        selections = select_topmost_per_image(dets_for_filter, direct=False, split='train')
        if len(selections) == 1:
            img_path, center_uv, depth_sel, det_sel = selections[0]
            
            # Get bounding box for segmentation
            bbox_xyxy = det_sel.get('bbox_xyxy', [])
            if len(bbox_xyxy) == 4:
                x1, y1, x2, y2 = map(int, bbox_xyxy)
                
                # Run segmentation on the detected region
                try:
                    # Crop image to detected bounding box
                    cropped_image = image[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        print("Empty cropped image")
                        continue
                    
                    # Run segmentation on cropped region
                    seg_results = seg_model(cropped_image, verbose=False)
                    
                    # Visualize results
                    vis_image = visualize_segmentation(image, det_sel, seg_results, (x1, y1, x2, y2))
                    
                    # Display the result
                    cv2.imshow('Detection + Segmentation', vis_image)
                    
                    # Wait for key press
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('s'):
                        # Save the visualization
                        output_path = f"segmentation_result_{i+1}.jpg"
                        cv2.imwrite(output_path, vis_image)
                        print(f"Saved: {output_path}")
                        
                except Exception as e:
                    print(f"Segmentation failed: {e}")
                    # Show just the detection
                    vis_image = visualize_segmentation(image, det_sel, None)
                    cv2.imshow('Detection Only', vis_image)
                    cv2.waitKey(1000)  # Wait 1 second
            else:
                print("No valid bounding box found")
        else:
            print("No valid detection found")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_segmentation_test()