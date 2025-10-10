import os
import csv
import glob
from typing import List

import cv2
import numpy as np
import open3d as o3d

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


def create_detection_point_cloud_mask(image_path: str, bbox_xyxy: List[float], split='train', default_depth: float = 1.0) -> o3d.geometry.PointCloud:
    """
    Create a point cloud mask from depth map for pixels within the bounding box.
    
    Args:
        image_path: Path to RGB image
        bbox_xyxy: Bounding box coordinates [x1, y1, x2, y2]
        default_depth: Default depth value for pixels with missing/invalid depth
    
    Returns:
        Open3D PointCloud object with 3D coordinates of bbox pixels
    """
    detection_points_3d = []
    
    if bbox_xyxy and len(bbox_xyxy) == 4:
        x1, y1, x2, y2 = [int(coord) for coord in bbox_xyxy]
        
        # Get depth image path
        depth_path = image_path.replace('rgb', 'depth')
        if os.path.isfile(depth_path):
            # Load depth image as grayscale
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            
            # Extract 3D coordinates for each pixel in bbox
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
                        depth_value = get_depth_pixel(image_path, (x,y), direct=True, set=split)
                        if depth_value is not None and depth_value > 0:
                            z_d = float(depth_value) / 1000.0
                        else:
                            z_d = default_depth  # Default depth if None or 0
                        
                        # Convert pixel to 3D camera coordinates
                        pixel_coords = [x, y]
                        x_cam_pixel, y_cam_pixel, _ = get_cam_coord(pixel_coords, COLOR_INTRINSIC)
                        detection_points_3d.append([x_cam_pixel, y_cam_pixel, z_d])
    
    # Create point cloud
    detection_point = o3d.geometry.PointCloud()
    if detection_points_3d:
        detection_point.points = o3d.utility.Vector3dVector(detection_points_3d)
    else:
        # Fallback: create empty point cloud
        detection_point.points = o3d.utility.Vector3dVector([])
    
    return detection_point


def predict_orientation_from_mask(mask_pcd, eps=0.03, min_points=20, max_iterations=1000, threshold=0.01):
    """
    Predict orientation from cloud mask using DBSCAN clustering and RANSAC on densest cluster.
    """
    if len(mask_pcd.points) == 0:
        return None
    
    points = np.asarray(mask_pcd.points)
    
    # Step 1: DBSCAN clustering
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(mask_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    if max_label < 0:
        return None

    # Step 2: Find the densest cluster (cluster with most points)
    cluster_sizes = {}
    for label in range(max_label + 1):
        cluster_sizes[label] = np.sum(labels == label)
    
    # Get cluster with maximum size
    densest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    cluster_points = points[labels == densest_cluster_label]
    
    if len(cluster_points) < min_points:
        return None

    # Step 3: RANSAC plane fitting on densest cluster
    best_normal = None
    best_inliers = 0
    
    for _ in range(max_iterations):
        if len(cluster_points) < 3:
            break
            
        # Randomly select 3 points from cluster
        sample_indices = np.random.choice(len(cluster_points), 3, replace=False)
        sample_points = cluster_points[sample_indices]
        
        try:
            # Calculate plane normal
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            
            if normal_norm < 1e-6:
                continue
                
            normal = normal / normal_norm
            
            # Count inliers within cluster
            plane_distances = np.abs(np.dot(cluster_points - sample_points[0], normal))
            inliers = np.sum(plane_distances <= threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                
        except:
            continue
    
    return best_normal



def run_and_export(output_csv: str, split) -> None:
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)

    T_MAT = np.asarray(PLY_MATRIX, dtype=float)
    
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
            dets_for_filter = [{"image_path": image_path, "center_uv": []}]

        selections = select_topmost_per_image(dets_for_filter, direct=False, split=split)


        
        if len(selections) == 1:
            img_path, center_uv, depth_sel, det_sel = selections[0]
                
            if img_path != image_path:
                print("ERROR: Image path not matching")
                continue
            
            
            
            # get 3d camera coord from center (float) -> [x,y,z] in meters
            xyz_cam = None
            if depth_sel is not None and float(depth_sel) > 0:
                z_d = float(depth_sel) / 1000.0
                #xyz = unproject_pixel_to_cam(cxf, cyf, z_m, COLOR_INTRINSIC)

                x_cam, y_cam, _ = get_cam_coord(center_uv, COLOR_INTRINSIC)
                        
                xyz_cam = tuple(map(float, (x_cam, y_cam, z_d)))
                x_cam, y_cam, z_cam = xyz_cam

                # Create cloud mask from bbox and estimate orientation
                bbox_xyxy = det_sel.get('bbox_xyxy', [])
                if not bbox_xyxy or len(bbox_xyxy) != 4:
                    print("No valid bbox found, using fallback orientation")
                    rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                else:
                    mask_pcd = create_detection_point_cloud_mask(image_path, bbox_xyxy, split=split)
                    
                    if len(mask_pcd.points) > 0:
                        try:
                            normal_vector = predict_orientation_from_mask(mask_pcd)
                            if normal_vector is not None:
                                normal_vector = np.asarray(normal_vector)
                                if normal_vector[2] < 0:
                                    normal_vector = normal_vector @ T_MAT
                                
                                print("Estimated normal vector: ", normal_vector)
                                rx, ry, rz = normal_vector
                            else:
                                rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                        except Exception as e:
                            print(f"Error processing mask: {e}")
                            rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                    else:
                        rx, ry, rz = 0.0, 0.0, 1.0  # fallback
            else:
                rx, ry, rz = 0.0, 0.0, 1.0  # fallback
    
            # Compose CSV row
            rows.append({
                "image_filename": f"image_{os.path.basename(image_path)}",
                "x": f"{x_cam:.3f}",
                "y": f"{y_cam:.3f}",
                "z": f"{z_cam:.3f}",
                "Rx": f"{rx:.3f}",
                "Ry": f"{ry:.3f}",
                "Rz": f"{rz:.3f}"
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
    out_csv = os.path.join(os.path.dirname(__file__), "submit", "Submit_test_pcd_hybrid.csv")
    print(f"Saving to {out_csv}")
    run_and_export(out_csv, split='test')

