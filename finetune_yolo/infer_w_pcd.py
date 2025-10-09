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
    from Code.camera_config import unproject_pixel_to_cam, COLOR_INTRINSIC
    from Code.finetune_yolo.model_utils import detect_in_area_batch
except ImportError:
    from config import RGB_TEST, RGB_TRAIN
    from file_utils import get_file_list, get_depth_pixel
    from phases.filter_phase import select_topmost_per_image
    from camera_config import unproject_pixel_to_cam, COLOR_INTRINSIC
    from finetune_yolo.model_utils import detect_in_area_batch

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


def predict_orientation_hybrid(pcd, position, eps=0.03, min_points=20, radius=0.05, max_iterations=1000, threshold=0.01):
    """
    Hybrid approach: DBSCAN for clustering + RANSAC for plane fitting
    """
    points = np.asarray(pcd.points)
    
    # Step 1: DBSCAN clustering
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    if max_label < 0:
        return None

    # Step 2: Find cluster containing target position, that cluster is closest to the target position
    distances = np.linalg.norm(points - position, axis=1)
    closest_point_index = np.argmin(distances)
    target_cluster_label = labels[closest_point_index]

    if target_cluster_label == -1:
        return None
        
    # Step 3: Get points from target cluster
    cluster_mask = labels == target_cluster_label
    cluster_points = points[cluster_mask]
    
    if len(cluster_points) < min_points:
        return None

    # Step 4: RANSAC plane fitting on cluster points
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


def predict_orientation_ransac(pcd, position, radius=0.05, max_iterations=1000, threshold=0.01):
    """
    Predict orientation using RANSAC plane fitting around the target position.
    """
    points = np.asarray(pcd.points)
    
    # 1. Find points within radius of target position
    distances = np.linalg.norm(points - position, axis=1)
    nearby_indices = np.where(distances <= radius)[0]
    
    if len(nearby_indices) < 10:
        return None
    
    nearby_points = points[nearby_indices]
    
    # 2. RANSAC plane fitting
    best_normal = None
    best_inliers = 0
    
    for _ in range(max_iterations):
        # Randomly select 3 points
        if len(nearby_points) < 3:
            break
            
        sample_indices = np.random.choice(len(nearby_points), 3, replace=False)
        sample_points = nearby_points[sample_indices]
        
        # Fit plane through 3 points
        try:
            # Calculate two vectors in the plane
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            
            # Normal vector (cross product)
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            
            if normal_norm < 1e-6:  # Degenerate case
                continue
                
            normal = normal / normal_norm
            
            # Count inliers (points close to the plane)
            plane_distances = np.abs(np.dot(nearby_points - sample_points[0], normal))
            inliers = np.sum(plane_distances <= threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                
        except:
            continue
    
    return best_normal


def predict_orientation_from_pcd(pcd, position, eps=0.03, min_points=50):
    """
    Predict orientation vector using DBSCAN clustering on PCD,
    identify cluster containing the 3D point, then estimate mean normal vector.
    """
    # 1. Apply DBSCAN clustering
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    if max_label < 0:
        return None

    # 2. Find cluster containing the target position
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points - position, axis=1)
    closest_point_index = np.argmin(distances)
    target_cluster_label = labels[closest_point_index]

    if target_cluster_label == -1:
        return None
        
    # 3. Filter points belonging to target cluster
    inliers = labels == target_cluster_label
    cluster_points = points[inliers]
    
    if len(cluster_points) < min_points:
        return None

    # 4. Create sub-point cloud from cluster
    local_pcd = o3d.geometry.PointCloud()
    local_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    
    # 5. Estimate normals on selected cluster
    local_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=eps * 2, max_nn=30)
    )
    
    # 6. Calculate mean normal vector
    normals = np.asarray(local_pcd.normals)
    if normals.size == 0:
        return None
        
    mean_normal = np.mean(normals, axis=0)
    predicted_vector = mean_normal / np.linalg.norm(mean_normal)
    
    return predicted_vector


def run_and_export(output_csv: str) -> None:
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)

    img_paths: List[str] = get_file_list(RGB_TEST, -1)
    if not img_paths:
        print('No test images found at:', RGB_TEST)
        return

    rows: List[dict] = []

    # Use ROI-based detection from model_utils for all images
    all_detections = detect_in_area_batch(model, img_paths, target_class='parcel-box')
    
    # Group detections by image path
    detections_by_image = {}
    for det in all_detections:
        img_path = det["image_path"]
        if img_path not in detections_by_image:
            detections_by_image[img_path] = []
        detections_by_image[img_path].append(det)

    for p in img_paths:
        # Get detections for this image
        dets_for_filter = detections_by_image.get(p, [])
        
        if not dets_for_filter:
            # Ensure we get ROI-center fallback from filter
            dets_for_filter = [{"image_path": p, "center": []}]

        selections = select_topmost_per_image(dets_for_filter, split='test')
        
        # There will be exactly one selection for current image path
        for img_path, center_uv, depth_sel, det_sel in selections:
            if img_path != p:
                continue
        
            # Compute 3D camera coordinates from center_uv and depth
            x_cam = y_cam = z_cam = 0.0
            rx = ry = rz = 0.0
            if depth_sel is not None and float(depth_sel) > 0:
                u_f, v_f = float(center_uv[0]), float(center_uv[1]) # float center 
                z_m = float(depth_sel) / 1000.0 # meter scale for z
                xyz = unproject_pixel_to_cam(u_f, v_f, z_m, COLOR_INTRINSIC)
                if xyz is not None:
                    x_cam, y_cam, z_cam = float(xyz[0]), float(xyz[1]), float(xyz[2])

                    # Load corresponding PLY file and estimate orientation
                    img_basename = os.path.splitext(os.path.basename(p))[0]
                    ply_path = os.path.join(os.path.dirname(p).replace('rgb', 'ply'), f"{img_basename}.ply")
                    
                    if os.path.isfile(ply_path):
                        try:
                            pcd = o3d.io.read_point_cloud(ply_path)
                            if len(pcd.points) > 0:
                                normal_vector = predict_orientation_hybrid(pcd, np.array([x_cam, y_cam, z_cam]))
                                if normal_vector is not None:
                                    # Ensure Rz is positive (pointing toward scene)
                                    if normal_vector[2] < 0:
                                        normal_vector = -normal_vector
                                    rx, ry, rz = normal_vector[0], normal_vector[1], normal_vector[2]
                        except Exception as e:
                            print(f"Error processing PLY {ply_path}: {e}")
                            rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                    else:
                        rx, ry, rz = 0.0, 0.0, 1.0  # fallback
            else:
                rx, ry, rz = 0.0, 0.0, 1.0  # fallback

            # Compose CSV row
            rows.append({
                "image_filename": f"image_{os.path.basename(p)}",
                "x": f"{x_cam:.3f}",
                "y": f"{y_cam:.3f}",
                "z": f"{z_cam:.3f}",
                "Rx": f"{rx:.3f}",
                "Ry": f"{ry:.3f}",
                "Rz": f"{rz:.3f}"
            })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename","x","y","z","Rx","Ry","Rz"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {output_csv}")


if __name__ == "__main__":
    out_csv = os.path.join(os.path.dirname(__file__), "submit", "Submit_test_pcd_hybrid.csv")
    print(f"Saving to {out_csv}")
    run_and_export(out_csv)

