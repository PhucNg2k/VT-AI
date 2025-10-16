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
    from Code.camera_config import get_cam_coord, COLOR_INTRINSIC, PLY_MATRIX, get_colorpix_depth_value
    from Code.finetune_yolo.model_utils import detect_in_area_batch, detect_on_original_image
except ImportError:
    from config import RGB_TEST, RGB_TRAIN
    from file_utils import get_file_list, get_depth_pixel
    from phases.filter_phase import select_topmost_per_image
    from camera_config import get_cam_coord, COLOR_INTRINSIC, PLY_MATRIX, get_colorpix_depth_value
    from finetune_yolo.model_utils import detect_in_area_batch, detect_on_original_image

from ultralytics import YOLO


T_MAT = np.asarray(PLY_MATRIX, dtype=float)
T_INV = np.linalg.inv(T_MAT)


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


def rgb_to_ply_path(rgb_path: str) -> str:
    base = os.path.splitext(os.path.basename(rgb_path))[0]
    ply_dir = os.path.dirname(rgb_path).replace('rgb', 'ply')
    return os.path.join(ply_dir, f"{base}.ply")


def create_detection_point_cloud_mask(image_path: str, bbox_xyxy: List[float], default_depth: float = 1.0, split='train') -> o3d.geometry.PointCloud:
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
                        depth_value = depth_img[y, x]
                        #depth_value = get_colorpix_depth_value( image_path, (x,y), )                        
                        #print('DEPTH VALUE: ', int(depth_value))
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
        detection_point.paint_uniform_color([0, 0, 1])  # Blue color
    else:
        # Fallback: create empty point cloud
        detection_point.points = o3d.utility.Vector3dVector([])
    
    return detection_point


def preprocess_pcd(detection_point: o3d.geometry.PointCloud, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using a voxel grid of the given size (in meters).
    """
    return detection_point.voxel_down_sample(voxel_size)


def filter_cloud_mask_dbscan(detection_point: o3d.geometry.PointCloud, eps: float = 0.03, min_points: int = 20, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Filter a mask point cloud (camera frame) to keep only the densest DBSCAN cluster.
    Operates entirely in camera frame.
    """
    if len(detection_point.points) == 0:
        return detection_point

    # Downsample before clustering for speed/robustness
    ds_cloud = preprocess_pcd(detection_point, voxel_size=voxel_size) if voxel_size > 0 else detection_point
    if len(ds_cloud.points) == 0:
        return detection_point

    ds_points = np.asarray(ds_cloud.points)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels = np.array(ds_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    if labels.size == 0 or labels.max() < 0:
        return detection_point

    # densest cluster in downsampled
    max_label = labels.max()
    cluster_sizes = {label: np.sum(labels == label) for label in range(max_label + 1)}
    densest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    filtered_points = ds_points[labels == densest_cluster_label]

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered


def predict_orientation_from_mask(mask_pcd: o3d.geometry.PointCloud, min_points: int = 20, max_iterations: int = 1000, threshold: float = 0.01):
    """
    Estimate plane normal using simple RANSAC on the provided camera-frame mask.
    Returns a unit normal vector in camera frame.
    """
    if len(mask_pcd.points) == 0:
        return None

    points = np.asarray(mask_pcd.points)
    if len(points) < min_points:
        return None

    best_normal = None
    best_inliers = 0
    for _ in range(max_iterations):
        if len(points) < 3:
            break
        idx = np.random.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        distances = np.abs(np.dot(points - p0, n))
        inliers = int(np.sum(distances <= threshold))
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = n

    if best_normal is None:
        return None

    # Ensure the normal points to positive Z (camera looks along +Z)
    if best_normal[2] < 0:
        best_normal = -best_normal
    return best_normal

def visualize_detection_with_pcd(image_path: str, model: YOLO, split: str = 'test'):
    """Visualize detected bbox coordinates (blue) with corresponding point cloud"""
    
    # Run inference pipeline like infer_final_submit.py
    dets_for_filter = detect_on_original_image(model, image_path, target_class='parcel-box')
    if not dets_for_filter:
        dets_for_filter = [{"image_path": image_path, "center": []}]

    selections = select_topmost_per_image(dets_for_filter, split=split)
    
    if len(selections) != 1:
        print("INVALID SELECTED RESULT")
        return
    
    img_path, center_uv, depth_sel, det_sel = selections[0]
    
    if img_path != image_path:
        print("ERROR: Image path not matching")
        return
    
    # Get 3D camera coordinates like infer_final_submit.py
    xyz_cam = None
    if depth_sel is not None and float(depth_sel) > 0:
        z_d = float(depth_sel) / 1000.0
        x_cam, y_cam, _ = get_cam_coord(center_uv, COLOR_INTRINSIC)

        xyz_cam = tuple(map(float, (x_cam, y_cam, z_d)))
        x_cam, y_cam, z_cam = xyz_cam
        
        print(f"Detected coordinates: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f}")

        # camera point
        target_point = o3d.geometry.PointCloud()
        target_point.points = o3d.utility.Vector3dVector([[x_cam, y_cam, z_cam] ])
        target_point.paint_uniform_color([1, 0, 0])  # Blue color

        bbox_xyxy = det_sel['bbox_xyxy']
        
        # Create detection point cloud mask from depth map
        detection_point = create_detection_point_cloud_mask(image_path, bbox_xyxy, split=split)
        print(f"Extracted {len(detection_point.points)} 3D points from bbox")

        
        # Load corresponding PLY file
        ply_path = rgb_to_ply_path(image_path)
        if os.path.isfile(ply_path):
            try:
                # Load original point cloud
                pcd_original = o3d.io.read_point_cloud(ply_path)
                # Convert to camera frame once for all downstream ops
                pcd_original.points = pcd_original.points @ T_MAT

                if len(pcd_original.points) > 0:
                    # Use the detection point cloud mask created above
                    if len(detection_point.points) == 0:
                        # Fallback to center point if no bbox points
                        detection_point.points = o3d.utility.Vector3dVector([[x_cam, y_cam, z_cam]])
                        detection_point.paint_uniform_color([0, 0, 1])  # Blue color
                    
                    # Filter to densest cluster (DBSCAN) in camera frame
                    mask_filtered = filter_cloud_mask_dbscan(detection_point, eps=0.02, min_points=20, voxel_size=0.01)

                    # Estimate normal via RANSAC in camera frame
                    normal_cam = predict_orientation_from_mask(mask_filtered)
                    if normal_cam is not None:
                        print("Estimated normal (camera frame): ", normal_cam)

                    # Prepare a normal arrow for visualization (optional)
                    normal_arrow = None
                    if normal_cam is not None and len(mask_filtered.points) > 0:
                        arrow_len = 0.1
                        normal_arrow = o3d.geometry.TriangleMesh.create_arrow(
                            cylinder_radius=0.003,
                            cone_radius=0.007,
                            cylinder_height=arrow_len * 0.7,
                            cone_height=arrow_len * 0.3
                        )
                        # Orientation to align +Z to normal_cam
                        z_axis = np.array([0.0, 0.0, 1.0])
                        if np.allclose(normal_cam, z_axis):
                            R = np.eye(3)
                        else:
                            axis = np.cross(z_axis, normal_cam)
                            axis_norm = np.linalg.norm(axis)
                            if axis_norm < 1e-6:
                                R = np.eye(3)
                            else:
                                axis = axis / axis_norm
                                angle = np.arccos(np.clip(np.dot(z_axis, normal_cam), -1.0, 1.0))
                                K = np.array([[0, -axis[2], axis[1]],
                                              [axis[2], 0, -axis[0]],
                                              [-axis[1], axis[0], 0]])
                                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                        normal_arrow.rotate(R, center=[0, 0, 0])
                        # Place arrow at centroid of mask_filtered
                        centroid = np.asarray(mask_filtered.points).mean(axis=0)
                        normal_arrow.translate(centroid)
                        normal_arrow.paint_uniform_color([1, 0, 0])

                    # Create coordinate frame
                    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    
                    # Visualize together
                    print(f"Visualizing detection vs point cloud: {ply_path}")
                    geoms = [pcd_original, detection_point, target_point, axis]
                    if 'normal_arrow' in locals() and normal_arrow is not None:
                        geoms.append(normal_arrow)
                    if len(mask_filtered.points) > 0:
                        mask_filtered.paint_uniform_color([1, 0, 0])
                        geoms.append(mask_filtered)
                    o3d.visualization.draw_geometries(
                        geoms, 
                        window_name=f"Detection vs PCD: {os.path.basename(image_path)}",
                        lookat=[x_cam, y_cam, z_cam], 
                        zoom=0.5
                    )
                else:
                    print("Empty point cloud:", ply_path)
            except Exception as e:
                print(f"Error loading PLY {ply_path}: {e}")
        else:
            print("PLY file not found:", ply_path)
    else:
        print("No valid depth data")


def main():
    # Find latest checkpoint
    checkpoint_root = os.path.join(os.path.dirname(__file__), '..', 'finetune_yolo', 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)
    
    print(f"Using weights: {weights}")
    
    split='test'
    print(f"Split: {split}")
    data_src = RGB_TRAIN if split=='train' else RGB_TEST

    img_paths: List[str] = get_file_list(data_src, -1)
    
    if not img_paths:
        print('No images found at:', data_src)
        return
    
    print("Data source: ", data_src)
    print(f"Found {len(img_paths)} test images")
    
    # Visualize first few images
    for i, image_path in enumerate(img_paths[:3]):  # Limit to first 3 images
        print(f"\nProcessing image {i+1}: {os.path.basename(image_path)}")
        visualize_detection_with_pcd(image_path, model, split=split)
        break
        # Ask user if they want to continue
        response = input("Press Enter to continue to next image, or 'q' to quit: ")
        if response.lower() == 'q':
            break


if __name__ == "__main__":
    main()