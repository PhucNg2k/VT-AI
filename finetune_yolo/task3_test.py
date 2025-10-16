import os
import csv
import glob
from typing import List

import torch
import cv2
import numpy as np
import sys
import open3d as o3d
sys.path.append(r"D:\ViettelAI\Code")


from config import RGB_TEST, RGB_TRAIN
from file_utils import get_file_list, get_depth_pixel, get_depth_pixel_batch as _get_depth_batch
from phases.filter_phase import select_topmost_per_image
from camera_config import get_cam_coord, COLOR_INTRINSIC, PLY_MATRIX, compute_homogen_transform, get_tmat_color2depth , DEPTH_INTRINSIC, get_pix_coord, R_MATRIX
from finetune_yolo.model_utils import detect_in_area_batch, detect_on_original_image
from finetune_yolo.PointNet.pointnetPytorch.pointnet.model import OrientRegressPointNet

from ultralytics import YOLO

T_MAT = np.asarray(PLY_MATRIX, dtype=float)
T_INV = np.linalg.inv(T_MAT)

R_MATRIX = np.asarray(R_MATRIX, dtype=float)


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

def convert_path(f_path: str, src, target) -> str:
    base = os.path.splitext(os.path.basename(f_path))[0]
    depth_dir = os.path.dirname(f_path).replace(src, target)
    ext = 'ply' if target == 'ply' else 'png'
    return os.path.join(depth_dir, f"{base}.{ext}")


def transform_to_camera_frame(pts_cloud):
    pcd_points = np.array(pts_cloud.points)
    new_points = pcd_points @ R_MATRIX.T
    pts_cloud.points = o3d.utility.Vector3dVector(new_points)
    return pts_cloud


def get_depth_correspond(image_path, coord):
    depth_path = convert_path(image_path, src='rgb', target='depth')
    depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    px,py = coord
    xyz_cam = get_cam_coord([px, py], COLOR_INTRINSIC)
    tmat_color2depth = get_tmat_color2depth()
    depth_cam_coord = compute_homogen_transform(xyz_cam, tmat_color2depth)
    depth_pix_coord = get_pix_coord(depth_cam_coord, DEPTH_INTRINSIC)

    u_d, v_d = int(depth_pix_coord[0]), int(depth_pix_coord[1])
    depth_value = depth_np[v_d, u_d][0] / 1000


    return float(depth_value)

def create_detection_point_cloud_mask(image_path: str, det_sel, filter_dense: bool = True, filter_patch=True) -> o3d.geometry.PointCloud:
    """
    Create a point cloud mask from depth map for pixels within the bounding box.
    
    Args:
        image_path: Path to RGB image
        bbox_xyxy: Bounding box coordinates [x1, y1, x2, y2]
        default_depth: Default depth value for pixels with missing/invalid depth
    
    Returns:
        Open3D PointCloud object with 3D coordinates of bbox pixels
    """
    depth_path = convert_path(image_path, src='rgb', target='depth')
    depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)


    detection_points_3d = []

    bbox_xyxy = det_sel.get('bbox_xyxy', [])

    color_3dpoints = []
    tmat_color2depth = get_tmat_color2depth()
    
    if isinstance(bbox_xyxy, np.ndarray):
        bbox_xyxy = bbox_xyxy.tolist()

    if bbox_xyxy and len(bbox_xyxy) == 4:
        x1, y1, x2, y2 = list(map(int, bbox_xyxy))
        for py in range(y1, y2):
            for px in range(x1, x2):
            # Get 3D camera coordinates from color pixel (with z=1 for ray direction)
                xyz_cam = get_cam_coord([px, py], COLOR_INTRINSIC)
                
                # Transform color camera coordinates to depth camera coordinates
                depth_cam_coord = compute_homogen_transform(xyz_cam, tmat_color2depth)
                depth_pix_coord = get_pix_coord(depth_cam_coord, DEPTH_INTRINSIC)

                x_d = float(depth_cam_coord[0])
                y_d = float(depth_cam_coord[1])

                if depth_pix_coord is None:
                    continue

                u_d, v_d = int(depth_pix_coord[0]), int(depth_pix_coord[1])

                if 0 <= v_d < depth_np.shape[0] and 0 <= u_d < depth_np.shape[1]:
                    # Get depth value and convert to meters
                    depth_value = float(depth_np[v_d, u_d][0]) 

                    if depth_value <= 0:
                        continue
            
                    if depth_value >= 1150 and depth_value <= 1200:  # remove background
                        continue

                    z_d = depth_value / 1000
                    color_3dpoints.append([x_d, y_d, z_d] @ R_MATRIX.T)
        
        
    detection_points_3d = np.array(color_3dpoints)
    
    # Create point cloud
    detection_point = o3d.geometry.PointCloud()
    if detection_points_3d is not None:
        detection_point.points = o3d.utility.Vector3dVector(detection_points_3d)
    else:
        # Fallback: create empty point cloud
        detection_point.points = o3d.utility.Vector3dVector([])

     # Keep only the top-grasp surface points (lowest depth values)
    # This helps focus on the surface that's most accessible for grasping
    if filter_patch:
        points = np.asarray(detection_point.points)
        
        # Method 1: Use depth percentile instead of raw lowest values
        # This is more robust to outliers and noise
        z_coords = points[:, 2]
        depth_percentile = np.percentile(z_coords, 80)  # Top 20% closest to camera
        top_surface_mask = z_coords <= depth_percentile
        top_surface_points = points[top_surface_mask]
        
        # Create new point cloud with filtered points
        detection_point = o3d.geometry.PointCloud()
        detection_point.points = o3d.utility.Vector3dVector(top_surface_points)
        
    # Optionally keep only the densest cluster from the mask
    if filter_dense and len(detection_point.points) > 0:
        detection_point = filter_cloud_mask_dbscan(detection_point, voxel_size=0.001)
    
    detection_point.paint_uniform_color([1, 0, 0]) # RED
    return detection_point

def calculate_orientation_error(pred_orient: np.ndarray, gt_orient: np.ndarray) -> float:
    """Calculate angle between predicted and ground truth orientation vectors (in degrees)."""
    # Normalize vectors
    pred_norm = pred_orient / np.linalg.norm(pred_orient)
    gt_norm = gt_orient / np.linalg.norm(gt_orient)
    
    # Calculate dot product (clamped to avoid numerical errors)
    dot_product = np.clip(np.dot(pred_norm, gt_norm), -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    print(f"Angle between predicted and ground truth orientation vectors: {angle_deg} degrees")
    return angle_deg

def compute_normal_from_points(mask_pcd, target_point):
    """
    Compute orientation vector from flattened mask_pcd by identifying 4 corners
    and computing object frame at target_point.
    
    Args:
        mask_pcd: Point cloud mask of detected object (should be flattened rectangle)
        target_point: Target point (centroid) where to compute orientation
        
    Returns:
        orient_vec: Normal vector representing object orientation in camera frame
    """
    if mask_pcd is None or len(mask_pcd.points) == 0:
        return None
    
    import numpy as np
    
    # Get point cloud data
    mask_pcd = preprocess_pcd(mask_pcd)
    points = np.asarray(mask_pcd.points)
    print(f"Computing normal from {len(points)} points")
    
    # Find the 4 corners of the rectangle
    # Method: Find extreme points in x and y directions
    x_min_idx = np.argmin(points[:, 0])
    x_max_idx = np.argmax(points[:, 0])
    y_min_idx = np.argmin(points[:, 1])
    y_max_idx = np.argmax(points[:, 1])
    
    # Get corner points
    corner1 = points[x_min_idx]  # Leftmost point
    corner2 = points[x_max_idx]  # Rightmost point
    corner3 = points[y_min_idx]  # Bottommost point
    corner4 = points[y_max_idx]  # Topmost point
    
    
    # Compute orientation vectors relative to target_point
    # Move corner points relative to target_point to form object frame
    corner1_rel = corner1 - target_point  # Vector from target to corner1
    corner2_rel = corner2 - target_point  # Vector from target to corner2
    corner3_rel = corner3 - target_point  # Vector from target to corner3
    corner4_rel = corner4 - target_point  # Vector from target to corner4
    
    # Compute two side vectors relative to target point
    side1 = corner2_rel - corner1_rel  # Vector along one side relative to target
    side2 = corner4_rel - corner3_rel  # Vector along other side relative to target
    
    # Determine which side is longer to establish x_axis and y_axis
    side1_length = np.linalg.norm(side1)
    side2_length = np.linalg.norm(side2)
        
    if side1_length >= side2_length:
        # side1 is the long side (x_axis), side2 is the short side (y_axis)
        x_axis = side1 / side1_length  # Normalize
        y_axis = side2 / side2_length   # Normalize
    else:
        # side2 is the long side (x_axis), side1 is the short side (y_axis)
        x_axis = side2 / side2_length  # Normalize
        y_axis = side1 / side1_length  # Normalize
    
    # Compute normal vector using cross product of x_axis and y_axis
    # This gives us the z-axis (normal to the surface)
    orient_vec = np.cross(x_axis, y_axis)
    
    # Ensure the normal vector points towards the camera (negative z direction)
    # This is typical for grasping applications
    if orient_vec[2] < 0:
        orient_vec = -orient_vec
    
    # Normalize the vector
    orient_vec = orient_vec / np.linalg.norm(orient_vec)
    
    print(f"X-axis: {x_axis}")
    print(f"Y-axis: {y_axis}")
    print(f"Computed orientation vector: {orient_vec}")
    return orient_vec



def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.001) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using a voxel grid of the given size (in meters).
    """
    #pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    
    #pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.01)
    pcd = pcd.voxel_down_sample(voxel_size)
    return pcd



def filter_cloud_mask_dbscan(detection_point: o3d.geometry.PointCloud, eps: float = 0.03, min_points: int = 20, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Filter a mask point cloud to keep only the densest DBSCAN cluster.
    Returns a new point cloud containing only points from the densest cluster.
    """
    if len(detection_point.points) == 0:
        return detection_point
    
   
    ds_cloud = detection_point
    
    if len(ds_cloud.points) == 0:
        return detection_point
        
    ds_points = np.asarray(ds_cloud.points)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(ds_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    if labels.size == 0 or labels.max() < 0:
        return detection_point
    
    # Find densest cluster among downsampled points
    max_label = labels.max()
    cluster_sizes = {label: np.sum(labels == label) for label in range(max_label + 1)}
    densest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    filtered_points = ds_points[labels == densest_cluster_label]
    
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered





def rgb_to_ply_path(rgb_path: str) -> str:
    base = os.path.splitext(os.path.basename(rgb_path))[0]
    ply_dir = os.path.dirname(rgb_path).replace('rgb', 'ply')
    return os.path.join(ply_dir, f"{base}.ply")


def get_depth_values_batch(image_path: str, coords: List[tuple], split='train') -> np.ndarray:
    """
    Fast depth fetch for multiple pixel coordinates in one shot.
    Returns an array of depth values (same units as the depth image), with -1 for out-of-bounds.
    """
    depth_path = image_path.replace('rgb', 'depth')
    if not os.path.isfile(depth_path):
        return np.full((len(coords),), -1, dtype=float)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        return np.full((len(coords),), -1, dtype=float)

    h, w = depth_img.shape[:2]
    out = np.empty((len(coords),), dtype=float)
    for i, (x, y) in enumerate(coords):
        if 0 <= int(x) < w and 0 <= int(y) < h:
            out[i] = float(depth_img[int(y), int(x)][0])
        else:
            out[i] = -1.0
    return out



def get_depth_line(coord, orient, color=[1,0,1]):
    
    cam_plane_point = np.array([coord[0], coord[1], 0.0], dtype=float)
    surface_point = np.array(coord, dtype=float)
    
    # Calculate camera ray direction (from camera plane to surface point)
    camera_ray = surface_point - cam_plane_point
    camera_ray_norm = np.linalg.norm(camera_ray)
    
    if camera_ray_norm > 1e-12:
        camera_ray = camera_ray / camera_ray_norm
    else:
        camera_ray = np.array([0.0, 0.0, 1.0])
    
    # Use camera ray direction for the line
    depth_len = float(max(0.0, coord[2]))
    line_end = cam_plane_point + camera_ray * depth_len
    depth_line_points = o3d.utility.Vector3dVector([
        cam_plane_point,
        line_end
    ])
    
    depth_line = o3d.geometry.LineSet(
        points=depth_line_points,
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    depth_line.colors = o3d.utility.Vector3dVector([color])


    return depth_line


def flatout_mask(mask_pcd, og_pcd):
    """
    Get the top 90 percentile of mask_pcd, compare with og_pcd in same area,
    interpolate with og_pcd to cover the flat surface of object.
    
    Args:
        mask_pcd: Point cloud mask of detected object
        og_pcd: Original full point cloud
        
    Returns:
        Interpolated flat surface point cloud
    """
    if mask_pcd is None or og_pcd is None or len(mask_pcd.points) == 0 or len(og_pcd.points) == 0:
        return mask_pcd
    
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata
    
    print("FLATTING OUT MASK")
    # Get top 90% of mask_pcd (closest to camera - lowest z values)
    mask_points = np.asarray(mask_pcd.points)
    z_coords = mask_points[:, 2]
    depth_percentile_90 = np.percentile(z_coords, 90)  # Top 90% closest to camera
    top_90_mask = z_coords <= depth_percentile_90
    top_90_points = mask_points[top_90_mask]
    
    if len(top_90_points) == 0:
        return mask_pcd
    
    # Get original point cloud points
    og_points = np.asarray(og_pcd.points)
    
    # Define spatial bounds of mask_pcd to find corresponding og_pcd points
    x_min, x_max = np.min(top_90_points[:, 0]), np.max(top_90_points[:, 0])
    y_min, y_max = np.min(top_90_points[:, 1]), np.max(top_90_points[:, 1])
    z_min, z_max = np.min(top_90_points[:, 2]), np.max(top_90_points[:, 2])
    
    # Add some tolerance to the bounds
    tolerance = 0.01  # 5cm tolerance
    x_min -= tolerance
    x_max += tolerance
    y_min -= tolerance
    y_max += tolerance
    z_min -= tolerance
    z_max += tolerance
    
    # Filter og_pcd points within the spatial bounds
    og_mask = (
        (og_points[:, 0] >= x_min) & (og_points[:, 0] <= x_max) &
        (og_points[:, 1] >= y_min) & (og_points[:, 1] <= y_max) &
        (og_points[:, 2] >= z_min) & (og_points[:, 2] <= z_max)
    )
    og_filtered_points = og_points[og_mask]
    
    if len(og_filtered_points) == 0:
        return mask_pcd
    
    # Create interpolated surface
    # Use the top 90% points as reference for interpolation
    try:
        # Create a grid for interpolation
        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Interpolate z values using the original point cloud
        # Use only x,y coordinates for interpolation
        points_xy = og_filtered_points[:, :2]  # x, y coordinates
        values_z = og_filtered_points[:, 2]    # z coordinates
        
        # Perform interpolation
        Z_interp = griddata(points_xy, values_z, (X_grid, Y_grid), method='linear', fill_value=np.nan)
        
        # Create interpolated points and filter to match og_pcd x,y coordinates
        interpolated_points = []
        og_points_xy = og_filtered_points[:, :2]  # x,y coordinates from original point cloud
        
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                if not np.isnan(Z_interp[i, j]):
                    interp_x, interp_y = X_grid[i, j], Y_grid[i, j]
                    interp_z = Z_interp[i, j] # -0.2
                    
                    # Check if this x,y coordinate exists in og_pcd (with small tolerance)
                    xy_tolerance = 0.01  # 1cm tolerance for x,y matching
                    distances = np.sqrt((og_points_xy[:, 0] - interp_x)**2 + (og_points_xy[:, 1] - interp_y)**2)
                    min_distance = np.min(distances)
                    
                    # Only keep points where x,y coordinates match og_pcd
                    if min_distance <= xy_tolerance:
                        interpolated_points.append([interp_x, interp_y, interp_z])
        
        if len(interpolated_points) > 0:
            # Create new point cloud with interpolated surface
            flat_surface_pcd = o3d.geometry.PointCloud()
            flat_surface_pcd.points = o3d.utility.Vector3dVector(np.array(interpolated_points))
            flat_surface_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for interpolated surface
            
            print(f"Flatout mask: kept {len(interpolated_points)} points with matching x,y coordinates")
            return flat_surface_pcd
        else:
            print("Flatout mask: no points with matching x,y coordinates found")
            return mask_pcd
            
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return mask_pcd

def visualize_pcd_mask(image_path, mask_pcd, target_point, normal_vector, x_axis_vector=None):
    ply_path = rgb_to_ply_path(image_path)
    try:
        # Load original point cloud
        pcd_original = o3d.io.read_point_cloud(ply_path)
        if len(pcd_original.points) > 0:
            # Transform original point cloud to camera frame
            try:
                pts_cloud = np.asarray(pcd_original.points)
                pts_cam = pts_cloud @ T_MAT.T  # cloud -> camera
                pcd_original.points = o3d.utility.Vector3dVector(pts_cam)
            except Exception:
                pass

            # Create geometries for visualization (like visualize_data.py)
            geometries = []
            
            # 1. Original PLY point cloud (light gray)
            #pcd_original.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
            geometries.append(pcd_original)
            
            # 2. Mask point cloud (red)
            if len(mask_pcd.points) > 0:
                mask_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
                geometries.append(mask_pcd)
            
            # 3. Camera plane point (blue)
            camera_plane_point = [target_point[0], target_point[1], 0.0]
            camera_plane_pcd = o3d.geometry.PointCloud()
            camera_plane_pcd.points = o3d.utility.Vector3dVector([camera_plane_point])
            camera_plane_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
            geometries.append(camera_plane_pcd)
            
            # 4. Surface point (red)
            surface_point = target_point
            surface_pcd = o3d.geometry.PointCloud()
            surface_pcd.points = o3d.utility.Vector3dVector([surface_point])
            surface_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            geometries.append(surface_pcd)
            
            # 5. Orientation vector (green arrow)
            if normal_vector is not None:
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.01,
                    cone_radius=0.02,
                    cylinder_height=0.15,
                    cone_height=0.05,
                )
                
                # Position arrow at surface point
                arrow.translate([target_point[0], target_point[1], target_point[2]])
                
                # Orient arrow along orientation vector
                orient = np.array(normal_vector, dtype=float)
                orient_norm = np.linalg.norm(orient)
                if orient_norm > 1e-12:
                    orient = orient / orient_norm
                    
                    # Create rotation matrix to align arrow with orientation
                    z_axis = np.array([0.0, 0.0, 1.0])
                    if np.allclose(orient, z_axis):
                        R = np.eye(3)
                    else:
                        axis = np.cross(z_axis, orient)
                        axis_norm = np.linalg.norm(axis)
                        if axis_norm < 1e-12:
                            R = np.eye(3)
                        else:
                            axis /= axis_norm
                            angle = np.arccos(np.clip(np.dot(z_axis, orient), -1.0, 1.0))
                            K = np.array([[0, -axis[2], axis[1]],
                                          [axis[2], 0, -axis[0]],
                                          [-axis[1], axis[0], 0]])
                            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    
                    arrow.rotate(R, center=[target_point[0], target_point[1], target_point[2]])
                
                arrow.paint_uniform_color([0.0, 1.0, 0.0])  # Green
                geometries.append(arrow)
            
            # 6. X-axis vector (yellow arrow) - parcel long side
            if x_axis_vector is not None:
                x_arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.008,
                    cone_radius=0.015,
                    cylinder_height=0.12,
                    cone_height=0.04,
                )
                
                # Position arrow at surface point
                x_arrow.translate([target_point[0], target_point[1], target_point[2]])
                
                # Orient arrow along X-axis vector
                x_orient = np.array(x_axis_vector, dtype=float)
                x_orient_norm = np.linalg.norm(x_orient)
                if x_orient_norm > 1e-12:
                    x_orient = x_orient / x_orient_norm
                    
                    # Create rotation matrix to align arrow with X-axis
                    z_axis = np.array([0.0, 0.0, 1.0])
                    if np.allclose(x_orient, z_axis):
                        R_x = np.eye(3)
                    else:
                        axis_x = np.cross(z_axis, x_orient)
                        axis_x_norm = np.linalg.norm(axis_x)
                        if axis_x_norm < 1e-12:
                            R_x = np.eye(3)
                        else:
                            axis_x /= axis_x_norm
                            angle_x = np.arccos(np.clip(np.dot(z_axis, x_orient), -1.0, 1.0))
                            K_x = np.array([[0, -axis_x[2], axis_x[1]],
                                            [axis_x[2], 0, -axis_x[0]],
                                            [-axis_x[1], axis_x[0], 0]])
                            R_x = np.eye(3) + np.sin(angle_x) * K_x + (1 - np.cos(angle_x)) * (K_x @ K_x)
                    
                    x_arrow.rotate(R_x, center=[target_point[0], target_point[1], target_point[2]])
                
                x_arrow.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
                geometries.append(x_arrow)
            
            # 7. Camera ray line (from camera plane to surface)
            cam_plane = np.array([target_point[0], target_point[1], 0.0])
            surface = np.array(target_point)
            
            # Create line from camera plane to surface
            line_points = o3d.utility.Vector3dVector([cam_plane, surface])
            line = o3d.geometry.LineSet(
                points=line_points,
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 1.0]])  # Cyan
            geometries.append(line)
            
            # 8. Coordinate frame at origin
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            geometries.append(coord_frame)
            
            # Visualize
            print(f"Visualizing detection vs point cloud: {ply_path}")
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"Detection vs PCD: {os.path.basename(image_path)}",
                lookat=np.array([0.0, 0.0, 0.0]),
                front=np.array([0.0, 0.0, -1.0]),
                up=np.array([0.0, -1.0, 0.0]),
                zoom=0.8
            )
        else:
            print("Empty point cloud:", ply_path)
    except Exception as e:
        print(f"Error loading PLY {ply_path}: {e}")



def visualize_detection(image_path, bbox_xyxy, z_d):
    # visualize box xyx - cv2
    tmp_img  = cv2.imread(image_path)
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
    x1f, y1f, x2f, y2f = bbox_xyxy
    x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
    depth_txt = f"{z_d:.3f} m" if z_d is not None and float(z_d) > 0 else "NA"
    label = f"depth={depth_txt}"
    cv2.rectangle(tmp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(tmp_img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(tmp_img, ( int( 0.5 * (x1 + x2)), int(0.5 * (y1 + y2))), 5, (0, 0, 255), -1)

    cv2.imshow('inference', tmp_img)
    key = cv2.waitKey(0) & 0xFF  # Non-blocking wait
    cv2.destroyAllWindows()


def find_best_grasp_point(mask_pcd):
    points = np.asarray(mask_pcd.points)
    # Use the point closest to the plane center

    depth_90pc = np.percentile(points[:, 2], 40)

    anchor_center = np.mean(points, axis=0)


    distances = np.linalg.norm(points - anchor_center, axis=1)
    best_idx = np.argmin(distances)
    return points[best_idx]


def load_model(ckpt_path: str, device: torch.device = None):
    """Load OrientRegressPointNet from a checkpoint path.

    Returns (model, meta) where meta contains optional fields like epoch and best_mean_err_deg.
    """
    print(f"[load] path={ckpt_path}")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = OrientRegressPointNet(feature_transform=True).to(device)

    # Support either full dict or under 'model_state'
    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        'epoch': ckpt.get('epoch', None),
        'best_mean_err_deg': ckpt.get('best_mean_err_deg', None)
    }
    print(f"[load] loaded. meta={meta}")
    return model, meta


def run_and_export(output_csv: str, split) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[main] device={device}")

    orient_ckpt = r'D:\ViettelAI\Code\finetune_yolo\PointNet\pointnetPytorch\utils\checkpoints\orient_best.pth'
    orient_model, meta = load_model(orient_ckpt, device)




    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoint')
    weights = find_latest_checkpoint(checkpoint_root)
    model = YOLO(weights)

    
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

                x_cam, y_cam, _ = get_cam_coord(center_uv, COLOR_INTRINSIC)
                        
                xyz_cam = list(map(float, (x_cam, y_cam, z_d)))
                x_cam, y_cam, z_cam = xyz_cam

                # Create cloud mask from bbox and estimate orientation
                bbox_xyxy = det_sel.get('bbox_xyxy', [])
                if not bbox_xyxy or len(bbox_xyxy) != 4:
                    print("No valid bbox found, using fallback orientation")
                    rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                
                else:
                    print("creating cloud mask")
                    #visualize_detection(image_path, bbox_xyxy, z_d)

                    # DEPTH CAM FRAME
                    mask_pcd = create_detection_point_cloud_mask(image_path, det_sel, filter_dense=True, filter_patch=True) # cloud
                    
                    #orient_pcd = mask_pcd.copy()
                    
                    print('Cloud mask size: ', len(mask_pcd.points))
                    
                    #mask_pcd = transform_to_camera_frame(mask_pcd)

                    ply_path = convert_path(image_path, src='rgb', target='ply')
                    og_pcd = o3d.io.read_point_cloud(ply_path)
                    if len(og_pcd.points) > 0:
                        # Transform to camera frame
                        pts_cloud = np.asarray(og_pcd.points)
                        pts_cam = pts_cloud @ T_MAT.T
                        og_pcd.points = o3d.utility.Vector3dVector(pts_cam)
                        mask_pcd = flatout_mask(mask_pcd, og_pcd)
                    else:
                        print("!!! NO FLAT SURFACE INTERPOLATION")

                    try:
                        if len(mask_pcd.points) > 0:
                            
                            x_cam, y_cam, z_cam = find_best_grasp_point(mask_pcd)
                            
                            xyz_cam = [x_cam, y_cam, z_cam]
                            
                            print("xyz_cam: ",  [x_cam, y_cam, z_cam])
                    except Exception as e:
                        print("error cloud mask: ", e)
                        pass
                    
                    if len(mask_pcd.points) > 0:
                        normal_vector = None
                    
                        try:
                            
                            #normal_vector = compute_normal_from_points(mask_pcd, xyz_cam)
                            #orient_pcd = create_detection_point_cloud_mask(image_path, det_sel, filter_dense=False, filter_patch=True)
                            orient_pts = np.array(mask_pcd.points) # (N,3)
    
                            point_set = torch.from_numpy(orient_pts.astype(np.float32))
                            point_set = point_set.unsqueeze(0) # (1,N,3)
                            point_set = point_set.transpose(2,1).to(device) # (1, 3,N)
                            with torch.no_grad():
                                pred_t, _, _ = orient_model(point_set) # (1,3)
                            
                            normal_vector = pred_t.detach().cpu().numpy()[0]
                        


                            if normal_vector is not None:
                                normal_vector = np.asarray(normal_vector) 
                               
                                # Use the normal vector directly as rx, ry, rz
                                # The normal vector is already oriented with X as parcel long side
                                #normal_vector = 0,0,1

                                # angle_error = calculate_orientation_error(normal_vector, [0,0,1])
                                # if angle_error < 10:
                                #     normal_vector = np.array([0,0,1])
                                #     print("normal set to [0,0,1]")

                                rx, ry, rz = normal_vector
                                
                            else:
                                rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                                normal_vector = [rx,ry,rz]
                            
                            # visualize mask_pcd with estimated normal and X-axis
                            visualize_pcd_mask(image_path, mask_pcd, xyz_cam, normal_vector)
                            

                        except Exception as e:
                            print(f"Error processing mask: {e}")
                            rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                    else:
                        rx, ry, rz = 0.0, 0.0, 1.0  # fallback
            else:
                rx, ry, rz = 0.0, 0.0, 1.0  # fallback

            print('orient: ', [rx,ry,rz])

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
    

    out_csv = os.path.join(os.path.dirname(__file__), "submit", "train_pred.csv")
    print(f"Saving to {out_csv}")
    run_and_export(out_csv, split='train')

