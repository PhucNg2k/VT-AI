import os
import csv
import glob
from typing import List

import cv2
import numpy as np
import open3d as o3d

try:
    from Code.config import RGB_TEST, RGB_TRAIN
    from Code.file_utils import get_file_list, get_depth_pixel, get_depth_pixel_batch as _get_depth_batch
    from Code.phases.filter_phase import select_topmost_per_image
    from Code.camera_config import get_cam_coord, COLOR_INTRINSIC, PLY_MATRIX
    from Code.finetune_yolo.model_utils import detect_in_area_batch, detect_on_original_image
except ImportError:
    from config import RGB_TEST, RGB_TRAIN
    from file_utils import get_file_list, get_depth_pixel, get_depth_pixel_batch as _get_depth_batch
    from phases.filter_phase import select_topmost_per_image
    from camera_config import get_cam_coord, COLOR_INTRINSIC, PLY_MATRIX
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


def create_detection_point_cloud_mask(image_path: str, det_sel, split='train', default_depth: float = 1.0, filter_dense: bool = True, filter_patch=True) -> o3d.geometry.PointCloud:
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

    bbox_xyxy = det_sel.get('bbox_xyxy', [])
    center_uv = det_sel.get("center_uv", [])
    
    if bbox_xyxy and len(bbox_xyxy) == 4:
        x1, y1, x2, y2 = [int(coord) for coord in bbox_xyxy]
        
        # Get depth image path
        depth_path = image_path.replace('rgb', 'depth')
        if os.path.isfile(depth_path):
            # Load depth image as grayscale
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            # Expand bbox by padding on all sides, clamped to image bounds
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(depth_img.shape[1], x2 + pad)
            y2 = min(depth_img.shape[0], y2 + pad)
            
            # Build coordinate list and fetch depths in batch
            coords = [(x, y) for y in range(y1, y2) for x in range(x1, x2)]
            
            depths = _get_depth_batch(image_path, coords, direct=False, set=split)

            for (x, y), depth_value in zip(coords, depths):
                if depth_value is not None and depth_value > 0:
                    z_d = float(depth_value) / 1000.0 
                else:
                    z_d = default_depth  # Default depth if None or 0
                        
                        # Convert pixel to 3D camera coordinates
                x_cam_pixel, y_cam_pixel, _ = get_cam_coord([x, y], COLOR_INTRINSIC)
                detection_points_3d.append([x_cam_pixel, y_cam_pixel, z_d])
    
    # Create point cloud
    detection_point = o3d.geometry.PointCloud()
    if detection_points_3d:
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
        depth_percentile = np.percentile(z_coords, 40)  # Top 20% closest to camera
        top_surface_mask = z_coords <= depth_percentile
        top_surface_points = points[top_surface_mask]
        
        # Create new point cloud with filtered points
        detection_point = o3d.geometry.PointCloud()
        detection_point.points = o3d.utility.Vector3dVector(top_surface_points)
        
    # Optionally keep only the densest cluster from the mask
    if filter_dense and len(detection_point.points) > 0:
        detection_point = filter_cloud_mask_dbscan(detection_point, voxel_size=0.05)
    
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

def compute_normal_from_points(mask_pcd):
    """
    Compute surface normal (nx, ny, nz) from a smooth patch of point cloud.
    """
    if mask_pcd is None:
        return None
    
    print('estimating parcel orientation from point cloud ranges...')
    
    # Get point cloud data
    points = np.asarray(mask_pcd.points)
    print(f"Point cloud size: {len(points)}")

    # 1️⃣ Compute centroid
    centroid = np.mean(points, axis=0)

    # 2️⃣ Center the points
    pts_centered = points - centroid

    # 3️⃣ Compute covariance
    cov = np.cov(pts_centered.T)

    # 4️⃣ Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 5️⃣ Smallest eigenvector = surface normal
    normal = eigvecs[:, np.argmin(eigvals)]

    # 6️⃣ Normalize
    normal = normal / np.linalg.norm(normal)

    if normal[2] < 0:
        normal = -normal

    print(f"Orient normal: {normal}")
    
    angle_error = calculate_orientation_error(normal, [0,0,1])
    if angle_error > 20:
        normal = np.array([0,0,1])
        print("normal set to [0,0,1]")
    
    return normal

def predict_orientation_from_mask(mask_pcd, min_points=40,
                                 max_iterations=1000,
                                 threshold=0.03,
                                 bbox_xyxy=None,
                                 K=None):
    """
    Estimate parcel orientation using point cloud analysis:
    1. Identify X,Y sides from point cloud ranges
    2. Compute Z-axis pointing outward (negative Z)
    3. Apply RANSAC to get fitting plane
    4. Visualize coordinate system and plane

    Returns:
        normal (np.ndarray of shape (3,)): normalized normal vector (rx, ry, rz)
    """
    if mask_pcd is None or len(mask_pcd.points) < min_points:
        return None
    
    print('estimating parcel orientation from point cloud ranges...')
    
    # Get point cloud data
    points = np.asarray(mask_pcd.points)
    print(f"Point cloud size: {len(points)}")
    
    # --- Step 1: Compute parcel coordinate system from point cloud patch
    # Origin is at the center of the point cloud patch
    centroid = np.mean(points, axis=0)
    print(f"Point cloud centroid: {centroid}")
    
    # Compute ranges to determine long and short sides
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    
    print(f"X range: {x_range:.4f}, Y range: {y_range:.4f}")
    
    # Compute nx (long side direction vector) by traversing along the long side of point cloud
    if x_range >= y_range:
        # X is long side - direction vector along X direction
        nx = np.array([1, 0, 0])  # Direction vector along X direction (long side)
        ny = np.array([0, 1, 0])  # Direction vector along Y direction (short side)
        print("Long side: X direction")
    else:
        # Y is long side - direction vector along Y direction  
        nx = np.array([0, 1, 0])  # Direction vector along Y direction (long side)
        ny = np.array([1, 0, 0])  # Direction vector along X direction (short side)
        print("Long side: Y direction")
    
    # Compute nz by cross product of nx and ny
    nz = np.cross(nx, ny)
    nz /= np.linalg.norm(nz)
    
    # Ensure nz points outward (negative Z in camera frame)
    if nz[2] > 0:
        nz = -nz
    
    print(f"nx (long side direction vector): {nx}")
    print(f"ny (short side direction vector): {ny}")
    print(f"nz (outward direction vector): {nz}")
    
    # All direction vectors start from parcel origin (centroid) and are relative to camera frame
    
    # --- Step 3: Compute surface normal using PCA (more robust for smooth patches)
    surface_normal = compute_normal_from_points(points)
    
    if surface_normal is None:
        print("Failed to compute surface normal")
        return None
    
    print(f"Surface normal (PCA): {surface_normal}")
    
    # --- Step 4: Apply RANSAC for comparison/validation
    try:
        plane_model, inliers = mask_pcd.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=max_iterations
        )
    except Exception:
        print("RANSAC failed")
        return surface_normal  # Return PCA normal if RANSAC fails

    if plane_model is None or len(inliers) == 0:
        print("No plane found by RANSAC, using PCA normal")
        return surface_normal

    a, b, c, d = plane_model
    ransac_normal = np.array([a, b, c], dtype=float)
    ransac_normal /= np.linalg.norm(ransac_normal)
    
    print(f"RANSAC plane normal: {ransac_normal}")
    print(f"RANSAC plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    
    # Compare PCA and RANSAC normals
    dot_product = np.dot(surface_normal, ransac_normal)
    print(f"PCA vs RANSAC normal similarity: {dot_product:.4f}")
    
    # Use PCA normal as it's more robust for smooth patches
    return surface_normal




def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using a voxel grid of the given size (in meters).
    """
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    
    pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.01)
    #pcd = pcd.voxel_down_sample(voxel_size)
    return pcd

def estimate_smooth_normal(pcd, voxel_size=0.001):
    # Denoise
    #pcd = preprocess_pcd(pcd, voxel_size)

    # Smooth normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    # Aggregate
    normals = np.asarray(pcd.normals)
    avg_normal = np.mean(normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)

    # Flip toward camera
    #if avg_normal[2] > 0:
    #    avg_normal = -avg_normal
    return avg_normal



def filter_cloud_mask_dbscan(detection_point: o3d.geometry.PointCloud, eps: float = 0.03, min_points: int = 20, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Filter a mask point cloud to keep only the densest DBSCAN cluster.
    Returns a new point cloud containing only points from the densest cluster.
    """
    if len(detection_point.points) == 0:
        return detection_point
    
    # Downsample before clustering for speed and robustness
    if voxel_size > 0:
        ds_cloud = preprocess_pcd(detection_point, voxel_size=voxel_size)
    else:
        ds_cloud = detection_point
    
    if len(ds_cloud.points) == 0:
        return detection_point
    
    print('dbscaning ...')
    
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




def visualize_pcd_mask(image_path, mask_pcd, target_point, normal_vector, x_axis_vector=None):
    ply_path = rgb_to_ply_path(image_path)
    try:
        # Load original point cloud
        pcd_original = o3d.io.read_point_cloud(ply_path)
        if len(pcd_original.points) > 0:
            # Transform original point cloud to camera frame
            try:
                pts_cloud = np.asarray(pcd_original.points)
                pts_cam = (T_MAT @ pts_cloud.T).T  # cloud -> camera
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

def run_and_export(output_csv: str, split) -> None:
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

                    mask_pcd = create_detection_point_cloud_mask(image_path, det_sel, split=split, filter_dense=True, filter_patch=True) # cloud
                    print('Cloud mask size: ', len(mask_pcd.points))

                    try:
                        if len(mask_pcd.points) > 0:
                            mask_pts_cloud = np.asarray(mask_pcd.points)
                            # Use centroid of mask for xyz_cam
                            #centroid = mask_pts_cloud.mean(axis=0)
                            #x_cam, y_cam, z_cam = list(map(float, centroid))

                            #x_cam, y_cam, z_cam = find_best_grasp_point(mask_pcd)
                            xyz_cam = [x_cam, y_cam, z_cam]
                            
                            print("xyz_cam: ", xyz_cam)
                    except Exception as e:
                        print("error cloud mask: ", e)
                        pass
                    
                    if len(mask_pcd.points) > 0:
                        normal_vector = None
                        x_axis_vector = None
                        cam_normal_vector = None
                        try:
                            #normal_vector = predict_orientation_from_mask(mask_pcd, bbox_xyxy=bbox_xyxy, K=COLOR_INTRINSIC)
                            normal_vector = compute_normal_from_points(mask_pcd)
                            
                            if normal_vector is not None:
                                normal_vector = np.asarray(normal_vector) 
                               
                                # Use the normal vector directly as rx, ry, rz
                                # The normal vector is already oriented with X as parcel long side
                                #normal_vector = 0,0,1
                                rx, ry, rz = normal_vector
                                
                            else:
                                rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                            
                            # visualize mask_pcd with estimated normal and X-axis
                            #visualize_pcd_mask(image_path, mask_pcd, xyz_cam, normal_vector, x_axis_vector)
                            

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
    

    out_csv = os.path.join(os.path.dirname(__file__), "submit", "Submission_3D.csv")
    print(f"Saving to {out_csv}")
    run_and_export(out_csv, split='test')

