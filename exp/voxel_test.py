# --- 3. Load Data and Setup Scene (Initial Load) ---
import numpy as np
import os
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from Code.config import RGB_TEST, RGB_TRAIN, PLY_TEST, PLY_TRAIN, ROI
from Code.file_utils import get_file_list
from Code.phases.filter_phase import select_topmost_per_image
from Code.camera_config import get_cam_coord, COLOR_INTRINSIC, PLY_MATRIX, get_tmat_color2depth, compute_homogen_transform, get_pix_coord, DEPTH_INTRINSIC, R_MATRIX
from Code.finetune_yolo.model_utils import detect_in_area_batch, detect_on_original_image

T_MAT = np.asarray(PLY_MATRIX, dtype=float)
T_INV = np.linalg.inv(T_MAT)

R_MATRIX = np.asarray(R_MATRIX, dtype=float)


def rgb_to_ply_path(rgb_path: str) -> str:
    base = os.path.splitext(os.path.basename(rgb_path))[0]
    ply_dir = os.path.dirname(rgb_path).replace('rgb', 'ply')
    return os.path.join(ply_dir, f"{base}.ply")

def convert_path(f_path: str, src, target) -> str:
    base = os.path.splitext(os.path.basename(f_path))[0]
    depth_dir = os.path.dirname(f_path).replace(src, target)
    ext = 'ply' if target == 'ply' else 'png'
    return os.path.join(depth_dir, f"{base}.{ext}")

def ply_to_rgb_path(ply_path: str) -> str:
    base = os.path.splitext(os.path.basename(ply_path))[0]
    rgb_dir = os.path.dirname(ply_path).replace('ply', 'rgb')
    return os.path.join(rgb_dir, f"{base}.png")

def transform_to_camera_frame(pts_cloud):
    return (T_MAT @ pts_cloud.T).T

def eda_pointcloud(points):
    # extract min, max range of z value of point
    # calculate histogram with range min, max
    # plot the histograme to see depth distribution
    
    
    if points is None:
        return None

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be an array of shape (N, 3) or wider")

    z_values = pts[:, 2]
    z_values = z_values[np.isfinite(z_values)]
    if z_values.size == 0:
        return None

    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))

    counts, bin_edges = np.histogram(z_values, bins=100, range=(z_min, z_max))

    plt.figure(figsize=(8, 4))
    plt.hist(z_values, bins=100, range=(z_min, z_max), color='steelblue', edgecolor='black', alpha=0.8)
    plt.title('Depth (Z) Distribution')
    plt.xlabel('Z (meters)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {"z_min": z_min, "z_max": z_max, "counts": counts, "bin_edges": bin_edges}

def read_depth_batch(image_path, coord_list):
    """Read depth values for a list of coordinates"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return [None] * len(coord_list)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    depth_values = []
    for coord in coord_list:
        x, y = int(coord[0]), int(coord[1])
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            depth_values.append(img[y, x] / 1000.0)  # Convert to meters
        else:
            depth_values.append(None)
    
    return depth_values

def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using a voxel grid of the given size (in meters).
    """
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    
    pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.01)
    pcd = pcd.voxel_down_sample(voxel_size)


    points = np.asarray(pcd.points)
    z_coords = points[:, 2]
    depth_percentile = np.percentile(z_coords, 80)  # Top 20% closest to camera
    top_surface_mask = z_coords <= depth_percentile
    top_surface_points = points[top_surface_mask]
    
    # Create new point cloud with filtered points
    filterd_point = o3d.geometry.PointCloud()
    filterd_point.points = o3d.utility.Vector3dVector(top_surface_points)
    filterd_point.paint_uniform_color([0.0, 1.0, 0.0])

    return filterd_point


if __name__ == "__main__":
    ply_paths = get_file_list(PLY_TEST, -1)


    for ply_path in ply_paths:
        print("processing: ", os.path.basename(ply_path))
        pcd_original = o3d.io.read_point_cloud(ply_path)
        pts_cloud = np.asarray(pcd_original.points)

        pts_cam = pts_cloud @ T_MAT  # cloud -> camera
        pcd_original.points = o3d.utility.Vector3dVector(pts_cam)



        x,y,w,h = list(map(int, ROI))
        
        # Get depth image
        depth_path = convert_path(ply_path, src='ply', target='depth')
        depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_np is None:
            print(f"Could not load depth image: {depth_path}")
            continue
            
      
        # Generate color camera coordinates for ROI pixels
        color_3dpoints = []
        depth_3dpoints = []
        
        tmat_color2depth = get_tmat_color2depth()

        bbox_in_patch = []
        
        for py in range(y, y+h):
            for px in range(x, x+w):
                # Get 3D camera coordinates from color pixel (with z=1 for ray direction)
                xyz_cam = get_cam_coord([px, py], COLOR_INTRINSIC)
                
                # Transform color camera coordinates to depth camera coordinates
                depth_cam_coord = compute_homogen_transform(xyz_cam, tmat_color2depth)
                
                # Project depth camera coordinates to depth pixel coordinates
                depth_pix_coord = get_pix_coord(depth_cam_coord, DEPTH_INTRINSIC)
                
                x_d = float(depth_cam_coord[0])
                y_d = float(depth_cam_coord[1])

                if depth_pix_coord is None:
                    continue

                u_d, v_d = int(depth_pix_coord[0]), int(depth_pix_coord[1])
            

                if 0 <= v_d < depth_np.shape[0] and 0 <= u_d < depth_np.shape[1]:
                    # Get depth value and convert to meters
                    depth_value = depth_np[v_d, u_d][0] # convert mm to meters
                    if depth_value <= 0:
                        continue
                    bbox_in_patch.append([x_d, y_d, depth_value])
                    
                    
                    if depth_value >= 1150 and depth_value <= 1200: 
                        continue
                        

                    depth_value -= 100
                    depth_value /= 1000
                    # Update the Z coordinate with actual depth    
                    depth_3dpoints.append([x_d, y_d, depth_value ])
                
                    xyz_cam = [x_d, y_d, depth_value] @ R_MATRIX
                    color_3dpoints.append(xyz_cam)

        # Convert lists to numpy arrays for transformation
        color_3dpoints_array = np.array(color_3dpoints)
        depth_3dpoints_array = np.array(depth_3dpoints)
        
        # Transform to camera frame (if needed - check if points are already in camera frame)
        # Note: get_cam_coord already returns camera coordinates, so this might be redundant
        color_3dpoints_cam = color_3dpoints_array  # Already in camera frame
        depth_3dpoints_cam = depth_3dpoints_array  # Already in camera frame

        color_pcd = o3d.geometry.PointCloud()
        color_pcd.points = o3d.utility.Vector3dVector(color_3dpoints_cam)
        color_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # green
        
        color_pcd = preprocess_pcd(color_pcd, 0.0001)


        depth_pcd = o3d.geometry.PointCloud()
        depth_pcd.points = o3d.utility.Vector3dVector(depth_3dpoints_cam)
        depth_pcd.paint_uniform_color([1.0, 0.0, 0.0])   # red

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

        geometries = [pcd_original, color_pcd, coord_frame]


        eda_pointcloud(bbox_in_patch)

        print(f"Visualizing detection vs point cloud: {ply_path}")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Detection vs PCD",
            lookat=np.array([0.0, 0.0, 0.0]),
            front=np.array([0.0, 0.0, -1.0]),
            up=np.array([0.0, -1.0, 0.0]),
            zoom=0.8
        )

        



