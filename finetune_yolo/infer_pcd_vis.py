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


def create_detection_point_cloud_mask(image_path: str, det_sel, split='train', default_depth: float = 1.0, filter_dense: bool = True) -> o3d.geometry.PointCloud:
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
                    z_d = float(depth_value - 20) / 1000.0 
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
    
    # Optionally keep only the densest cluster from the mask
    if filter_dense and len(detection_point.points) > 0:
        detection_point = filter_cloud_mask_dbscan(detection_point, voxel_size=0.01)
    
    detection_point.paint_uniform_color([1, 0, 0]) # RED
    return detection_point


def predict_orientation_from_mask(mask_pcd, min_points=20,
                                  max_iterations=1000,
                                  threshold=0.01,
                                  z_flip_to_camera=True):
    """
    Hybrid RANSAC + PCA normal estimation for a parcel patch.

    1. Fit a robust plane with RANSAC (coarse normal)
    2. Run PCA on inliers to refine tangent axes
    3. Enforce orthogonality, flip normal to face camera if needed

    Returns:
        normal (np.ndarray of shape (3,)): normalized normal (rx, ry, rz)
    """
    if mask_pcd is None or len(mask_pcd.points) < min_points:
        return None

    # --- 1️⃣ RANSAC plane fitting (robust coarse normal)
    try:
        plane_model, inliers = mask_pcd.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=max_iterations
        )
    except Exception:
        return None

    if plane_model is None or len(inliers) == 0:
        return None

    a, b, c, d = plane_model
    z_axis_ransac = np.array([a, b, c], dtype=float)
    z_axis_ransac /= np.linalg.norm(z_axis_ransac)

    # --- 2️⃣ PCA refinement on inlier points
    inlier_cloud = mask_pcd.select_by_index(inliers)
    points = np.asarray(inlier_cloud.points)
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid

    if pts_centered.shape[0] < 3:
        return z_axis_ransac

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    x_axis_pca, y_axis_pca, _ = eigvecs.T

    # --- 3️⃣ Reconstruct orthogonal frame, align Z with plane
    z_axis = z_axis_ransac
    y_axis = np.cross(z_axis, x_axis_pca)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # --- 4️⃣ Flip Z to face camera (camera at origin)
    view_dir = -centroid  # from parcel → camera
    if np.dot(z_axis, view_dir) < 0 and z_flip_to_camera:
        z_axis = -z_axis
        x_axis = -x_axis  # maintain right-handedness

    # Return refined normal only (rx, ry, rz)
    return z_axis_ransac




def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using a voxel grid of the given size (in meters).
    """
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    
    pcd, _ = pcd.remove_radius_outlier(nb_points=20, radius=0.01)
    pcd = pcd.voxel_down_sample(voxel_size)
    return pcd

def estimate_smooth_normal(pcd, voxel_size=0.002):
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
    orient_dir = np.array(orient, dtype=float)
    orient_norm = np.linalg.norm(orient_dir)
    
    if orient_norm > 1e-12:
        orient_dir = orient_dir / orient_norm
    else:
        orient_dir = np.array([0.0, 0.0, 1.0])
    depth_len = float(max(0.0, coord[2]))

    line_end = cam_plane_point + orient_dir * depth_len
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

def visualize_pcd_mask(image_path, mask_pcd, target_point, normal_vector):
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

            # Target point already in camera frame
            target_point_cam = np.array(target_point)
            
            # Create coordinate frame at origin
            w_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

            # Create target point visualization
            target_point_pcd = o3d.geometry.PointCloud()
            target_point_pcd.points = o3d.utility.Vector3dVector([target_point_cam])
            target_point_pcd.paint_uniform_color([0, 1, 1])  
            
            # Create ground truth visualization (for testing)
            gt_coord = (-0.058, 0.043, 1.059)  # already camera frame
            gt_orient = (0, 0, 1)            # already camera frame
            
            gt_depth_line = get_depth_line(gt_coord, gt_orient, [0,1,0])
            target_depth_line = get_depth_line(target_point, normal_vector, [1,0,0])

            # Minimal: visualize predicted normal (as-is, not converted) from mask surface
            surf_normal_arrow = None
            tangent_plane = None
            if normal_vector is not None and len(mask_pcd.points) > 0:
                try:
                    centroid = np.asarray(mask_pcd.points).mean(axis=0)
                    n = np.asarray(normal_vector, dtype=float)
                    n /= (np.linalg.norm(n) + 1e-12)

                    # build arrow and align +Z to n
                    surf_normal_arrow = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=0.004,
                        cone_radius=0.008,
                        cylinder_height=0.07,
                        cone_height=0.03,
                    )
                    z_axis = np.array([0.0, 0.0, 1.0])
                    if np.allclose(n, z_axis):
                        R = np.eye(3)
                    else:
                        axis = np.cross(z_axis, n)
                        axis_norm = np.linalg.norm(axis)
                        if axis_norm < 1e-12:
                            R = np.eye(3)
                        else:
                            axis /= axis_norm
                            angle = np.arccos(np.clip(np.dot(z_axis, n), -1.0, 1.0))
                            K = np.array([[0, -axis[2], axis[1]],
                                          [axis[2], 0, -axis[0]],
                                          [-axis[1], axis[0], 0]])
                            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    surf_normal_arrow.rotate(R, center=[0, 0, 0])
                    surf_normal_arrow.translate(centroid)
                    surf_normal_arrow.paint_uniform_color([1.0, 0.5, 0.0])  # orange

                    # Tangent plane (yellow) centered at centroid, oriented by n
                    plane_size = 0.12
                    plane_thickness = 0.001
                    tangent_plane = o3d.geometry.TriangleMesh.create_box(
                        width=plane_size, height=plane_size, depth=plane_thickness
                    )
                    # center the box at origin first
                    tangent_plane.translate([-plane_size/2.0, -plane_size/2.0, -plane_thickness/2.0])
                    tangent_plane.rotate(R, center=[0, 0, 0])
                    tangent_plane.translate(centroid)
                    tangent_plane.paint_uniform_color([1.0, 1.0, 0.0])
                except Exception:
                    surf_normal_arrow = None
                    tangent_plane = None

            # Prepare geometries for visualization
            geometries = [pcd_original, mask_pcd, target_depth_line, w_axis]
            if surf_normal_arrow is not None:
                geometries.append(surf_normal_arrow)
            if tangent_plane is not None:
                geometries.append(tangent_plane)
        
            # Visualize together (now centered at origin)
            print(f"Visualizing detection vs point cloud: {ply_path}")
            o3d.visualization.draw_geometries(
                geometries, 
                window_name=f"Detection vs PCD: {os.path.basename(image_path)}",
                lookat=np.array([0.0, 0.0, 0.0]),  # Look at origin
                front=np.array([0.0, 0.0, -1.0]),  
                up=np.array([0.0, -1.0, 0.0]),     
                zoom=-3
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

    cv2.imshow('inference', tmp_img)
    key = cv2.waitKey(0) & 0xFF  # Non-blocking wait
    cv2.destroyAllWindows()


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
                    #visualize_detection(image_path, bbox_xyxy, z_d, split)

                    mask_pcd = create_detection_point_cloud_mask(image_path, det_sel, split=split, filter_dense=True) # cloud
    
                    try:
                        if len(mask_pcd.points) > 0:
                            mask_pts_cloud = np.asarray(mask_pcd.points)
                            # Use centroid of mask for xyz_cam
                            centroid = mask_pts_cloud.mean(axis=0)
                            x_cam, y_cam, z_cam = list(map(float, centroid))
                            xyz_cam = [x_cam, y_cam, z_cam]
                            print("xyz_cam (centroid): ", xyz_cam)
                    except Exception as e:
                        print("error cloud mask: ", e)
                        pass

                    if len(mask_pcd.points) > 0:
                        normal_vector = None
                        cam_normal_vector = None
                        try:
                            print("predicting orient normal..")
                            normal_vector = predict_orientation_from_mask(mask_pcd)
                            #normal_vector = estimate_smooth_normal(mask_pcd)

                            if normal_vector is not None:
                                normal_vector = np.asarray(normal_vector) # cloud coord
                                print("Estimated normal vector: ", normal_vector)
                               
                                cam_normal_vector = T_MAT @ normal_vector
                                cam_normal_vector /= np.linalg.norm(cam_normal_vector)
                                
                                print("Camera normal vector: ", cam_normal_vector)
                                rx, ry, rz = normal_vector
                                
                            else:
                                rx, ry, rz = 0.0, 0.0, 1.0  # fallback
                            
                            # visualize mask_pcd with estimated normal
                            visualize_pcd_mask(image_path, mask_pcd, xyz_cam, normal_vector)
                            

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
    out_csv = os.path.join(os.path.dirname(__file__), "submit", "Submit_test_pcd_hybrid-1.csv")
    print(f"Saving to {out_csv}")
    run_and_export(out_csv, split='test')

