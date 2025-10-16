import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import copy

# NOTE: Mocking external imports for runnable example
# from config import * # from file_utils import get_file_list 

# --- 1. Define Pose Parameters (Position and Orientation) ---



# --- PREDICTION TOGGLE ---
# Set to True to use Normal Vector Estimation from the Point Cloud at word_pos.
# Set to False to use the hardcoded approach_direction_vector.
USE_PCD_PREDICTION = True

# --- CRITICAL CONVENTION SETTING ---
# If your data convention means (0,0,1) points INTO the scene, and Open3D normals point OUT, 
# you likely still need to invert the resulting vector for the final grasp pose.
INVERT_APPROACH_VECTOR = True

# --- 2. Core Logic for Pose Prediction and Transformation ---

def predict_orientation_from_pcd(pcd, position, eps=0.03, min_points=10):
    """
    Dự đoán hướng gắp (Orientation Vector) bằng cách phân cụm DBSCAN trên PCD,
    xác định cụm chứa word_pos, sau đó ước tính vector pháp tuyến trung bình (Normal)
    cho cụm đó.
    
    Tham số:
    - eps: Khoảng cách tối đa (epsilon) giữa hai mẫu để coi chúng là láng giềng (DBSCAN).
    - min_points: Số lượng điểm tối thiểu để tạo thành một khu vực có mật độ cao (DBSCAN).
    """
    # 1. Phân cụm DBSCAN trên toàn bộ PCD
    print(f"Applying DBSCAN (eps={eps}, min_points={min_points})...")
    
    # Open3D's cluster_dbscan trả về chỉ số cụm cho mỗi điểm. -1 nghĩa là nhiễu (noise).
    # VerbosityContextManager được sử dụng để giảm bớt output của Open3D.
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    if max_label < 0:
        print("WARNING: DBSCAN found no clusters (only noise). Falling back to default vector.")
        return None

    # 2. Xác định cụm chứa vị trí gắp (position / word_pos)
    points = np.asarray(pcd.points)
    
    # Tìm điểm gần nhất với word_pos để lấy nhãn cụm (cluster label)
    distances = np.linalg.norm(points - position, axis=1)
    closest_point_index = np.argmin(distances)
    target_cluster_label = labels[closest_point_index]

    if target_cluster_label == -1:
        print("WARNING: word_pos's nearest neighbor is classified as noise (label -1). Falling back to default vector.")
        return None
        
    print(f"Target point's nearest neighbor belongs to Cluster ID: {target_cluster_label}")
    
    # 3. Lọc các điểm thuộc cụm mục tiêu
    inliers = labels == target_cluster_label
    cluster_points = points[inliers]
    
    if len(cluster_points) < 10:
        print(f"WARNING: Target cluster size ({len(cluster_points)}) is too small. Falling back to default vector.")
        return None

    # 4. Tạo đám mây điểm con (sub-point cloud) từ cluster
    local_pcd = o3d.geometry.PointCloud()
    local_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    
    # 5. Ước tính pháp tuyến trên cụm đã chọn
    # Sử dụng bán kính lớn hơn (ví dụ: eps * 2) để đảm bảo có đủ láng giềng.
    local_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=eps * 2, max_nn=30)
    )
    
    # 6. Tính Normal Vector trung bình
    normals = np.asarray(local_pcd.normals)
    if normals.size == 0:
        print("ERROR: Normal estimation failed on cluster points.")
        return None
        
    mean_normal = np.mean(normals, axis=0)
    
    # Chuẩn hóa kết quả
    predicted_vector = mean_normal / np.linalg.norm(mean_normal)
    print(f"PCD Prediction: Mean Normal Vector from Cluster calculated: {predicted_vector}")
    
    return predicted_vector

def construct_rotation_matrix_from_z_direction(target_z_vector):
    """
    Xây dựng Ma trận xoay 3x3 (R) từ vector hướng tiếp cận (Z-axis).
    """
    # 1. Define the local Z-axis (Approach Vector)
    Z_local = target_z_vector / np.linalg.norm(target_z_vector)
    
    # 2. Define a desired "right" reference vector (World X-axis)
    X_ref = np.array([1, 0, 0])
    
    # Handle the singularity
    if np.abs(np.dot(Z_local, X_ref)) > 0.9:
        X_ref = np.array([0, 1, 0])
    
    # 3. Calculate the local X-axis (Projection onto the plane perpendicular to Z_local)
    X_local_prime = X_ref - np.dot(X_ref, Z_local) * Z_local
    X_local = X_local_prime / np.linalg.norm(X_local_prime)
    
    # 4. Calculate the local Y-axis (Y = Z x X)
    Y_local = np.cross(Z_local, X_local)
    Y_local = Y_local / np.linalg.norm(Y_local)
    
    # 5. Build the rotation matrix [X_local | Y_local | Z_local]
    R_matrix = np.stack([X_local, Y_local, Z_local], axis=1)
    
    return R_matrix

def construct_homogeneous_transformation_matrix(rotation_matrix, translation_vector):
    """
    Xây dựng Ma trận biến đổi Thuần nhất (Homogeneous Transformation Matrix) 4x4 (T).
    """
    T_matrix = np.eye(4)
    T_matrix[:3, :3] = rotation_matrix
    T_matrix[:3, 3] = translation_vector
    return T_matrix

def create_local_coordinate_frame(position, rotation_matrix, size=0.05):
    """
    Tạo một hệ tọa độ (TriangleMesh) được biến đổi sử dụng Ma trận Xoay đã cung cấp.
    """
    local_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    local_frame.rotate(rotation_matrix, center=(0, 0, 0))
    local_frame.translate(position)
    return local_frame

# Placeholder functions for the 4x4 matrix usage (kept for completeness)
def transform_local_to_world(local_point_or_vector, T_matrix):
    if local_point_or_vector.shape == (3,):
        homogeneous_local = np.append(local_point_or_vector, 0.0) 
    else:
        return None 
    world_homogeneous = T_matrix @ homogeneous_local
    return world_homogeneous[:3]

def transform_world_to_local(world_point_or_vector, T_matrix):
    T_inverse = np.linalg.inv(T_matrix)
    if world_point_or_vector.shape == (3,):
        homogeneous_world = np.append(world_point_or_vector, 0.0)
    else:
        return None
    local_homogeneous = T_inverse @ homogeneous_world
    return local_homogeneous[:3]


# --- 3. Load Data and Setup Scene (Initial Load) ---
test_ply = "D:/ViettelAI/Public data task 3/Public data/Public data train/ply/0002.ply"


pcd = o3d.io.read_point_cloud(test_ply)
   

# Cần ước tính normals trên toàn bộ PCD trước khi dùng local prediction, 
# vì hàm `predict_orientation_from_pcd` sẽ ước tính cục bộ nếu cần.

# --- 4. Determine Rotation Matrix (R) Source ---

# Position for a realistic grasp point
word_pos = (-0.086,-0.141,1.046) 

gt_local_grasp_vector = np.array([-0.2642,-0.1098,0.9582]) 

if USE_PCD_PREDICTION:
    # OPredict orientation from point cloud normal using DBSCAN with word pos
    word_predicted_vector = predict_orientation_from_pcd(pcd, np.array(word_pos), eps=0.03, min_points=100)

    
# --- 5. Create Visualization Elements ---

# Xây dựng Ma trận biến đổi Thuần nhất 4x4 (T)
print("\n--- Ma trận biến đổi Thuần nhất 4x4 (World -> Local) ---")
# Calculate the rotation matrix R by assumption - will replace with 6D model
rotation_matrix = construct_rotation_matrix_from_z_direction(word_predicted_vector)
T_world_local = construct_homogeneous_transformation_matrix(rotation_matrix, np.array(word_pos))

predict_local_grasp_vector = transform_world_to_local(word_predicted_vector, T_world_local)


# a) Global Coordinate Frame (large reference)
frame_size = 0.5 
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])

# b) Local Grasp Coordinate Frame (small frame at the desired pose)
local_grasp_frame = create_local_coordinate_frame(
    position=word_pos, 
    rotation_matrix=rotation_matrix, 
    size=0.1 
)

# --- 6. Run Visualization and Test Transformation ---

# Vector kiểm tra trong Local Frame (Local Z-axis)
gt_world_vector = np.array([0.0, 0.0, -1.0])  #

# Thực hiện chuyển đổi Local -> World (sử dụng Ma trận 4x4)
gt_rot_mat = construct_rotation_matrix_from_z_direction(gt_world_vector)
gt_transform = construct_homogeneous_transformation_matrix(gt_rot_mat, np.array(word_pos))
gt_world_vector = transform_local_to_world(gt_world_vector, T_world_local)

print(f"\n--- Visualizing Grasp Pose ---")
print(f"Input Method: {'DBSCAN Cluster Normal Prediction' if USE_PCD_PREDICTION else 'Hardcoded Vector'}")
print("------------------------------------------------------------------")
print(f"GT Grasp vector: {gt_local_grasp_vector}")
print(f"Predict grasp vector: {predict_local_grasp_vector}")
print("------------------------------------------------------------------\n")

# o3d.visualization.draw_geometries(
#     [pcd, local_grasp_frame, world_frame], 
#     window_name="Point Cloud with Local Grasp Pose",
#     lookat=pcd.get_center(), 
#     zoom=0.1, 
#     point_show_normal=False # Tắt normal trên toàn bộ PCD để hình ảnh rõ ràng
# )
