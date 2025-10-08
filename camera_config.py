"""
color_intrinsics: Chứa các tham số nội tại của camera màu (RGB), 
    bao gồm tiêu cự (fx, fy), điểm chính (cx, cy), và các hệ số méo ảnh (coeffs).

depth_intrinsics: Chứa các tham số nội tại của camera độ sâu (Depth).

extrinsics_depth_to_color: Chứa các tham số ngoại tại, xác định phép biến đổi (xoay R và tịnh tiến t) từ hệ tọa độ của camera độ sâu
             sang hệ tọa độ của camera màu, cho phép căn chỉnh hai loại dữ liệu này với nhau.
"""
import numpy as np
import cv2 
import os
import open3d as o3d
try:
    from Code.config import *
except ImportError:
    from config import *

COLOR_INTRINSIC = {
    "width": 1280,
    "height": 720,
    "fx": 643.90087890625,
    "fy": 643.1365356445312,
    "cx": 650.2113037109375,
    "cy": 355.79559326171875,
    "model": "distortion.inverse_brown_conrady",
    "coeffs": [-0.05658450722694397, 0.06544225662946701,-0.0008694113348610699, 0.00016751799557823688,-0.020957745611667633]
}

DEPTH_INTRINSIC = {
    "width": 1280,
    "height": 720,
    "fx": 650.0616455078125,
    "fy": 650.0616455078125,
    "cx": 649.5928955078125,
    "cy": 360.9415588378906,
    "model": "distortion.brown_conrady",
    "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
}

# Extrinsic from DEPTH to COLOR
R_MATRIX = [
    [0.9999898076057434, -0.00020347206736914814, -0.004507721401751041],
    [0.00018898719281423837, 0.9999948143959045, -0.0032135415822267532],
    [0.004508351907134056, 0.003212657058611512, 0.9999846816062927]
]

T_VECTOR = [
    [-0.05905],
    [8.67399e-05],
    [0.00041]
]

def get_tmat_depth2color():
    R = np.asarray(R_MATRIX, dtype=np.float64)
    t = np.asarray(T_VECTOR, dtype=np.float64).reshape(3, 1)
    tmat = np.eye(4, dtype=np.float64)
    tmat[:3, :3] = R
    tmat[:3, 3:4] = t
    return tmat

def get_tmat_color2depth():
    R = np.asarray(R_MATRIX, dtype=np.float64)
    t = np.asarray(T_VECTOR, dtype=np.float64).reshape(3, 1)
    R_inv = R.T
    t_inv = -R_inv @ t
    tmat = np.eye(4, dtype=np.float64)
    tmat[:3, :3] = R_inv
    tmat[:3, 3:4] = t_inv
    return tmat


def get_homogen_vec(vec):
    # add 1 dim at the end
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    return np.concatenate([vec, np.array([1.0], dtype=np.float64)], axis=0)

def compute_homogen_transform(cam_coord, tmat):
    hv = get_homogen_vec(cam_coord).reshape(4, 1)
    out = (np.asarray(tmat, dtype=np.float64) @ hv).reshape(-1)
    return out[:3]

def read_img_np(image_path):
    img = o3d.io.readImage(image_path)
    return np.asarray(img)

def get_cam_coord(coord, intrinisic_mat=COLOR_INTRINSIC):
    if not len(coord) == 2:
        return None

    fx = float(intrinisic_mat["fx"]) 
    fy = float(intrinisic_mat["fy"]) 
    cx = float(intrinisic_mat["cx"]) 
    cy = float(intrinisic_mat["cy"]) 

    u, v = float(coord[0]), float(coord[1])
    z = 1.0
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    cam_coord = [x, y, z]

    if len(cam_coord) == 3:
        return cam_coord
    
def get_pix_coord(coord, initrinsic_mat=DEPTH_INTRINSIC):
    if not len(coord) == 3:
        return None

    fx = float(initrinsic_mat["fx"]) 
    fy = float(initrinsic_mat["fy"]) 
    cx = float(initrinsic_mat["cx"]) 
    cy = float(initrinsic_mat["cy"]) 

    x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
    if z == 0:
        return None

    u = x / z * fx + cx
    v = y / z * fy + cy

    width = int(initrinsic_mat.get("width", 0))
    height = int(initrinsic_mat.get("height", 0))
    u_i = int(round(u))
    v_i = int(round(v))
    if width > 0 and height > 0:
        u_i = max(0, min(width - 1, u_i))
        v_i = max(0, min(height - 1, v_i))
    cam_coord = [u_i, v_i]

    if len(cam_coord) == 2:
        return cam_coord


def unproject_pixel_to_cam(u, v, z, intrinsics):
    fx = float(intrinsics["fx"]) 
    fy = float(intrinsics["fy"]) 
    cx = float(intrinsics["cx"]) 
    cy = float(intrinsics["cy"]) 
    x = (float(u) - cx) / fx * float(z)
    y = (float(v) - cy) / fy * float(z)
    return np.array([x, y, float(z)], dtype=np.float64)


def project_cam_to_pixel(xyz, intrinsics):
    fx = float(intrinsics["fx"]) 
    fy = float(intrinsics["fy"]) 
    cx = float(intrinsics["cx"]) 
    cy = float(intrinsics["cy"]) 
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if z == 0:
        return None
    u = x / z * fx + cx
    v = y / z * fy + cy
    return np.array([u, v], dtype=np.float64)


def get_depth_at_color_pixel(image_name, color_uv, split="train", search_radius=2, return_agg="median"):
    """
    Retrieve depth at a color pixel by projecting nearby depth pixels into color frame.
    - color_uv: (u_c, v_c) in color image coordinates
    - search_radius: neighborhood radius in depth image to consider (pixels)
    - return_agg: aggregation for multiple matches: 'nearest' | 'mean' | 'median' | 'min' | 'max'
    """
    if split == 'train':
        depth_src = DEPTH_TRAIN
    else:
        depth_src = DEPTH_TEST

    depth_np = read_img_np(os.path.join(depth_src, image_name))
    if depth_np is None:
        return None

    h, w = depth_np.shape[0], depth_np.shape[1]
    u_c, v_c = int(round(color_uv[0])), int(round(color_uv[1]))
    matches = []

    # Precompute transform
    t_depth2color = get_tmat_depth2color()

    # Iterate small neighborhood in color image by back-projecting depth pixels around corresponding location in depth image space
    # Strategy: search local window in depth image (cheap, dense) and forward-project to color
    # Heuristic center: use same (u_c, v_c) as starting guess
    u0, v0 = u_c, v_c
    for dv in range(-search_radius, search_radius + 1):
        for du in range(-search_radius, search_radius + 1):
            u_d = u0 + du
            v_d = v0 + dv
            if u_d < 0 or v_d < 0 or u_d >= w or v_d >= h:
                continue
            z_d = float(depth_np[v_d, u_d])
            if z_d <= 0:
                continue
            # Unproject depth pixel to 3D in depth cam
            X_d = unproject_pixel_to_cam(u_d, v_d, z_d, DEPTH_INTRINSIC)
            # Transform to color cam
            X_c = compute_homogen_transform(X_d, t_depth2color)
            # Project to color pixel
            uv_c = project_cam_to_pixel(X_c, COLOR_INTRINSIC)
            if uv_c is None:
                continue
            u_proj, v_proj = int(round(uv_c[0])), int(round(uv_c[1]))
            if u_proj == u_c and v_proj == v_c:
                matches.append(z_d)

    if not matches:
        return None

    if return_agg == 'nearest':
        # Choose the sample whose projected color coords are nearest to (u_c, v_c)
        # In this implementation, all matches land exactly on (u_c, v_c), so just pick first
        return float(matches[0])
    if return_agg == 'mean':
        return float(np.mean(matches))
    if return_agg == 'min':
        return float(np.min(matches))
    if return_agg == 'max':
        return float(np.max(matches))
    # default median
    return float(np.median(matches))

def get_colorpix_depth_value(image_name, pix_coord, split="train"):
    '''
        image_name: '0001.png'
        pix_coord: (200,300) # (x,y)
    '''
    
    if split=='train':
        depth_src = DEPTH_TRAIN
    else:
        depth_src = DEPTH_TEST

    color_cam_coord = get_cam_coord(pix_coord, COLOR_INTRINSIC)
    
    tmat_color2depth = get_tmat_color2depth()
    
    depth_cam_coord = compute_homogen_transform(color_cam_coord, tmat_color2depth)

    depth_pix_coord = get_pix_coord(depth_cam_coord, DEPTH_INTRINSIC)

    depth_np = read_img_np(os.path.join(depth_src, image_name))
    if depth_pix_coord is None:
        return None
    u, v = depth_pix_coord[0], depth_pix_coord[1]
    # numpy images index as [row (v), col (u)]
    if v < 0 or u < 0 or v >= depth_np.shape[0] or u >= depth_np.shape[1]:
        return None
    depth_value = depth_np[v, u]
    
    return depth_value



# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import open3d as o3d

    bbox = (560, 150, 860, 480)  # example ROI

    points_3d_color, centroid, uv, colors = bbox_to_color_aligned_patch(depth, bbox, rgb)
    print(f"Centroid (x,y,z): {centroid}")
    print(f"Num valid points: {len(points_3d_color)}")
