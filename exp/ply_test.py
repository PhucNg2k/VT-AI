PLY_MATRIX = [
    [1, 0, 0],
    [0,-1, 0],
    [0, 0,-1]
 ]

import os
import numpy as np
import open3d as o3d

try:
    from Code.config import RGB_TRAIN
    from Code.file_utils import get_file_list
except ImportError:
    from config import RGB_TRAIN
    from file_utils import get_file_list


def load_and_transform_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd is None or len(pcd.points) == 0:
        return None
    points = np.asarray(pcd.points)
    T = np.asarray(PLY_MATRIX, dtype=float)
    points_cam = points @ T.T
    pcd.points = o3d.utility.Vector3dVector(points_cam)
    return pcd


def rgb_to_ply_path(rgb_path: str) -> str:
    base = os.path.splitext(os.path.basename(rgb_path))[0]
    ply_dir = os.path.dirname(rgb_path).replace('rgb', 'ply')
    return os.path.join(ply_dir, f"{base}.ply")


def main():
    rgb_images = get_file_list(RGB_TRAIN, -1)
    if not rgb_images:
        print("No training RGB images found at:", RGB_TRAIN)
        return

    for img_path in rgb_images:
        ply_path = rgb_to_ply_path(img_path)
        if not os.path.isfile(ply_path):
            # Skip if corresponding PLY is missing
            continue

        pcd = load_and_transform_point_cloud(ply_path)
        if pcd is None:
            print("Failed to load or empty point cloud:", ply_path)
            continue

        print("Visualizing:", ply_path)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, axis], lookat=pcd.get_center(), zoom=0.1)
        break


if __name__ == "__main__":
    main()

 