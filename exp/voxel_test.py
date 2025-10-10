# --- 3. Load Data and Setup Scene (Initial Load) ---
import numpy as np
import open3d as o3d

test_ply = r"D:\ViettelAI\Code\Public data\Public data train\ply\0002.ply"


pcd = o3d.io.read_point_cloud(test_ply)

pcd_np = np.asarray(pcd.points)

print(pcd_np.shape)
print(pcd_np[0])