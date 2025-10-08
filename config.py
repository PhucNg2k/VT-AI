import os

# Resolve data base relative to the repository root (this file is in Code/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PUB_DATA_BASE = os.path.abspath(os.path.join(REPO_ROOT, 'Public data task 3', 'Public data'))

PUB_TRAIN_BASE = os.path.join(PUB_DATA_BASE, "Public data train")
PUB_TEST_BASE = os.path.join(PUB_DATA_BASE, "Public data test")

print("Data base: ", PUB_DATA_BASE)


RGB_TRAIN = os.path.join(PUB_TRAIN_BASE, 'rgb')
PLY_TRAIN = os.path.join(PUB_TRAIN_BASE, 'ply')
DEPTH_TRAIN = os.path.join(PUB_TRAIN_BASE, 'depth')

RGB_TEST = os.path.join(PUB_TEST_BASE, 'rgb')
PLY_TEST = os.path.join(PUB_TEST_BASE, 'ply')
DEPTH_TEST = os.path.join(PUB_TEST_BASE, 'depth')

#######################

IMG_WIDTH, IMG_HEIGHT = 1280, 720
IMAGE_SIZE = (1280, 720)

ROI = (560, 150, 300, 330)  # (x, y, w, h)
HEIGHT_TOL_MM = 5.0


ROBOT_POS = (1000, 200) # u,v