import os
try:
    from Code.config import *
except ImportError:
    from config import *

# 2D: image
# 2.5D: depth map
# 3D: point cloud


def get_file_list(path, n_img:None):
    img_list =  sorted(os.listdir(path), key=lambda x: int(os.path.splitext(x)[0]))
    img_list = [os.path.join(path, img) for img in img_list]

    if n_img > 0:
        return img_list[:n_img]
    return img_list


def get_depth_pixel(image_name, pixel_coord, set='train'):
    if set == 'train':
        depth_dir = DEPTH_TRAIN
    else: 
        depth_dir = DEPTH_TEST

    # image_name can be a path or filename. Normalize to filename.
    fname = os.path.basename(image_name)
    return get_depth_at_color_pixel(fname, pixel_coord, split='train' if set=='train' else 'test', search_radius=2, return_agg='median')