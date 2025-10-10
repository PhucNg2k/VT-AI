import os
try:
    from Code.config import *
except ImportError:
    from config import *

# Import depth lookup utility
try:
    from Code.camera_config import (
        get_depth_at_color_pixel,
        get_colorpix_depth_value,
        get_cam_coord,
        get_tmat_color2depth,
        compute_homogen_transform,
        get_pix_coord,
        DEPTH_INTRINSIC,
        DEPTH_TRAIN,
        DEPTH_TEST,
        read_img_np,
        COLOR_INTRINSIC,
    )
except ImportError:
    from camera_config import (
        get_depth_at_color_pixel,
        get_colorpix_depth_value,
        get_cam_coord,
        get_tmat_color2depth,
        compute_homogen_transform,
        get_pix_coord,
        DEPTH_INTRINSIC,
        DEPTH_TRAIN,
        DEPTH_TEST,
        read_img_np,
        COLOR_INTRINSIC,
    )

# 2D: image
# 2.5D: depth map
# 3D: point cloud


def get_file_list(path, n_img:None):
    img_list =  sorted(os.listdir(path), key=lambda x: int(os.path.splitext(x)[0]))
    img_list = [os.path.join(path, img) for img in img_list]

    if n_img > 0:
        return img_list[:n_img]
    return img_list


def get_depth_pixel(image_name, pixel_coord, direct=True, set='train'):
    if set == 'train':
        depth_dir = DEPTH_TRAIN
    else: 
        depth_dir = DEPTH_TEST

    # image_name can be a path or filename. Normalize to filename.
    fname = os.path.basename(image_name)
    #return get_depth_at_color_pixel(fname, pixel_coord, split='train' if set=='train' else 'test', search_radius=2, return_agg='median')
    return get_colorpix_depth_value(fname, pixel_coord, direct, split='train' if set=='train' else 'test')


def get_depth_pixel_batch(image_name, pixel_coords, direct=True, set='train'):
    """
    Batched depth fetch with single depth image load, following get_depth_pixel logic.
    - Maps color pixel(s) to depth pixel(s) via calibrated transforms
    - Supports 'direct' flag
    - Returns a list of floats (depth values); missing/out-of-bounds -> None
    """
    fname = os.path.basename(image_name)
    depth_src = DEPTH_TRAIN if set == 'train' else DEPTH_TEST
    try:
        depth_np = read_img_np(os.path.join(depth_src, fname))
    except Exception:
        depth_np = None
    if depth_np is None:
        return [None] * len(pixel_coords)

    tmat_color2depth = get_tmat_color2depth()
    results = []
    for uv in pixel_coords:
        try:
            og_u, og_v = int(uv[0]), int(uv[1])
            if direct:
                # Use original color pixel index directly
                if 0 <= og_v < depth_np.shape[0] and 0 <= og_u < depth_np.shape[1]:
                    results.append(float(depth_np[og_v, og_u]))
                else:
                    results.append(None)
            else:
                # Map color pixel to depth pixel via calibrated transforms
                color_cam = get_cam_coord(uv, COLOR_INTRINSIC)
                depth_cam = compute_homogen_transform(color_cam, tmat_color2depth)
                depth_uv = get_pix_coord(depth_cam, DEPTH_INTRINSIC)
                if depth_uv is None:
                    results.append(None)
                else:
                    du, dv = int(depth_uv[0]), int(depth_uv[1])
                    if 0 <= dv < depth_np.shape[0] and 0 <= du < depth_np.shape[1]:
                        results.append(float(depth_np[dv, du]))
                    else:
                        results.append(None)
        except Exception:
            results.append(None)
    return results