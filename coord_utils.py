from typing import Tuple, List
import numpy as np

try:
    from Code.config import ROI, ROBOT_POS
except ImportError:
    from config import ROI, ROBOT_POS


def in_roi(point_uv: Tuple[float, float], roi: Tuple[int, int, int, int] = ROI) -> bool:
    u, v = point_uv
    x, y, w, h = roi
    return (u >= x) and (v >= y) and (u < x + w) and (v < y + h)


def dist_from_robot(point_uv: Tuple[float, float], robot_uv: Tuple[float, float] = ROBOT_POS) -> float:
    du = float(point_uv[0]) - float(robot_uv[0])
    dv = float(point_uv[1]) - float(robot_uv[1])
    return float(np.hypot(du, dv))




def bbox_center_from_xywhr(xywhr: List[float]) -> Tuple[float, float]:
    # for YOLO OBB, center is provided as (cx, cy)
    return float(xywhr[0]), float(xywhr[1])


def bbox_center_from_polygon(polygon: List[List[float]]) -> Tuple[float, float]:
    arr = np.asarray(polygon, dtype=np.float64)
    cx = float(np.mean(arr[:, 0]))
    cy = float(np.mean(arr[:, 1]))
    return cx, cy