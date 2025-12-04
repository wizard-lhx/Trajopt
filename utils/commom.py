import numpy as np
from Kinematics.state_definitions import CAR_WIDTH, CAR_LENGTH

def get_car_corners(state: np.ndarray) -> np.ndarray:
    """
    根据车子的状态计算其在世界坐标系下的四个角点。
    """
    x, y, theta = state
    half_L = CAR_LENGTH / 2
    half_W = CAR_WIDTH / 2

    # 局部坐标系下的角点 (x, y)
    local_corners = np.array([
        [-half_L, -half_W],
        [ half_L, -half_W],
        [ half_L,  half_W],
        [-half_L,  half_W]
    ])

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # 转换到世界坐标系 (旋转后平移)
    world_corners = (local_corners @ rotation_matrix.T) + np.array([x, y])
    return world_corners

def get_obstacle_corners(obs: np.ndarray) -> np.ndarray:
    """
    根据障碍物数据计算其在世界坐标系下的四个角点。
    """
    cx, cy, w, l, theta = obs
    half_l = l / 2
    half_w = w / 2

    # 局部坐标系下的角点 (x, y)
    local_corners = np.array([
        [-half_l, -half_w],
        [ half_l, -half_w],
        [ half_l,  half_w],
        [-half_l,  half_w]
    ])

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # 转换到世界坐标系 (旋转后平移)
    world_corners = (local_corners @ rotation_matrix.T) + np.array([cx, cy])
    return world_corners

def convert_2d_to_3d(points_2d: np.ndarray) -> np.ndarray:
    """
    将2D点转换为3D点 (z=0)
    """
    num_points = points_2d.shape[0]
    points_3d = np.hstack((points_2d, np.zeros((num_points, 1))))
    return points_3d
def convert_3d_to_2d(points_3d: np.ndarray) -> np.ndarray:
    """
    将3D点转换为2D点 (丢弃z坐标)
    """
    return points_3d[:, :2]