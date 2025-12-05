"""
2D碰撞检测：使用 Separating Axis Theorem (SAT) 算法
用于检测旋转矩形之间的碰撞和计算穿透深度
"""

import numpy as np
from typing import Tuple


def get_rectangle_axes(corners: np.ndarray) -> np.ndarray:
    """
    获取矩形的两个轴向量（边的方向）
    
    参数:
        corners: (4, 2) 矩形的4个角点，按逆时针顺序
    
    返回:
        axes: (2, 2) 两个归一化的轴向量
    """
    # 边1: 从corner[0]到corner[1]
    edge1 = corners[1] - corners[0]
    # 边2: 从corner[1]到corner[2]  
    edge2 = corners[2] - corners[1]
    
    # 归一化
    axis1 = edge1 / np.linalg.norm(edge1)
    axis2 = edge2 / np.linalg.norm(edge2)
    
    return np.array([axis1, axis2])


def project_onto_axis(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """
    将矩形的所有角点投影到指定轴上，返回投影范围
    
    参数:
        corners: (4, 2) 矩形角点
        axis: (2,) 归一化的轴向量
    
    返回:
        min_proj, max_proj: 投影的最小值和最大值
    """
    projections = corners @ axis  # (4,)
    return np.min(projections), np.max(projections)


def compute_overlap(min1: float, max1: float, min2: float, max2: float) -> float:
    """
    计算两个区间在轴上的重叠量（穿透深度）
    
    参数:
        min1, max1: 第一个矩形的投影范围
        min2, max2: 第二个矩形的投影范围
    
    返回:
        overlap: 重叠量。如果不重叠返回0
    """
    # 检查是否重叠
    if max1 < min2 or max2 < min1:
        return 0.0
    
    # 计算重叠量：有两种推开方式，选择较小的
    overlap1 = max1 - min2  # 向右推矩形2
    overlap2 = max2 - min1  # 向左推矩形2
    
    return min(overlap1, overlap2)


def sat_collision_2d(corners1: np.ndarray, corners2: np.ndarray) -> Tuple[bool, float, np.ndarray]:
    """
    使用 Separating Axis Theorem (SAT) 检测两个矩形是否碰撞
    
    SAT原理：
    - 如果两个凸多边形在任意轴上的投影不重叠，则它们分离
    - 对于矩形，只需要检查4个轴（每个矩形的2条边的法向量）
    - 如果所有轴上都有重叠，则发生碰撞
    - MTV (Minimum Translation Vector) 是穿透深度最小的轴
    
    参数:
        corners1: (4, 2) 矩形1的角点，逆时针顺序
        corners2: (4, 2) 矩形2的角点，逆时针顺序
    
    返回:
        is_collision: 是否碰撞
        penetration_depth: 穿透深度（沿MTV方向）
        mtv: (2,) Minimum Translation Vector，将矩形1推出矩形2的最小位移向量
    """
    # 获取两个矩形的轴
    axes1 = get_rectangle_axes(corners1)  # (2, 2)
    axes2 = get_rectangle_axes(corners2)  # (2, 2)
    
    # 合并所有需要检查的轴
    all_axes = np.vstack([axes1, axes2])  # (4, 2)
    
    min_overlap = float('inf')
    mtv_axis = None
    
    # 检查每个轴
    for axis in all_axes:
        # 投影两个矩形到当前轴
        min1, max1 = project_onto_axis(corners1, axis)
        min2, max2 = project_onto_axis(corners2, axis)
        
        # 计算重叠量
        overlap = compute_overlap(min1, max1, min2, max2)
        
        # 如果在任何轴上不重叠，则矩形分离
        if overlap == 0:
            return False, 0.0, np.zeros(2)
        
        # 记录最小重叠（这就是MTV）
        if overlap < min_overlap:
            min_overlap = overlap
            mtv_axis = axis.copy()
            
            # 确定MTV方向：应该将矩形1推离矩形2
            # 计算两个矩形的中心
            center1 = np.mean(corners1, axis=0)
            center2 = np.mean(corners2, axis=0)
            center_diff = center1 - center2
            
            # 如果MTV方向与中心差向量方向相反，翻转MTV
            if np.dot(mtv_axis, center_diff) < 0:
                mtv_axis = -mtv_axis
    
    # 所有轴都有重叠，发生碰撞
    mtv = min_overlap * mtv_axis
    return True, min_overlap, mtv


def compute_collision_gradient(
    car_state: np.ndarray,
    obstacle: np.ndarray,
    car_length: float,
    car_width: float,
    penetration_depth: float,
    mtv: np.ndarray
) -> np.ndarray:
    """
    计算碰撞惩罚对车辆状态的梯度
    
    使用有限差分法计算 ∂(penetration_depth)/∂(x, y, θ)
    
    参数:
        car_state: (3,) [x, y, θ]
        obstacle: (5,) [cx, cy, w, l, θ]
        car_length, car_width: 车辆尺寸
        penetration_depth: 当前穿透深度
        mtv: (2,) 当前的最小平移向量
    
    返回:
        gradient: (3,) 梯度 [∂pd/∂x, ∂pd/∂y, ∂pd/∂θ]
    """
    from utils.commom import get_car_corners, get_obs_corners
    
    eps = 1e-6
    gradient = np.zeros(3)
    
    # 对每个状态变量进行数值微分
    for i in range(3):
        # 前向扰动
        state_plus = car_state.copy()
        state_plus[i] += eps
        
        corners_car_plus = get_car_corners(state_plus)
        corners_obs = get_obs_corners(obstacle)
        
        is_collision_plus, pd_plus, _ = sat_collision_2d(corners_car_plus, corners_obs)
        
        # 如果扰动后不再碰撞，穿透深度为0
        if not is_collision_plus:
            pd_plus = 0.0
        
        # 数值导数
        gradient[i] = (pd_plus - penetration_depth) / eps
    
    return gradient


def test_sat():
    """测试SAT算法"""
    print("=" * 70)
    print("测试 SAT 2D 碰撞检测")
    print("=" * 70)
    
    # 测试1：重叠的矩形
    print("\n测试1: 两个重叠的矩形")
    rect1 = np.array([
        [-1.0, -0.5],
        [ 1.0, -0.5],
        [ 1.0,  0.5],
        [-1.0,  0.5]
    ])
    
    rect2 = np.array([
        [-0.5, -0.5],
        [ 1.5, -0.5],
        [ 1.5,  0.5],
        [-0.5,  0.5]
    ])
    
    is_collision, pd, mtv = sat_collision_2d(rect1, rect2)
    print(f"碰撞: {is_collision}")
    print(f"穿透深度: {pd:.4f}")
    print(f"MTV: {mtv}")
    
    # 测试2：分离的矩形
    print("\n测试2: 两个分离的矩形")
    rect3 = np.array([
        [3.0, -0.5],
        [5.0, -0.5],
        [5.0,  0.5],
        [3.0,  0.5]
    ])
    
    is_collision2, pd2, mtv2 = sat_collision_2d(rect1, rect3)
    print(f"碰撞: {is_collision2}")
    print(f"穿透深度: {pd2:.4f}")
    print(f"MTV: {mtv2}")
    
    # 测试3：旋转的矩形
    print("\n测试3: 旋转45度的矩形")
    theta = np.pi / 4
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # 旋转矩阵
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    
    rect4_center = np.array([[0.5, 0.0]])  # 中心在(0.5, 0)
    rect4_local = np.array([
        [-1.0, -0.5],
        [ 1.0, -0.5],
        [ 1.0,  0.5],
        [-1.0,  0.5]
    ])
    
    rect4 = (rect4_local @ R.T) + rect4_center
    
    is_collision3, pd3, mtv3 = sat_collision_2d(rect1, rect4)
    print(f"碰撞: {is_collision3}")
    print(f"穿透深度: {pd3:.4f}")
    print(f"MTV: {mtv3}")
    print(f"MTV范数: {np.linalg.norm(mtv3):.4f}")


if __name__ == "__main__":
    test_sat()
