# 03_Costs_Constraints/no_collision_cost.py

import numpy as np
from typing import Tuple, List
# 假设我们已经从 Environment 文件夹导入了障碍物数据
from Environment.obstacles import OBSTACLES
from utils.commom import get_obstacle_corners,get_car_corners, convert_2d_to_3d
from utils.collision_2d import sat_collision_2d

from distance3d import gjk, epa, colliders

# --- 核心碰撞检测和线性化函数 ---

def compute_signed_distance(
    car_state: np.ndarray, 
    obs_data: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    【TrajOpt 核心函数：替代 GJK/EPA 的调用】

    该函数必须返回:
    1. sd: 带符号距离 (Signed Distance)
    2. n_hat: 接触法线 (Contact Normal) - 必须是单位向量
    3. pA_local: 小车上最近点在小车局部坐标系下的位置
    4. pB_world: 障碍物上最近点在世界坐标系下的位置
    """
    
    # 获取小车和障碍物的凸形状表示
    car_corners = get_car_corners(car_state)
    obs_corners = get_obstacle_corners(obs_data)
    
    collider_car = colliders.ConvexHullVertices(convert_2d_to_3d(car_corners))
    collider_obs = colliders.ConvexHullVertices(convert_2d_to_3d(obs_corners))
    
    # 使用 GJK 计算距离
    dist, pA_world, pB_world, simplex = gjk.gjk(collider_car, collider_obs)
    
    if dist > 0:
        # 分离状态
        sd = dist
        n_hat = (pB_world - pA_world) / np.linalg.norm(pB_world - pA_world)
    else:
        # # 碰撞状态，使用 EPA 计算最小平移向量 (MTV)
        # mtv, _, success = epa.epa(simplex, collider_car, collider_obs)
        # assert success, "EPA failed to compute MTV in collision state."
        # 2D 情况下用 SAT 代替 EPA 算法计算 MTV
        is_collision, penetration_depth, mtv = sat_collision_2d(car_corners, obs_corners)
        print(pA_world, pB_world)
        sd = -np.linalg.norm(mtv)
        n_hat = -mtv / np.linalg.norm(mtv)
    
    # 计算 pA_local (小车局部坐标系下的点)
    R_inv = np.array([
        [np.cos(-car_state[2]), -np.sin(-car_state[2])],
        [np.sin(-car_state[2]),  np.cos(-car_state[2])]
    ])
    pA_local = R_inv @ (pA_world[:2] - car_state[:2])
    pB_world = pB_world[:2]
    n_hat = n_hat[:2]  # 只保留2D部分
    
    return sd, n_hat, pA_local, pB_world


def get_collision_penalty_gradient(
    car_state: np.ndarray, 
    obs_data: np.ndarray,
    d_safe: float = 0.1 
) -> np.ndarray:
    """
    计算碰撞惩罚项 |d_safe - sd(x)|+ 对状态 x 的梯度 (线性化)。
    这是 TrajOpt 构造 QP 子问题中碰撞约束项的关键。
    """
    
    sd, n_hat, pA_local, pB_world = compute_signed_distance(car_state, obs_data)
    
    # 只有当 sd < d_safe 时，惩罚项才非零，才需要计算梯度
    if sd >= d_safe:
        return np.zeros(car_state.shape)

    # 计算线性化所需的 Jacobian J_pA(x)
    # J_pA 是小车上点 pA 的世界坐标对小车状态 x 的雅可比 J_pA = d(Fw * pA) / d(x)
    
    # J_pA 结构是 2x3: d(pA_world)/d(x,y,theta)
    J_pA = np.zeros((2, 3))
    
    # 1. d(pA_world)/d(x, y) = I (平移部分)
    J_pA[0, 0] = 1.0
    J_pA[1, 1] = 1.0
    
    # 2. d(pA_world)/d(theta) (旋转部分)
    # pA_local = [px, py]
    px, py = pA_local
    J_pA[0, 2] = -px * np.sin(car_state[2]) - py * np.cos(car_state[2])
    J_pA[1, 2] = px * np.cos(car_state[2]) - py * np.sin(car_state[2])
    
    # 梯度公式: ∇_x sd(x) ≈ n_hat^T * J_pA(x)
    # 梯度的维度: 1 x 3
    # 碰撞惩罚项的梯度: ∇_x |d_safe - sd(x)|+ = - ∇_x sd(x)
    
    collision_gradient = - n_hat @ J_pA
    
    return collision_gradient

# --- 遍历所有障碍物生成所有惩罚项的逻辑 (在主 SCO 循环中使用) ---

def linearize_all_collisions(car_trajectory: List[np.ndarray], d_safe: float = 0.1):
    """
    遍历轨迹中所有时间步和所有障碍物，生成碰撞惩罚项的线性近似。
    """
    collision_approximations = []
    
    for t, state_t in enumerate(car_trajectory):
        for obs in OBSTACLES:
            gradient = get_collision_penalty_gradient(state_t, obs, d_safe)
            
            # 如果梯度非零 (即发生碰撞或在 d_safe 范围内)
            if np.linalg.norm(gradient) > 1e-6:
                # sd 在 d_safe 处的线性化近似为: sd(x) ≈ sd(x_t) + ∇sd * (x - x_t)
                sd, _, _, _ = compute_signed_distance(state_t, obs)
                
                # 我们需要近似的是惩罚项 |d_safe - sd(x)|+
                
                # 碰撞线性项: ∇_x |d_safe - sd(x)|+ * (x - x_t) + |d_safe - sd(x_t)|+
                linear_term = {
                    'time_step': t,
                    'gradient': gradient,
                    'initial_value': np.max([0, d_safe - sd])
                }
                collision_approximations.append(linear_term)
                
    return collision_approximations

if __name__ == "__main__":
    # 简单测试
    car_state = np.array([0.0, 0.0, 0.0])  # 车子在原点，朝向x轴
    obs = OBSTACLES[1]  # 选择第一个障碍物
    
    sd, n_hat, pA_local, pB_world = compute_signed_distance(car_state, obs)
    print("Signed Distance:", sd)
    print("Contact Normal:", n_hat)
    print("Car Local Point:", pA_local)
    print("Obstacle World Point:", pB_world)
    
    gradient = get_collision_penalty_gradient(car_state, obs)
    print("Collision Penalty Gradient:", gradient)