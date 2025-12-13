# 03_Costs_Constraints/no_collision_cost.py

import numpy as np
from typing import Tuple, List, Optional
from Environment.obstacles import OBSTACLES
from utils.bullet_collision import BulletCollisionChecker

# 全局碰撞检测器（避免重复初始化）
_global_collision_checker: Optional[BulletCollisionChecker] = None

def get_collision_checker() -> BulletCollisionChecker:
    """
    获取全局碰撞检测器实例（单例模式）
    """
    global _global_collision_checker
    if _global_collision_checker is None:
        _global_collision_checker = BulletCollisionChecker(use_gui=False)
        # 创建车辆碰撞体
        _global_collision_checker.create_car_body()
        # 创建所有障碍物碰撞体
        for i, obs in enumerate(OBSTACLES):
            cx, cy, w, l, theta = obs
            _global_collision_checker.create_obstacle(i, cx, cy, w, l, theta)
    return _global_collision_checker

# --- 核心碰撞检测和线性化函数 ---

def compute_signed_distance(
    car_state: np.ndarray, 
    obs_id: int
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    【TrajOpt 核心函数：使用 PyBullet 进行碰撞检测】

    该函数必须返回:
    1. sd: 带符号距离 (Signed Distance)
    2. n_hat: 接触法线 (Contact Normal) - 必须是单位向量
    3. pA_local: 小车上最近点在小车局部坐标系下的位置
    4. pB_world: 障碍物上最近点在世界坐标系下的位置
    """
    checker = get_collision_checker()
    sd, n_hat, pA_local, pB_world = checker.compute_signed_distance(
        car_state, obs_id, max_distance=1.0
    )
    return sd, n_hat, pA_local, pB_world


def get_collision_penalty_gradient(
    car_state: np.ndarray, 
    obs_id: int,
    d_safe: float = 0.1 
) -> np.ndarray:
    """
    计算碰撞惩罚项 |d_safe - sd(x)|+ 对状态 x 的梯度 (线性化)。
    这是 TrajOpt 构造 QP 子问题中碰撞约束项的关键。
    """
    
    sd, n_hat, pA_local, pB_world = compute_signed_distance(car_state, obs_id)
    
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
    # 碰撞惩罚项的梯度: ∇_x (d_safe - sd(x)) = - ∇_x sd(x)
    
    collision_gradient = - n_hat @ J_pA
    
    return collision_gradient

# --- 遍历所有障碍物生成所有惩罚项的逻辑 (在主 SCO 循环中使用) ---

def linearize_all_collisions(car_trajectory: List[np.ndarray], d_safe: float = 0.1):
    """
    遍历轨迹中所有时间步和所有障碍物，生成碰撞惩罚项的线性近似。
    """
    collision_approximations = []
    
    for t, state_t in enumerate(car_trajectory):
        for i, obs in enumerate(OBSTACLES):
            gradient = get_collision_penalty_gradient(state_t, i, d_safe)
            
            # 如果梯度非零 (即发生碰撞或在 d_safe 范围内)
            if np.linalg.norm(gradient) > 1e-6:
                # sd 在 d_safe 处的线性化近似为: sd(x) ≈ sd(x_t) + ∇sd * (x - x_t)
                sd, _, _, _ = compute_signed_distance(state_t, i)
                
                # 我们需要近似的是惩罚项 d_safe - sd(x)
                
                # 碰撞线性项: ∇_x (d_safe - sd(x)) * (x - x_t) + (d_safe - sd(x_t))
                linear_term = {
                    'time_step': t,
                    'obstacle_id': i,
                    'gradient': gradient,
                    'initial_value': d_safe - sd,
                    'type': 'discrete'  # 标记为离散碰撞
                }
                collision_approximations.append(linear_term)
                
    return collision_approximations


def linearize_continuous_collisions(
    car_trajectory: List[np.ndarray], 
    d_safe: float = 0.1
):
    """
    连续时间碰撞检测：检测相邻时间步之间swept volume凸包与障碍物的碰撞
    
    根据TrajOpt论文公式(23)，对每对相邻时间步(t, t+1)：
    sd_AB(θ^t, θ^{t+1}) ≈ sd_AB(θ_0^t, θ_0^{t+1})
                          + α * n̂^T * J_{p0}(θ_0^t) * (θ^t - θ_0^t)
                          + (1-α) * n̂^T * J_{p1}(θ_0^{t+1}) * (θ^{t+1} - θ_0^{t+1})
    
    参数:
        car_trajectory: 轨迹 (T, STATE_DIM)
        d_safe: 安全距离
    
    返回:
        collision_approximations: 线性化的连续碰撞约束列表
    """
    collision_approximations = []
    T = len(car_trajectory)
    checker = get_collision_checker()
    
    for t in range(T - 1):
        state_t = car_trajectory[t]
        state_t1 = car_trajectory[t + 1]
        
        for i, obs in enumerate(OBSTACLES):
            # 使用PyBullet创建凸包并计算与障碍物的最近距离
            sd_swept, n_hat, pA_local_t0, pB_world, alpha = checker.compute_swept_volume_distance(
                state_t, state_t1, i, max_distance=d_safe + 0.2
            )
            
            # 只在swept volume接近碰撞时添加约束
            if sd_swept < d_safe + 0.1:
                # 计算两个端点状态的梯度
                grad_t = get_collision_penalty_gradient(state_t, i, d_safe)
                grad_t1 = get_collision_penalty_gradient(state_t1, i, d_safe)
                
                # 根据alpha加权组合梯度
                # 论文公式(23): (1-α)·∇sd_t + α·∇sd_{t+1}
                gradient_combined = (1 - alpha) * grad_t + alpha * grad_t1
                
                if np.linalg.norm(gradient_combined) > 1e-6:
                    linear_term = {
                        'time_step': t,              # 起始时间步
                        'time_step_end': t + 1,      # 结束时间步
                        'obstacle_id': i,
                        'gradient': gradient_combined,
                        'gradient_t': grad_t,        # t时刻的梯度
                        'gradient_t1': grad_t1,      # t+1时刻的梯度
                        'alpha': alpha,              # 接触点的估计插值参数
                        'initial_value': d_safe - sd_swept,  # 使用swept凸包的距离
                        'type': 'continuous'         # 标记为连续碰撞
                    }
                    collision_approximations.append(linear_term)
    
    return collision_approximations

if __name__ == "__main__":
    """
    测试 PyBullet 碰撞检测集成
    """
    print("=" * 70)
    print("测试 PyBullet 碰撞检测集成到 no_collision_cost")
    print("=" * 70)
    
    # 测试场景1：车辆靠近障碍物
    obs_id = 0  # 使用第1个障碍物
    obs = OBSTACLES[obs_id]

    car_state1 = np.array([obs[0] - 0.3, obs[1], 0.0])
    print(f"\n测试1: 车辆状态 {car_state1}")
    
    sd1, n_hat1, pA_local1, pB_world1 = compute_signed_distance(car_state1, obs_id)
    print(f"  Signed Distance: {sd1:.4f}")
    print(f"  Contact Normal: {n_hat1}")
    
    gradient1 = get_collision_penalty_gradient(car_state1, obs_id, d_safe=0.5)
    print(f"  Collision Gradient: {gradient1}")
    
    # 测试线性化所有碰撞
    print("\n测试2: 轨迹碰撞线性化")
    trajectory = [
        np.array([obs[0] - 0.3, 0.0, 0.0]),
        np.array([obs[0], 0.0, 0.0]),
        np.array([obs[0] + 0.3, 0.0, 0.0])
    ]
    
    collision_terms = linearize_all_collisions(trajectory, d_safe=0.5)
    print(f"  总碰撞约束数: {len(collision_terms)}")
    for term in collision_terms:
        print(f"    时间步{term['time_step']}, 障碍物{term['obstacle_id']}: " 
              f"梯度范数={np.linalg.norm(term['gradient']):.4f}, "
              f"初始值={term['initial_value']:.4f}")
    
    # 清理
    checker = get_collision_checker()
    checker.cleanup()
    
    print("\n" + "=" * 70)