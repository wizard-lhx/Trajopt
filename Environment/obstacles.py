# 04_Environment/obstacles.py

import numpy as np

# --- 障碍物数据结构 ---

# 简单起见，我们定义障碍物为矩形（Rectangular Obstacles）
# 定义格式: [中心x, 中心y, 宽度w, 长度l, 旋转角度theta_rad]
# 注意：宽度 w 沿局部 Y 轴，长度 l 沿局部 X 轴
ObstacleData = np.ndarray  # shape: (5,)

# --- 预设障碍物列表 ---
OBSTACLES: np.ndarray = np.array([
    # 障碍物 1: 位于 (5, 0) 的小墙
    [5.0, 0.0, 0.5, 2.0, 0.0], 
    
    # 障碍物 2: 位于 (3, 3) 的倾斜墙
    [3.0, 3.0, 1.0, 3.0, np.pi / 6],
    
    # 障碍物 3: 位于 (-1, 5) 的竖直墙
    [-1.0, 5.0, 3.0, 0.5, np.pi / 2],
])  # shape: (3, 5) - 3个障碍物，每个5个参数

# --- 碰撞距离辅助函数（为 TrajOpt 线性化做准备） ---

def distance_to_obstacle(
    car_state: np.ndarray, 
    obs: ObstacleData
) -> float:
    """
    【简化的距离函数 - 仅用于演示】
    
    在 TrajOpt 中，我们需要的是两个凸形状（小车和障碍物）之间的带符号距离 (Signed Distance)。
    
    由于实现精确的凸-凸 GJK/EPA 算法过于复杂，我们这里仅返回一个占位值，
    真正的实现依赖于专门的碰撞库（如 Bullet），并在 03_Costs_Constraints 中完成线性化。
    
    这里我们返回一个示例距离，假定为中心点距离与形状大小的函数。
    """
    car_x, car_y, _ = car_state
    obs_x, obs_y, obs_w, obs_l, _ = obs
    
    center_dist = np.sqrt((car_x - obs_x)**2 + (car_y - obs_y)**2)
    
    # 假设一个简化的安全距离界限
    safe_margin = (obs_w + obs_l) / 4 
    
    # 返回一个粗略的带符号距离：正值表示安全，负值表示碰撞
    return center_dist - safe_margin


# --- 测试可视化函数（可选，用于验证障碍物绘制）---
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-5, 8)
    ax.set_ylim(-5, 8)
    
    # 测试绘制障碍物
    from utils.visualize import visualize_obstacle
    for i, obs in enumerate(OBSTACLES):
        obstacles_plot = visualize_obstacle(obs, ax, label=f'Obs {i+1}', obstacle_id=i+1)

    plt.title('2D Obstacle Map')
    plt.grid(True)
    plt.legend()
    plt.show()