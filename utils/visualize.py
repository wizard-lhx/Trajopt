# 01_Kinematics/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches_module
from typing import List, Any
from Kinematics.state_definitions import CAR_LENGTH
from Environment.obstacles import OBSTACLES
from utils.commom import get_car_corners, get_obstacle_corners
from Cost_Constraints.no_collision_cost import compute_signed_distance_and_gradient

# --- 可视化核心函数 (与上一步一致) ---
def visualize_car(
    state: np.ndarray,
    ax: plt.Axes,
    color: str = 'teal',
    label: str = 'Car',
) -> List[Any]:
    """
    绘制车子模型在世界坐标系下的姿态。
    返回绘制出的 Matplotlib 对象列表。
    """
    x, y, theta = state
    
    # 转换到世界坐标系
    world_car_corners = get_car_corners(state)

    # 绘制元素列表
    patches = []

    # 绘制车子主体 (Polygon)
    car_polygon = Polygon(world_car_corners, closed=True, color=color, alpha=0.6, label=label)
    ax.add_patch(car_polygon)
    patches.append(car_polygon)
    
    # 绘制车子中心点
    center_plot, = ax.plot(x, y, 'o', color='black', markersize=3)
    patches.append(center_plot)
    
    # 绘制车子方向 (Arrow)
    half_L = CAR_LENGTH / 2
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    heading_vector_local = np.array([half_L * 1.5, 0])
    heading_vector_world = heading_vector_local @ rotation_matrix.T
    arrow_patch = ax.arrow(
        x, y, 
        heading_vector_world[0], heading_vector_world[1],
        head_width=0.15, head_length=0.2, fc='k', ec='k', 
        linewidth=1.5, zorder=5
    )
    patches.append(arrow_patch)

    return patches

def visualize_obstacle(
    obs: np.ndarray,
    ax: plt.Axes,
    color: str = 'red',
    label: str = 'Obstacle',
    show_id: bool = True,
    obstacle_id: int = None,
) -> List[Any]:
    """
    绘制障碍物模型在世界坐标系下的位置和姿态。
    
    参数:
        obs: 障碍物数据 [cx, cy, w, l, theta]
        ax: Matplotlib 坐标轴
        color: 障碍物颜色
        label: 图例标签
        show_id: 是否显示障碍物编号
        obstacle_id: 障碍物编号（如果为None则不显示）
    
    返回绘制出的 Matplotlib 对象列表。
    """
    corners = get_obstacle_corners(obs)
    patches = []

    obs_polygon = Polygon(corners, closed=True, color=color, alpha=0.3, label=label)
    ax.add_patch(obs_polygon)
    patches.append(obs_polygon)

    # 绘制中心点标记
    center_plot, = ax.plot(obs[0], obs[1], 'x', color='darkred', markersize=5)
    patches.append(center_plot)

    # 添加编号文本
    if show_id and obstacle_id is not None:
        text = ax.text(
            obs[0], obs[1],  # 位置：障碍物中心
            f'{obstacle_id}',  # 文本内容
            fontsize=10,
            fontweight='bold',
            color='white',
            ha='center',  # 水平居中
            va='center',  # 垂直居中
            bbox=dict(boxstyle='circle,pad=0.2', facecolor=color, edgecolor='darkred', linewidth=1)
        )
        patches.append(text)

    return patches

def remove_patches(patches: List[Any]):
    """清除Matplotlib绘制的元素"""
    for item in patches:
        if isinstance(item, Polygon) or isinstance(item, patches_module.Arrow):
            item.remove()
        elif isinstance(item, plt.Line2D):
            item.remove()
        elif isinstance(item, plt.Text):
            item.remove()


def dynamic_visualization_test(initial_state, controls, obstacles):
    """
    动态演示小车运动轨迹 (作为测试入口)
    """
    from Kinematics.car_model import update_state
    from Kinematics.state_definitions import TIME_STEP
    
    current_state = np.array(initial_state)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_xlim(-5, 10)
    ax.set_ylim(-5, 10)
    plt.ion()
    fig.show()
    
    all_patches = []
    
    # 绘制障碍物（带编号）
    for i, obs in enumerate(obstacles):
        obs_patches = visualize_obstacle(obs, ax, label=f'Obs {i+1}', obstacle_id=i+1)
    ax.legend()
    for vx, omega, duration in controls:
        num_steps = int(duration / TIME_STEP)
        control_input = np.array([vx, omega])
        
        for i in range(num_steps):
            
            # 1. 移除上一帧
            remove_patches(all_patches)
            all_patches = []

            # 2. 更新状态
            current_state = update_state(current_state, control_input)
            x_new, y_new, _ = current_state

            # # 打印碰撞距离信息
            # obs = OBSTACLES[1]  # 选择第一个障碍物
            # sd, n_hat, pA_local, pB_world = compute_signed_distance_and_gradient(current_state, obs)
            # print("Signed Distance:", sd)
            
            # 3. 绘制新状态和轨迹
            all_patches = visualize_car(current_state, ax)
            ax.plot(x_new, y_new, '.', color='gray', markersize=2, alpha=0.5)

            # 4. 刷新
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            
    plt.ioff()
    plt.show()

# --- 示例运行块 (作为测试入口) ---
if __name__ == '__main__':
    # 初始状态：(x, y, theta) = (0.0, 0.0, 0.0 rad)
    initial_s = (0.0, 0.0, 0.0)
    
    # 控制序列：(vx, omega, duration)
    control_sequence = [
        (1.0, 0.0, 2.0),
        (0.0, np.pi / 2, 1.0), 
        (1.5, 0.0, 3.0),
        (1.0, -0.5, 3.0),
        (0.0, 0.0, 1.0)
    ]
    
    
    dynamic_visualization_test(initial_s, control_sequence, OBSTACLES)