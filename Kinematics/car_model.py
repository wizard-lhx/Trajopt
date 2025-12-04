# 01_Kinematics/car_model.py

import numpy as np
from typing import Tuple
from Kinematics.state_definitions import TIME_STEP

def update_state(
    current_state: np.ndarray, 
    control_input: np.ndarray, 
    dt: float = TIME_STEP
) -> np.ndarray:
    """
    使用 Unicycle Model (非完整约束模型) 更新车子的状态。

    状态: [x, y, theta]
    控制: [vx, omega]
    
    微分方程:
    x_dot = vx * cos(theta)
    y_dot = vx * sin(theta)
    theta_dot = omega
    """
    x, y, theta = current_state
    vx, omega = control_input
    
    # --- 运动学模型 (欧拉积分) ---
    
    # 1. 计算状态导数 (x_dot, y_dot, theta_dot)
    x_dot = vx * np.cos(theta)
    y_dot = vx * np.sin(theta)
    theta_dot = omega
    
    # 2. 计算增量
    dx = x_dot * dt
    dy = y_dot * dt
    dtheta = theta_dot * dt
    
    # 3. 更新状态
    new_state = current_state + np.array([dx, dy, dtheta])
    
    return new_state

# --- Jacobian (用于 TrajOpt 的线性化) ---

def get_kinematics_jacobian(
    current_state: np.ndarray, 
    control_input: np.ndarray, 
    dt: float = TIME_STEP
) -> np.ndarray:
    """
    计算状态 x_{t+1} 对当前状态 x_t 和控制量 u_t 的解析雅可比矩阵。
    
    F(x, u) = x + dt * [vx*cos(theta), vx*sin(theta), omega]
    
    J_x = dF / dx
    J_u = dF / du
    
    在 TrajOpt 中，我们主要关注 x_{t+1} 对 x_t 的雅可比 J_x (用于线性化 x_{t+1} - f(x_t, u_t) = 0)。
    """
    x, y, theta = current_state
    vx, omega = control_input
    
    # 状态转移矩阵 J_x = d(x_{t+1}) / d(x_t)
    J_x = np.eye(3)
    
    # d(x_{t+1}) / d(theta_t) = -vx * sin(theta) * dt
    J_x[0, 2] = -vx * np.sin(theta) * dt
    
    # d(y_{t+1}) / d(theta_t) = vx * cos(theta) * dt
    J_x[1, 2] = vx * np.cos(theta) * dt
    
    # d(theta_{t+1}) / d(theta_t) = 1 (已包含在 np.eye(3) 中)
    
    # 控制转移矩阵 J_u = d(x_{t+1}) / d(u_t)
    J_u = np.zeros((3, 2))
    
    # d(x_{t+1}) / d(vx) = cos(theta) * dt
    J_u[0, 0] = np.cos(theta) * dt
    
    # d(y_{t+1}) / d(vx) = sin(theta) * dt
    J_u[1, 0] = np.sin(theta) * dt
    
    # d(theta_{t+1}) / d(omega) = dt
    J_u[2, 1] = dt
    
    return J_x, J_u

if __name__ == '__main__':
    init_state = np.array([0.0, 0.0, 0.0])
    control = np.array([1.0, np.pi * 2])

    # 模型测试
    print(update_state(init_state, control))

    # 可视化测试
    import utils.visualize
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_xlim(-5, 10)
    ax.set_ylim(-5, 10)
    plt.ion()
    fig.show()
    utils.visualize.visualize_car(update_state(init_state, control), ax)
    plt.ioff()
    plt.show()