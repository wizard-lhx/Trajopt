# 01_Kinematics/car_model.py

import numpy as np
from typing import Tuple
from Kinematics.state_definitions import TIME_STEP, CAR_WIDTH, CAR_LENGTH, CAR_HEIGHT

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
    """
    使用 PyBullet 测试和可视化运动学模型
    """
    import pybullet as p
    import pybullet_data
    import time
    import math
    
    # 1. 初始化 PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 2. 加载地面
    plane_id = p.loadURDF("plane.urdf")
    
    # 3. 创建车子模型 (长方体)
    # 使用与 Kinematics 定义相同的尺寸    
    col_shape_id = p.createCollisionShape(
        p.GEOM_BOX, 
        halfExtents=[CAR_LENGTH/2, CAR_WIDTH/2, CAR_HEIGHT/2]
    )
    vis_shape_id = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[CAR_LENGTH/2, CAR_WIDTH/2, CAR_HEIGHT/2], 
        rgbaColor=[0, 0, 1, 1]
    )
    
    car_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=col_shape_id,
        baseVisualShapeIndex=vis_shape_id,
        basePosition=[0, 0, CAR_HEIGHT/2]
    )
    
    # 设置摩擦系数
    p.changeDynamics(car_id, -1, lateralFriction=0.5)
    
    # 4. 加载障碍物
    from Environment.obstacles import OBSTACLES
    obstacle_ids = []
    for obs in OBSTACLES:
        cx, cy, w, l, theta = obs
        obs_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[l/2, w/2, 0.5]
        )
        obs_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[l/2, w/2, 0.5],
            rgbaColor=[1, 0, 0, 0.5]
        )
        
        # 将2D旋转角转换为四元数
        orn = p.getQuaternionFromEuler([0, 0, theta])
        obs_id = p.createMultiBody(
            baseMass=0,  # 静态障碍物
            baseCollisionShapeIndex=obs_col,
            baseVisualShapeIndex=obs_vis,
            basePosition=[cx, cy, 0.5],
            baseOrientation=orn
        )
        obstacle_ids.append(obs_id)
    
    print("=== PyBullet 运动学模型测试 ===")
    print("使用键盘控制:")
    print("  UP/DOWN: 线速度 vx")
    print("  LEFT/RIGHT: 角速度 omega")
    print("  ESC: 退出")
    print()
    
    # 启用碰撞检测和物理模拟
    p.setRealTimeSimulation(0)  # 手动步进
    
    dt = TIME_STEP
    
    # 主循环
    step_count = 0
    while p.isConnected():
        # 获取键盘输入
        keys = p.getKeyboardEvents()
        
        vx = 0.0
        omega = 0.0
        
        # 键盘控制
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            vx = 1.0
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            vx = -0.5
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            omega = 1.0
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            omega = -1.0
        
        # 从 PyBullet 获取当前状态（物理引擎计算的真实状态）
        pos, orn = p.getBasePositionAndOrientation(car_id)
        euler = p.getEulerFromQuaternion(orn)
        current_theta = euler[2]
        
        # 计算全局坐标系下的速度
        vx_global = vx * np.cos(current_theta)
        vy_global = vx * np.sin(current_theta)
        
        # 使用速度控制而不是位置控制，让物理引擎处理碰撞
        # 强制 z=0 平面运动和只绕 z 轴旋转
        p.resetBaseVelocity(car_id, [vx_global, vy_global, 0], [0, 0, omega])
        
        # 同时约束车子高度，防止因碰撞被顶起
        # p.resetBasePositionAndOrientation(
        #     car_id,
        #     [pos[0], pos[1], CAR_HEIGHT/2],  # 保持 z 高度恒定
        #     p.getQuaternionFromEuler([0, 0, current_theta])  # 保持 roll=0, pitch=0
        # )
        
        # 步进仿真
        p.stepSimulation()
        
        # 相机跟随
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=math.degrees(current_theta) - 90,
            cameraPitch=-45,
            cameraTargetPosition=[pos[0], pos[1], 0]
        )
        
        # 每秒打印一次状态
        step_count += 1
        if step_count % int(1.0/dt) == 0:
            print(f"State: x={pos[0]:.2f}, y={pos[1]:.2f}, θ={np.degrees(current_theta):.1f}°")
        
        time.sleep(dt)
    
    p.disconnect()