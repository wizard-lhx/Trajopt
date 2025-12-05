# 03_Costs_Constraints/kinematics_constraint.py

import numpy as np
from typing import List, Tuple, Dict
# 导入模型和维度定义
from Kinematics.car_model import update_state, get_kinematics_jacobian
from Kinematics.state_definitions import TIME_STEP, STATE_DIM, CONTROL_DIM

# --- 核心函数：运动学约束的线性化 ---

def linearize_kinematics_constraint(
    x_t: np.ndarray, 
    u_t: np.ndarray, 
    x_t1: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    对运动学等式约束 h(x_t, u_t, x_{t+1}) = x_{t+1} - f(x_t, u_t) = 0 
    进行线性化，并计算当前残差 (违反程度)。
    
    线性化形式: h(x) ≈ h(x_0) + ∇h(x_0)^T (x - x_0) = 0
    其中 x 是所有变量 [x_t, u_t, x_{t+1}] 的堆叠。

    参数:
    x_t: t 时刻的状态 (np.ndarray of shape (STATE_DIM,))
    u_t: t 时刻的控制 (np.ndarray of shape (CONTROL_DIM,))
    x_t1: t+1 时刻的状态 (np.ndarray of shape (STATE_DIM,))

    返回:
    Tuple[A_lin, b_lin]: 线性化约束 A_lin * [x_t, u_t, x_{t+1}]^T + b_lin = 0
                       A_lin 是 3 x (STATE_DIM + CONTROL_DIM + STATE_DIM) 矩阵。
                       b_lin 是 3 x 1 向量。
    """
    
    # 1. 计算当前残差 (h(x_0))
    # 残差是当前轨迹点对运动学模型的违反程度
    x_t_pred = update_state(x_t, u_t, dt=TIME_STEP)
    residual = x_t1 - x_t_pred  # 形状 (STATE_DIM,)
    
    # 2. 计算雅可比矩阵 (∇f(x_t, u_t))
    # J_x = ∂f / ∂x_t, J_u = ∂f / ∂u_t
    J_x, J_u = get_kinematics_jacobian(x_t, u_t, dt=TIME_STEP)
    
    # 3. 计算约束的雅可比 (∇h(x))
    # h = x_{t+1} - f(x_t, u_t)
    # 变量顺序: [x_t, u_t, x_{t+1}]
    
    # J_xt = ∂h / ∂x_t = -∂f / ∂x_t = -J_x
    # J_ut = ∂h / ∂u_t = -∂f / ∂u_t = -J_u
    # J_xt1 = ∂h / ∂x_{t+1} = I (单位矩阵)
    
    # A_lin 矩阵 (形状: STATE_DIM x (STATE_DIM + CONTROL_DIM + STATE_DIM))
    A_lin = np.hstack([-J_x, -J_u, np.eye(STATE_DIM)])
    
    # 4. 计算线性化截距 (b_lin)
    # 线性化公式: h(x) ≈ h(x_0) + ∇h(x_0) (x - x_0) = 0
    # A_lin * x + b_lin = 0
    # b_lin = h(x_0) - ∇h(x_0) x_0 
    #       = residual - A_lin * [x_t, u_t, x_{t+1}]^T
    
    x_vec = np.hstack([x_t, u_t, x_t1]) # 当前变量的堆叠 (x_0)
    b_lin = residual - A_lin @ x_vec
    
    # 我们返回 A_lin 和 residual，因为在 TrajOpt 中，h(x)=0 约束通常转化为
    # 惩罚项 ||h(x)||^2 或 L1 惩罚 |h(x)|，它们都需要残差信息。
    # 但由于用户要求实现线性化，我们返回 A_lin 和 b_lin (b_lin 是残差的调整项)。
    return A_lin, b_lin, residual

def linearize_all_kinematics(
    trajectory_x: np.ndarray, 
    trajectory_u: np.ndarray
) -> List[Dict]:
    """
    遍历整个轨迹，对所有时间步的运动学约束进行线性化。

    参数:
    trajectory_x: 形状为 (T, STATE_DIM) 的轨迹状态序列。
    trajectory_u: 形状为 (T-1, CONTROL_DIM) 的控制输入序列。

    返回:
    List[Dict]: 包含每个时间间隔线性化结果的列表。
    """
    T = trajectory_x.shape[0]
    linearized_constraints = []
    
    for t in range(T - 1):
        x_t = trajectory_x[t]
        u_t = trajectory_u[t]
        x_t1 = trajectory_x[t+1]
        
        A_lin, b_lin, residual = linearize_kinematics_constraint(x_t, u_t, x_t1)
        
        linearized_constraints.append({
            'time_step': t,
            'A_lin': A_lin,        # 线性化矩阵 A
            'b_lin': b_lin,        # 线性化截距 b
            'residual': residual   # 当前残差 h(x_0)
        })
        
    return linearized_constraints


# --- 示例运行块 ---
if __name__ == '__main__':
    # 假设 T=3, STATE_DIM=3, CONTROL_DIM=2
    # 1. 定义一个初始轨迹 (例如，一个碰撞轨迹)
    x_init = np.array([
        [0.0, 0.0, 0.0],   # x0
        [1.0, 0.1, 0.0],   # x1 (违反运动学)
        [2.0, 0.0, 0.0]    # x2 (违反运动学)
    ])
    u_init = np.array([
        [1.0, 0.1],      # u0
        [1.0, -0.1]      # u1
    ])
    
    print(f"--- 轨迹线性化测试 (T={x_init.shape[0]}): ---")
    
    # 2. 对第一个时间步进行线性化 (t=0)
    t = 0
    x_t_true = x_init[t]
    u_t_true = u_init[t]
    x_t1_true = x_init[t+1]
    
    # 预测的下一个状态
    x_t1_pred = update_state(x_t_true, u_t_true)
    
    A, b, residual = linearize_kinematics_constraint(x_t_true, u_t_true, x_t1_true)
    
    print(f"\n时间步 t={t} 的残差 (h(x_0)): {residual}")
    print(f"  预测 x_{t+1}: {x_t1_pred}")
    print(f"  实际 x_{t+1}: {x_t1_true}")
    
    # 验证线性化矩阵 A 的形状
    print(f"\n线性化矩阵 A 的形状: {A.shape} (应为 {STATE_DIM} x {STATE_DIM + CONTROL_DIM + STATE_DIM})")
    
    # 验证线性化近似
    # 假设我们进行一个微小调整 d_x (例如，只调整 x1)
    d_x = np.zeros(STATE_DIM * 2 + CONTROL_DIM)
    
    # 假设只调整 x1 的 y 坐标 (索引 4)
    d_x[STATE_DIM + CONTROL_DIM + 1] = 0.001 
    
    # 线性化预测的残差变化 (A * dx + h(x0))
    linear_change = A @ d_x + residual 
    print(f"\n线性化预测的残差变化 (A*dx + h(x0)): {linear_change}")
    
    # 3. 遍历整个轨迹
    all_constraints = linearize_all_kinematics(x_init, u_init)
    print(f"\n已线性化 {len(all_constraints)} 个时间间隔的约束。")
    print(f"时间步 t=1 的线性化矩阵 A:\n {all_constraints[1]['A_lin']}")