# 02_Core_TrajOpt/penalty_utilities.py

import numpy as np
from typing import Tuple, List, Optional

# --- 1. 转化 L1 铰链损失 (|g(x)|⁺) ---

def convert_hinge_loss_to_qp(
    a_vec: np.ndarray, 
    b_scalar: float, 
    qp_cost_weight: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    将铰链损失惩罚项 μ * |a·Δx + b|⁺ 转化为 QP 约束和目标项。
    
    原始惩罚项:  最小化 μ * max(0, a·Δx + b)
    转换后:      最小化 μ * t
                 约束: [a·Δx - t ≤ -b] 且 [-t ≤ 0]
    
    参数:
    a_vec (ndarray): 线性化后的不等式约束梯度 ∇g(x) (形状 M,)。
    b_scalar (float): 线性化后的截距 (g(x₀) + ∇g(x₀)x₀ 的剩余部分)。
    qp_cost_weight (float): 惩罚系数 μ。
    
    返回:
    (A_ineq, b_ineq, c_penalty, H_penalty, num_slack): 
        A_ineq, b_ineq: 新增的线性不等式约束。
        c_penalty, H_penalty: 惩罚项产生的 QP 目标项 (线性项和二次项)。
        num_slack: 新增的松弛变量数量 (1)。
    """
    M = a_vec.shape[0] # 原始变量数量
    num_slack = 1      # 松弛变量 t (t >= 0)
    
    # 1. QP 目标项的调整 (最小化 μ * t)
    # 目标函数：... + c_penalty^T * [Δx, t]^T
    c_penalty = np.zeros(M + num_slack)
    c_penalty[-1] = qp_cost_weight # 将 μ 赋予松弛变量 t 的线性项
    H_penalty = np.zeros((M + num_slack, M + num_slack)) # H_penalty 为零矩阵

    # 2. 线性不等式约束 A_ineq @ [Δx, t]^T <= b_ineq
    # 约束总数 = 2
    A_ineq = np.zeros((2, M + num_slack))
    b_ineq = np.zeros(2)
    
    # (a) 约束: a·Δx + b ≤ t   =>  a·Δx - t ≤ -b
    A_ineq[0, :M] = a_vec
    A_ineq[0, M] = -1.0 
    b_ineq[0] = -b_scalar
    
    # (b) 约束: 0 ≤ t         => -t ≤ 0
    A_ineq[1, M] = -1.0
    b_ineq[1] = 0.0

    return A_ineq, b_ineq, c_penalty, H_penalty, num_slack

# --- 2. 转化 L1 绝对值 (|h(x)|) ---

def convert_absolute_value_to_qp(
    a_vec: np.ndarray, 
    b_scalar: float, 
    qp_cost_weight: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """
    将绝对值惩罚项 μ * |a·Δx + b| 转化为 QP 约束和目标项。
    
    原始惩罚项:  最小化 μ * |a·Δx + b|
    转换后:      最小化 μ * (s + t)
                 约束: [s - t = a·Δx + b] 且 [0 ≤ s, 0 ≤ t]
    
    参数:
    ... (同上)
    
    返回:
    (A_ineq, b_ineq, c_penalty, H_penalty, num_slack, A_eq, b_eq): 
        包含不等式约束、等式约束、目标项和松弛变量数量。
    """
    M = a_vec.shape[0]
    num_slack = 2 # 两个松弛变量 s, t
    
    # 1. QP 目标项的调整 (最小化 μ * (s + t))
    # 目标函数：... + c_penalty^T * [Δx, s, t]^T
    c_penalty = np.zeros(M + num_slack)
    c_penalty[-2] = qp_cost_weight # s 的线性项系数 μ
    c_penalty[-1] = qp_cost_weight # t 的线性项系数 μ
    H_penalty = np.zeros((M + num_slack, M + num_slack))

    # 2. 线性等式约束 A_eq @ [Δx, s, t]^T = b_eq
    # 约束: s - t = a·Δx + b  =>  -a·Δx + s - t = b_scalar
    
    A_eq = np.zeros((1, M + num_slack))
    b_eq = np.zeros(1)
    
    # 约束 (a): [-a_vec | 1 | -1] @ [Δx, s, t]^T = b_scalar
    A_eq[0, :M] = -a_vec 
    A_eq[0, M] = 1.0     # s
    A_eq[0, M+1] = -1.0  # t
    b_eq[0] = b_scalar
    
    # 3. 线性不等式约束 A_ineq @ [Δx, s, t]^T <= b_ineq (非负约束)
    # 约束总数 = 2
    A_ineq = np.zeros((2, M + num_slack))
    b_ineq = np.zeros(2)
    
    # (b) 约束: 0 ≤ s => -s ≤ 0
    A_ineq[0, M] = -1.0
    
    # (c) 约束: 0 ≤ t => -t ≤ 0
    A_ineq[1, M+1] = -1.0

    return A_ineq, b_ineq, c_penalty, H_penalty, num_slack, A_eq, b_eq


# --- 示例运行块 ---
if __name__ == '__main__':
    # 假设有 3 个原始优化变量 M=3 (例如 x, y, theta)
    M_test = 3
    mu_test = 5.0
    
    # 碰撞约束的线性近似 (梯度和截距)
    a_hinge = np.array([0.5, -0.2, 0.0])
    b_hinge = -0.1 # 负截距意味着当前在碰撞边界附近 (d_safe - sd(x0) < 0)
    
    # ----------------------------------------------------
    # 1. 测试 L1 铰链损失 (|g(x)|⁺)
    A_ineq, b_ineq, c_pen, H_pen, n_slack = convert_hinge_loss_to_qp(a_hinge, b_hinge, mu_test)
    
    print("--- L1 Hinge Loss (|g(x)|⁺) ---")
    print(f"原始变量 M={M_test}, 新增松弛变量: {n_slack}")
    print(f"QP 线性目标项 c (最后一位是 t): {c_pen}") 
    print(f"不等式约束 A_ineq:\n {A_ineq}")
    print(f"不等式约束 b_ineq: {b_ineq}")
    # ----------------------------------------------------

    # 绝对值约束的线性近似 (例如运动学残差)
    a_abs = np.array([1.0, 0.0, -1.0])
    b_abs = 0.05 # 当前的运动学残差 (h(x0) != 0)
    
    # ----------------------------------------------------
    # 2. 测试 L1 绝对值 (|h(x)|)
    (A_ineq_abs, b_ineq_abs, c_pen_abs, H_pen_abs, n_slack_abs, A_eq_abs, b_eq_abs
     ) = convert_absolute_value_to_qp(a_abs, b_abs, mu_test)
    
    print("\n--- L1 Absolute Value (|h(x)|) ---")
    print(f"原始变量 M={M_test}, 新增松弛变量: {n_slack_abs}")
    print(f"QP 线性目标项 c (最后两位是 s, t): {c_pen_abs}") 
    print(f"等式约束 A_eq:\n {A_eq_abs}") 
    print(f"等式约束 b_eq: {b_eq_abs}")
    # ----------------------------------------------------