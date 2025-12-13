"""
凸化模块：将非凸问题凸化为QP子问题
"""

import numpy as np
from typing import Tuple
from Kinematics.state_definitions import STATE_DIM
from Cost_Constraints.no_collision_cost import linearize_all_collisions, linearize_continuous_collisions
from Cost_Constraints.trajectory_cost import linearize_and_quadraticize_path_cost


def convexify_objective(
    x_curr: np.ndarray,
    T: int,
    N: int,
    num_slack: int,
    mu: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将目标函数（路径长度+松弛变量惩罚）凸化为二次形式
    
    参数:
        x_curr: 当前轨迹 (T, N)
        T: 时间步数
        N: 状态维度
        num_slack: 松弛变量数量
        mu: 惩罚系数
    
    返回:
        H: Hessian矩阵 (M, M)
        c: 梯度向量 (M,)
    """
    M = T * N + num_slack
    
    # a. 路径长度的二次化
    grad_path_state, H_path_state = linearize_and_quadraticize_path_cost(x_curr)
    
    # 扩展到完整优化变量维度
    grad_path = np.zeros(M)
    grad_path[:T*N] = grad_path_state
    
    H_path = np.zeros((M, M))
    H_path[:T*N, :T*N] = H_path_state
    
    # b. 添加松弛变量的线性惩罚项 μ·t
    c = grad_path.copy()
    c[T*N:] += mu  # 所有松弛变量系数为μ
    
    return H_path, c


def convexify_collision_constraints(
    x_curr: np.ndarray,
    d_safe: float,
    T: int,
    N: int,
    use_continuous: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    将碰撞约束凸化为线性不等式约束（铰链损失 → 松弛变量）
    
    约束形式: d_safe - sd(x) ≤ slack
    转换为QP: ∇g·Δx - slack ≤ -g(x₀) 且 slack ≥ 0
    
    参数:
        x_curr: 当前轨迹 (T, N)
        d_safe: 安全距离
        T: 时间步数
        N: 状态维度
        use_continuous: 是否使用连续时间碰撞检测
    
    返回:
        A_ineq: 不等式约束矩阵 (2*n_constraints, M)
        b_ineq: 不等式约束向量 (2*n_constraints,)
        num_slack: 松弛变量数量
    """
    # 获取离散碰撞约束
    col_approximations = linearize_all_collisions(x_curr, d_safe)
    
    # 如果启用连续碰撞检测，添加连续约束
    if use_continuous:
        continuous_approximations = linearize_continuous_collisions(x_curr, d_safe)
        col_approximations.extend(continuous_approximations)
    
    num_col_constraints = len(col_approximations)
    num_slack = num_col_constraints  # 每个约束对应一个松弛变量
    
    M = T * N + num_slack
    
    # 每个碰撞约束生成2个不等式
    A_ineq = np.zeros((2 * num_col_constraints, M))
    b_ineq = np.zeros(2 * num_col_constraints)
    
    for constraint_idx, col_term in enumerate(col_approximations):
        t = col_term['time_step']
        gradient = col_term['gradient']  # ∇g(x)
        offset = col_term['initial_value']  # g(x₀)
        
        # 对于连续碰撞，需要同时约束两个时间步
        if col_term.get('type') == 'continuous':
            t_end = col_term['time_step_end']
            alpha = col_term['alpha']
            grad_t = col_term['gradient_t']
            grad_t1 = col_term['gradient_t1']
            
            # 约束1: α*∇g_t·Δx_t + (1-α)*∇g_{t+1}·Δx_{t+1} - slack ≤ -g(x₀)
            idx_start_t = t * N
            idx_end_t = (t + 1) * N
            idx_start_t1 = t_end * N
            idx_end_t1 = (t_end + 1) * N
            
            A_ineq[2*constraint_idx, idx_start_t:idx_end_t] = alpha * grad_t
            A_ineq[2*constraint_idx, idx_start_t1:idx_end_t1] = (1 - alpha) * grad_t1
            A_ineq[2*constraint_idx, T*N + constraint_idx] = -1.0
            b_ineq[2*constraint_idx] = -offset
        else:
            # 离散碰撞：只约束单个时间步
            idx_start = t * N
            idx_end = (t + 1) * N
            
            A_ineq[2*constraint_idx, idx_start:idx_end] = gradient
            A_ineq[2*constraint_idx, T*N + constraint_idx] = -1.0
            b_ineq[2*constraint_idx] = -offset
        
        # 约束2: -slack ≤ 0 (确保 slack ≥ 0)
        A_ineq[2*constraint_idx + 1, T*N + constraint_idx] = -1.0
        b_ineq[2*constraint_idx + 1] = 0.0
    
    return A_ineq, b_ineq, num_slack


def add_endpoint_constraints(
    T: int,
    N: int,
    num_slack: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    添加起点和终点固定的等式约束
    
    约束形式: Δx[0] = 0, Δx[T-1] = 0
    
    参数:
        T: 时间步数
        N: 状态维度
        num_slack: 松弛变量数量
    
    返回:
        A_eq: 等式约束矩阵 (2*N, M)
        b_eq: 等式约束向量 (2*N,)
    """
    M = T * N + num_slack
    num_endpoint_constraints = 2 * N
    
    A_eq = np.zeros((num_endpoint_constraints, M))
    b_eq = np.zeros(num_endpoint_constraints)
    
    # 起点约束: Δx[0] = 0
    for i in range(N):
        A_eq[i, i] = 1.0
        b_eq[i] = 0.0
    
    # 终点约束: Δx[T-1] = 0
    for i in range(N):
        A_eq[N + i, (T-1)*N + i] = 1.0
        b_eq[N + i] = 0.0
    
    return A_eq, b_eq


def build_qp_subproblem(
    x_curr: np.ndarray,
    T: int,
    N: int,
    d_safe: float,
    mu: float,
    use_continuous_collision: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    构建完整的QP子问题
    
    QP形式:
        min  1/2 Δx^T H Δx + c^T Δx
        s.t. A_eq Δx = b_eq
             A_ineq Δx ≤ b_ineq
    
    参数:
        x_curr: 当前轨迹 (T, N)
        T: 时间步数
        N: 状态维度
        d_safe: 安全距离
        mu: 惩罚系数
        use_continuous_collision: 是否使用连续时间碰撞检测（默认False）
    
    返回:
        H: Hessian矩阵
        c: 梯度向量
        A_eq: 等式约束矩阵
        b_eq: 等式约束向量
        A_ineq: 不等式约束矩阵
        b_ineq: 不等式约束向量
        M: 总变量维度
    """
    # 1. 凸化碰撞约束（确定松弛变量数量）
    A_ineq, b_ineq, num_slack = convexify_collision_constraints(
        x_curr, d_safe, T, N, use_continuous=use_continuous_collision
    )
    
    # 2. 凸化目标函数
    H, c = convexify_objective(x_curr, T, N, num_slack, mu)
    
    # 3. 添加端点约束
    A_eq, b_eq = add_endpoint_constraints(T, N, num_slack)
    
    M = T * N + num_slack
    
    return H, c, A_eq, b_eq, A_ineq, b_ineq, M
