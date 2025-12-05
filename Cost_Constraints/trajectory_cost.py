# 03_Costs_Constraints/trajectory_cost.py

import numpy as np
from typing import Tuple
# 假设我们从 Kinematics 文件夹导入状态维度
from Kinematics.state_definitions import STATE_DIM

# --- 目标函数：最小路径长度 ---

def compute_path_length_cost(trajectory: np.ndarray) -> float:
    """
    计算轨迹的总路径长度成本（平方差之和）。
    
    目标函数: f(x) = Σ_{t=1}^{T-1} ||x_{t+1} - x_t||^2
    
    参数:
    trajectory: 形状为 (T, STATE_DIM) 的数组，T 是时间步数。
    
    返回:
    float: 总路径长度成本。
    """
    # 确保轨迹至少有两个时间步
    if trajectory.shape[0] < 2:
        return 0.0
        
    # 计算连续时间步之间的差异
    diffs = trajectory[1:] - trajectory[:-1]
    
    # 成本是所有差异向量的 L2 范数平方之和
    cost = np.sum(diffs**2)
    return cost


# --- 目标函数：梯度和 Hessian (用于二次近似) ---

def linearize_and_quadraticize_path_cost(
    trajectory: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算路径长度成本函数 f(x) 的梯度 (∇f) 和 Hessian (∇²f) 矩阵。
    
    由于 f(x) 是二次函数，它的二次近似是精确的。
    Hessian 是常数，梯度是线性的。
    
    参数:
    trajectory: 形状为 (T, STATE_DIM) 的数组。
    
    返回:
    Tuple[gradient, hessian]: 梯度和 Hessian 矩阵，它们都是针对所有优化变量（所有时间步的状态）的。
    """
    T, N = trajectory.shape # T: 时间步数, N: 状态维度 (3 for our car)
    
    # 总优化变量维度 M = T * N
    M = T * N
    
    # 1. Hessian 矩阵 (∇²f) - 恒定且稀疏的块对角矩阵
    H = np.zeros((M, M))
    
    # 核心公式: ∇²f = 2 * (D^T D)
    # 其中 D 是一个稀疏矩阵，用于计算连续状态的差异 (x_{t+1} - x_t)
    
    # D 矩阵的结构：
    # D = [
    #   [-I  I  0  0 ...]
    #   [ 0 -I  I  0 ...]
    #   [ 0  0 -I  I ...]
    # ]
    I_N = np.eye(N)
    
    for t in range(T - 1):
        # 成本 ||x_{t+1} - x_t||^2
        
        # 块 t 到 t
        idx_t = t * N
        H[idx_t:idx_t+N, idx_t:idx_t+N] += 2 * I_N
        
        # 块 t+1 到 t+1
        idx_t1 = (t + 1) * N
        H[idx_t1:idx_t1+N, idx_t1:idx_t1+N] += 2 * I_N
        
        # 块 t 到 t+1 (交叉项)
        H[idx_t:idx_t+N, idx_t1:idx_t1+N] += -2 * I_N
        H[idx_t1:idx_t1+N, idx_t:idx_t+N] += -2 * I_N

    # 2. 梯度向量 (∇f) - 线性依赖于当前轨迹
    # ∇f = 2 * D^T * (D x)
    # ∇f_t = -2 * (x_{t+1} - x_t) + 2 * (x_t - x_{t-1})
    
    grad = np.zeros(M)
    
    # 处理中间点 (t = 1 到 T-2)
    for t in range(1, T - 1):
        x_prev = trajectory[t-1]
        x_curr = trajectory[t]
        x_next = trajectory[t+1]
        
        # 梯度块对应于 x_t
        grad_t = 2 * (x_curr - x_prev) - 2 * (x_next - x_curr)
        grad[t*N:(t+1)*N] = grad_t

    # 处理起点 (t = 0)
    x_curr = trajectory[0]
    x_next = trajectory[1]
    grad_0 = -2 * (x_next - x_curr)
    grad[0:N] = grad_0
    
    # 处理终点 (t = T-1)
    x_prev = trajectory[T-2]
    x_curr = trajectory[T-1]
    grad_T1 = 2 * (x_curr - x_prev)
    grad[(T-1)*N:M] = grad_T1
    
    return grad, H


# --- 示例运行块 ---
if __name__ == '__main__':
    # 示例轨迹：3个时间步，2个状态维度 (简化)
    # x0 = (0, 0), x1 = (1, 1), x2 = (2, 0)
    
    # 初始轨迹 (T=3, N=2)
    trajectory_2d = np.array([
        [0.0, 0.0],  # t=0
        [1.0, 1.0],  # t=1
        [2.0, 0.0]   # t=2
    ])
    
    # 1. 计算成本
    cost = compute_path_length_cost(trajectory_2d)
    # ||x1-x0||^2 = ||(1, 1)||^2 = 2
    # ||x2-x1||^2 = ||(1, -1)||^2 = 2
    # 总成本应为 4.0
    print(f"Total Path Length Cost: {cost:.2f}")

    # 2. 计算梯度和 Hessian
    grad, H = linearize_and_quadraticize_path_cost(trajectory_2d)
    
    print("\nTotal Optimization Variables (M):", len(grad))
    print("\nGradient Vector (∇f):\n", grad)
    # 期望梯度:
    # ∇f_0 = -2(x1-x0) = -2(1, 1) = (-2, -2)
    # ∇f_1 = 2(x1-x0) - 2(x2-x1) = 2(1, 1) - 2(1, -1) = (2-2, 2-(-2)) = (0, 4)
    # ∇f_2 = 2(x2-x1) = 2(1, -1) = (2, -2)
    # 期望：[-2. -2.  0.  4.  2. -2.]
    
    print("\nHessian Matrix (∇²f, Sparse):\n", H)
    # H 是稀疏矩阵，非零元素主要在对角线和次对角线上。
    # 中间块 (t=1) 应该为 4 * I