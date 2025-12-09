# 02_Core_TrajOpt/sequential_convex_opt.py

import numpy as np
from typing import Tuple
# 导入模型和成本
from Kinematics.state_definitions import STATE_DIM, CONTROL_DIM
from Kinematics.car_model import update_state
from Cost_Constraints.no_collision_cost import compute_signed_distance, linearize_all_collisions
from Cost_Constraints.trajectory_cost import linearize_and_quadraticize_path_cost
from Cost_Constraints.kinematics_constraint import linearize_all_kinematics
from Cost_Constraints.trust_region import (
    compute_merit_function, evaluate_step, update_trust_region, 
    TAU_PLUS, TAU_MINUS, C_ACCEPT
)
from Environment.obstacles import OBSTACLES
from Core_Trajopt.qp_solver_interface import solve_qp

# --- 算法参数 (Algorithm 1) ---
MU_INITIAL = 1.0     # μ₀: 初始惩罚系数
S_INITIAL = 1.0      # s₀: 初始信赖域大小
K_PENALTY = 10.0     # k: 惩罚系数缩放因子
MAX_ITER_PENALTY = 20
MAX_ITER_CONVEXIFY = 100
MAX_ITER_TRUST_REGION = 10
XTOL = 1e-6          # xtol: 变量改进阈值
FTOL = 1e-4          # ftol: Merit function 改进阈值
CTOL = 1e-4          # ctol: 约束满足阈值
D_SAFE = 0.1         # 安全距离

# --- 核心算法：Sequential Convex Optimization ---

def trajopt_sco_solver(
    trajectory_x_init: np.ndarray, 
) -> Tuple[np.ndarray, bool]:
    """
    TrajOpt 的 SCO 主循环 (Algorithm 1)。

    参数:
    trajectory_x_init: 初始状态轨迹 (T, STATE_DIM)。
    goal_state: 目标状态 (STATE_DIM)。
    
    返回:
    Tuple[final_x, final_u, success]: 最终轨迹和是否成功。
    """
    # 初始化变量
    x_curr = trajectory_x_init.copy()
    mu = MU_INITIAL
    s = S_INITIAL
    T = x_curr.shape[0]
    N = STATE_DIM
    
    # 计算松弛变量数量：每个时间步 × 每个障碍物对应一个松弛变量
    num_obstacles = len(OBSTACLES)
    num_slack = T * num_obstacles  # 松弛变量总数
    
    M = T * N + num_slack  # 总优化变量维度 (状态 + 松弛变量)
    
    # 将所有变量堆叠成一个大向量 X = [x_0, ..., x_{T-1}, t_0, ..., t_{num_slack-1}]
    # 其中 x 是状态，t 是松弛变量
    slack_curr = np.zeros(num_slack)  # 初始化松弛变量为0
    X_curr = np.hstack([x_curr.flatten(), slack_curr])
    
    # --- 辅助函数：计算当前 Merit Function 和约束违反程度 ---
    def calculate_current_violations(X_vector):
        x_flat = X_vector[:T * N]
        slack_flat = X_vector[T * N:]  # 提取松弛变量
        x_traj = x_flat.reshape(T, N)

        # 1. 目标函数成本 (f_cost)
        diffs = x_traj[1:] - x_traj[:-1]
        f_cost = np.sum(diffs**2)
        
        # 添加松弛变量的惩罚 (在QP中通过线性项 mu*t 实现)
        # 这里只计算路径成本，松弛变量惩罚在 merit function 中体现

        # 3. 碰撞约束违反 (不等式约束 g)
        # 原始约束: sd(x_t, obs) ≥ d_safe
        # 转换后: d_safe - sd(x_t, obs) ≤ t_{t,obs}
        # 违反量: max(0, d_safe - sd - t)
        collision_violations = []
        for t in range(T):
            for i, obs in enumerate(OBSTACLES):
                # 使用 compute_signed_distance 计算带符号距离
                sd, _, _, _ = compute_signed_distance(x_traj[t], i)
                
                # 计算违反量 (考虑松弛变量)
                violation = max(0, D_SAFE - sd)
                collision_violations.append(violation)
        
        # 总碰撞违反量 (L1 范数)
        residual_ineq = np.sum(collision_violations)

        # 等式约束违反 (h)
        residual_eq = 0.0
        
        return f_cost, residual_ineq, residual_eq

    # 初始评估
    f_cost, residual_ineq, residual_eq = calculate_current_violations(X_curr)
    merit_old = compute_merit_function(f_cost, residual_ineq, residual_eq, mu)
    
    # --- 1. PenaltyIteration (外层循环：增加惩罚系数 μ) ---
    for pen_iter in range(MAX_ITER_PENALTY):
        
        # 检查是否满足约束
        if residual_ineq < CTOL and residual_eq < CTOL:
            print(f"SCO Success: Constraints satisfied (μ={mu:.2f})")
            return X_curr[:T*N].reshape(T,N), True
        
        print(f"\n--- PENALTY ITERATION {pen_iter + 1}: μ={mu:.2f}, s={s:.2f} ---")
        
        # --- 2. ConvexifyIteration (内层循环：解决凸子问题) ---
        for conv_iter in range(MAX_ITER_CONVEXIFY):
            
            # --- 构造 QP 子问题 (线性化和二次化) ---
            
            # a. 目标函数 (路径长度) - 二次项 H 和线性项 c
            grad_path_state, H_path_state = linearize_and_quadraticize_path_cost(x_curr)
            
            # 扩展梯度和 Hessian 到完整的优化变量维度 (包含松弛变量)
            # grad_path_state 只对状态有梯度 (T*N,)，松弛变量部分梯度为 0
            grad_path = np.zeros(M)
            grad_path[:T*N] = grad_path_state  # 状态部分
            # grad_path[T*N:] = 0  # 松弛变量部分（已经是0）
            
            # Hessian 也需要扩展
            H_path = np.zeros((M, M))
            H_path[:T*N, :T*N] = H_path_state  # 状态-状态块
            # H_path[T*N:, T*N:] = 0  # 松弛变量-松弛变量块（已经是0）
            
            # b. 碰撞约束转换为松弛变量形式
            # 将 |a·Δx + b|⁺ 转换为 QP: min μ·t, s.t. 0 ≤ t, a·Δx + b ≤ t
            col_approximations = linearize_all_collisions(x_curr, D_SAFE)
            
            # 构建不等式约束矩阵 A_ineq 和向量 b_ineq
            # 约束形式: A_ineq @ ΔX ≤ b_ineq
            num_col_constraints = len(col_approximations)
            # 每个碰撞约束生成2个不等式: (1) a·Δx - t ≤ -b  (2) -t ≤ 0
            A_ineq = np.zeros((2 * num_col_constraints, M))
            b_ineq = np.zeros(2 * num_col_constraints)
            
            slack_idx = 0
            for constraint_idx, col_term in enumerate(col_approximations):
                t = col_term['time_step']  # 时间步索引
                gradient = col_term['gradient']  # ∇g(x) (shape: (STATE_DIM,))
                offset = col_term['initial_value']  # g(x₀) - ∇g(x₀)·x₀
                
                # 约束1: a·Δx - t ≤ -b  =>  ∇g·Δx - t ≤ -(g(x₀) - ∇g·x₀)
                idx_start = t * N
                idx_end = (t + 1) * N
                A_ineq[2*constraint_idx, idx_start:idx_end] = gradient  # a·Δx 部分
                A_ineq[2*constraint_idx, T*N + slack_idx] = -1.0  # -t 部分
                b_ineq[2*constraint_idx] = -offset  # -b
                
                # 约束2: -t ≤ 0  (确保 t ≥ 0)
                A_ineq[2*constraint_idx + 1, T*N + slack_idx] = -1.0
                b_ineq[2*constraint_idx + 1] = 0.0
                
                slack_idx += 1
            
            # c. 综合 QP 目标项 (Hessian H 和梯度 c)
            H = H_path  # 路径长度的 Hessian (M, M)
            c = grad_path.copy()  # 路径长度的梯度 (M,)
            
            # 添加松弛变量的线性惩罚项 μ·t
            # 对每个松弛变量 t_i，目标函数添加 μ·t_i
            c[T*N:] += mu  # 所有松弛变量的系数都是 μ
            
            # 验证维度
            assert H.shape == (M, M), f"Hessian 维度错误: {H.shape} != {(M, M)}"
            assert c.shape == (M,), f"梯度 c 维度错误: {c.shape} != {(M,)}"
            assert A_ineq.shape[1] == M, f"不等式约束维度错误: {A_ineq.shape[1]} != {M}"
            
            merit_old = compute_merit_function(f_cost, residual_ineq, residual_eq, mu)
            
            # --- 3. TrustRegionIteration (最内层循环：尝试步长) ---
            for trust_iter in range(MAX_ITER_TRUST_REGION):
                
                # 求解 QP (获取增量 ΔX)
                delta_X, success = solve_qp(H, c, s, None, None, A_ineq, b_ineq)
                assert success, "QP 求解失败"

                # 计算模型预测的改进 (ModelImprove)
                # ModelImprove = -(c^T * ΔX + 1/2 * ΔX^T * H * ΔX)
                model_improve = -(c @ delta_X + 0.5 * delta_X @ H @ delta_X)
                
                # 尝试新解 X_new
                X_new = X_curr + delta_X
                f_cost_new, r_ineq_new, r_eq_new = calculate_current_violations(X_new)
                merit_new = compute_merit_function(f_cost_new, r_ineq_new, r_eq_new, mu)
                
                # 评估步长
                true_improve, ratio = evaluate_step(merit_old, merit_new, model_improve)
                
                # 更新信赖域 s 和决定是否接受步长
                s_new, accepted = update_trust_region(s, ratio)
                
                if accepted:
                    X_curr = X_new
                    s = s_new
                    f_cost, residual_ineq, residual_eq = f_cost_new, r_ineq_new, r_eq_new
                    merit_old = merit_new # 更新 Merit
                    print(f"  [{conv_iter}.{trust_iter}] Accepted. Merit={merit_old:.2f}, Ratio={ratio:.2f}, s={s:.2f}")
                    break # 跳出 Trust Region 循环
                else:
                    s = s_new
                    print(f"  [{conv_iter}.{trust_iter}] Rejected. Ratio={ratio:.2f}, s={s:.2f}")

                # 检查信赖域是否收缩到零 (可能陷入局部最优或终止)
                if s < XTOL:
                    print("Trust region collapsed.")
                    break # 跳出 Trust Region 循环
            
            # 检查收敛 (基于 Merit Function 改进)
            if true_improve < FTOL or np.linalg.norm(delta_X) < XTOL:
                print(f"Convexify loop converged (Merit change < {FTOL})")
                break # 跳出 Convexify 循环
        
        # 惩罚系数更新 (如果约束未满足)
        if residual_ineq > CTOL or residual_eq > CTOL:
            mu *= K_PENALTY
            s = S_INITIAL # 重新初始化信赖域
        
    print("SCO Failure: Max iterations reached.")
    return X_curr[:T*N].reshape(T,N), False


# --- 示例运行块 ---
if __name__ == '__main__':
    
    # 1. 设置模型和初始轨迹
    T_steps = 20 # 时间步数
    
    # 初始状态: [x, y, theta]
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([4.0, 6.0, np.pi/2]) 
    
    # 线性插值生成初始轨迹
    x_init = np.linspace(start, goal, T_steps)
    
    # 2. 运行求解器
    final_x, success = trajopt_sco_solver(x_init)
    
    if success:
        print("\nOptimization Successful!")
    else:
        print("\nOptimization Failed (Check constraints).")
        
    # 可视化最终轨迹 (需要运行 visualization.py 中的 dynamic_visualization_test)
    # 简化：只打印结果
    print(f"Final Path Length Cost: {linearize_and_quadraticize_path_cost(final_x)[0]}")
    # print(final_x)