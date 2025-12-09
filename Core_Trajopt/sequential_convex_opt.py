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
FTOL = 1e-4          # ftol: Merit function 改进阈值
CTOL = 1e-4          # ctol: 约束满足阈值
D_SAFE = 0.1         # 安全距离


# # --- 占位函数：求解二次规划 (QP Solver Placeholder) ---
# def solve_qp(H, c, A_eq, b_eq, A_ineq, b_ineq, s_trust_region, M) -> np.ndarray:
#     """
#     【占位函数】模拟 QP 求解器。
    
#     TrajOpt 的 QP 子问题:
#     min Δx  (1/2 * Δx^T * H * Δx + c^T * Δx)
#     subject to: 线性约束, 信赖域约束 ||Δx|| <= s
    
#     由于我们没有集成 Gurobi/cvxpy，这里使用一个粗糙的占位符：
#     如果初始轨迹不撞墙，就假设 QP 总是计算出沿着梯度方向的一个小步长。
#     """
#     # 模拟 QP 求解器返回的步长 Δx
    
#     # 在实际的 QP 中，Hessian H 是正定的，确保问题是凸的。
#     # 我们用一个小的梯度下降步长来模拟 QP 的效果。
    
#     # 如果没有线性项，QP 解是 Δx = 0 (我们不希望这样)
#     if np.linalg.norm(c) < 1e-6:
#         return np.zeros(M)
    
#     # 模拟步长（反梯度方向，被信赖域约束）
#     step_length = 0.1 # 假设的步长
#     # 强制步长不超过信赖域
#     step_length = min(step_length, s_trust_region)
    
#     # 沿着负梯度方向走
#     delta_x = -c * step_length / np.linalg.norm(c)
    
#     return delta_x

# --- 核心算法：Sequential Convex Optimization ---

def trajopt_sco_solver(
    trajectory_x_init: np.ndarray, 
    trajectory_u_init: np.ndarray,
    goal_state: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    TrajOpt 的 SCO 主循环 (Algorithm 1)。

    参数:
    trajectory_x_init: 初始状态轨迹 (T, STATE_DIM)。
    trajectory_u_init: 初始控制轨迹 (T-1, CONTROL_DIM)。
    goal_state: 目标状态 (STATE_DIM)。
    
    返回:
    Tuple[final_x, final_u, success]: 最终轨迹和是否成功。
    """
    # 初始化变量
    x_curr = trajectory_x_init.copy()
    u_curr = trajectory_u_init.copy()
    mu = MU_INITIAL
    s = S_INITIAL
    T = x_curr.shape[0]
    N = STATE_DIM
    M = T * N + (T - 1) * CONTROL_DIM # 总优化变量维度
    
    # 将所有变量堆叠成一个大向量 X (状态 x 和控制 u)
    X_curr = np.hstack([x_curr.flatten(), u_curr.flatten()])
    
    # --- 辅助函数：计算当前 Merit Function 和约束违反程度 ---
    def calculate_current_violations(X_vector):
        x_flat = X_vector[:T * N]
        u_flat = X_vector[T * N:]
        x_traj = x_flat.reshape(T, N)
        u_traj = u_flat.reshape(T - 1, CONTROL_DIM)

        # 1. 目标函数成本 (f_cost)
        diffs = x_traj[1:] - x_traj[:-1]
        f_cost = np.sum(diffs**2)

        # 2. 运动学约束违反 (等式约束 h)
        # 运动学约束 h = x_{t+1} - f(x_t, u_t)
        kin_violations = []
        for t in range(T - 1):
            x_t_pred = update_state(x_traj[t], u_traj[t])
            kin_violations.append(x_traj[t+1] - x_t_pred)
        residual_eq = np.sum(np.abs(np.concatenate(kin_violations)))

        # 3. 碰撞约束违反 (不等式约束 g)
        # 碰撞约束: sd(x_t, obs) ≥ d_safe 对所有障碍物
        # 违反量: max(0, d_safe - sd)
        collision_violations = []
        for t in range(T):
            for obs in OBSTACLES:
                # 使用 compute_signed_distance_and_gradient 计算带符号距离
                sd, _, _, _ = compute_signed_distance(x_traj[t], obs)
                
                # 计算违反量 (铰链函数)
                violation = max(0, D_SAFE - sd)
                collision_violations.append(violation)
        
        # 总碰撞违反量 (L1 范数)
        residual_ineq = np.sum(collision_violations)
        
        # 4. 目标位姿约束 (x_T = goal_state)
        goal_error = x_traj[-1] - goal_state
        residual_goal = np.sum(np.abs(goal_error))
        
        # 综合等式违反
        total_residual_eq = residual_eq + residual_goal
        
        return f_cost, residual_ineq, total_residual_eq

    # 初始评估
    f_cost, residual_ineq, residual_eq = calculate_current_violations(X_curr)
    merit_old = compute_merit_function(f_cost, residual_ineq, residual_eq, mu)
    
    # --- 1. PenaltyIteration (外层循环：增加惩罚系数 μ) ---
    for pen_iter in range(MAX_ITER_PENALTY):
        
        # 检查是否满足约束
        if residual_ineq < CTOL and residual_eq < CTOL:
            print(f"SCO Success: Constraints satisfied (μ={mu:.2f})")
            return X_curr[:T*N].reshape(T,N), X_curr[T*N:].reshape(T-1, CONTROL_DIM), True
        
        print(f"\n--- PENALTY ITERATION {pen_iter + 1}: μ={mu:.2f}, s={s:.2f} ---")
        
        # --- 2. ConvexifyIteration (内层循环：解决凸子问题) ---
        for conv_iter in range(MAX_ITER_CONVEXIFY):
            
            # --- 构造 QP 子问题 (线性化和二次化) ---
            
            # a. 目标函数 (路径长度) - 二次项 H 和线性项 c
            grad_path_state, H_path_state = linearize_and_quadraticize_path_cost(x_curr)
            
            # 扩展梯度和 Hessian 到完整的优化变量维度 (包含控制)
            # grad_path_state 只对状态有梯度 (T*N,)，控制部分梯度为 0
            grad_path = np.zeros(M)
            grad_path[:T*N] = grad_path_state  # 状态部分
            # grad_path[T*N:] = 0  # 控制部分（已经是0）
            
            # Hessian 也需要扩展
            H_path = np.zeros((M, M))
            H_path[:T*N, :T*N] = H_path_state  # 状态-状态块
            # H_path[T*N:, T*N:] = 0  # 控制-控制块（已经是0）
            
            # b. 碰撞惩罚项 (L1-ineq) - 线性近似和梯度
            grad_col_penalty = np.zeros(M)
            col_approximations = linearize_all_collisions(x_curr, D_SAFE)
            # 累加所有碰撞约束的梯度
            for col_term in col_approximations:
                t = col_term['time_step']  # 时间步索引
                gradient = col_term['gradient']  # 对应的梯度 (shape: (STATE_DIM,))
                
                # 将该时间步的梯度累加到总梯度向量的对应位置
                # 注意：grad_col_penalty 是针对整个轨迹的大向量 (shape: (M,))
                # 其中前 T*N 个元素对应状态，后面对应控制
                idx_start = t * N
                idx_end = (t + 1) * N
                grad_col_penalty[idx_start:idx_end] += gradient

            # c. 运动学约束惩罚项 (L1-eq) - 线性近似和梯度
            kin_constraints = linearize_all_kinematics(x_curr, u_curr)
            grad_kin_penalty = np.zeros(M)
            # 遍历所有运动学约束
            for kin_term in kin_constraints:
                t = kin_term['time_step']  # 时间步索引
                A_lin = kin_term['A_lin']  # 雅可比矩阵 (N, 2N+U)
                residual = kin_term['residual']  # 当前违反量 (N,)
                
                # L1 惩罚的次梯度：sign(residual)
                sign_residual = np.sign(residual)  # (N,)
                
                # 梯度 = sign(h)ᵀ · J = (N,)ᵀ · (N, 2N+U) = (2N+U,)
                local_grad = sign_residual @ A_lin  # shape: (2N+U,)
                
                # 将局部梯度放到大向量的对应位置
                # A_lin 的列对应: [x_t (N维), u_t (U维), x_{t+1} (N维)]
                
                # x_t 的位置
                idx_xt = t * N
                grad_kin_penalty[idx_xt:idx_xt+N] += local_grad[:N]
                
                # u_t 的位置（在状态部分之后）
                idx_ut = T * N + t * CONTROL_DIM
                grad_kin_penalty[idx_ut:idx_ut+CONTROL_DIM] += local_grad[N:N+CONTROL_DIM]
                
                # x_{t+1} 的位置
                idx_xt1 = (t + 1) * N
                grad_kin_penalty[idx_xt1:idx_xt1+N] += local_grad[N+CONTROL_DIM:]
            
            # d. 综合 QP 目标项 (Hessian H 和梯度 c)
            H = H_path  # 路径长度的 Hessian (M, M)
            c = grad_path  # 路径长度的梯度 (M,)
            
            # 验证维度
            assert H.shape == (M, M), f"Hessian 维度错误: {H.shape} != {(M, M)}"
            assert c.shape == (M,), f"梯度 c 维度错误: {c.shape} != {(M,)}"
            
            # 实际需要将 mu * 碰撞/运动学惩罚的线性化梯度加到 c 中
            c += mu * (grad_col_penalty + grad_kin_penalty)
            
            merit_old = compute_merit_function(f_cost, residual_ineq, residual_eq, mu)
            
            # --- 3. TrustRegionIteration (最内层循环：尝试步长) ---
            for trust_iter in range(MAX_ITER_TRUST_REGION):
                
                # 求解 QP (获取增量 ΔX)
                delta_X, success = solve_qp(H, c, s, None, None, None, None)
                
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
                if s < 1e-8:
                    print("Trust region collapsed.")
                    break # 跳出 Trust Region 循环
            
            # 检查收敛 (基于 Merit Function 改进)
            if true_improve < FTOL:
                print(f"Convexify loop converged (Merit change < {FTOL})")
                break # 跳出 Convexify 循环
        
        # 惩罚系数更新 (如果约束未满足)
        if residual_ineq > CTOL or residual_eq > CTOL:
            mu *= K_PENALTY
            s = S_INITIAL # 重新初始化信赖域
        
    print("SCO Failure: Max iterations reached.")
    return X_curr[:T*N].reshape(T,N), X_curr[T*N:].reshape(T-1, CONTROL_DIM), False


# --- 示例运行块 ---
if __name__ == '__main__':
    
    # 1. 设置模型和初始轨迹
    T_steps = 20 # 时间步数
    
    # 初始状态: [x, y, theta]
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([4.0, 6.0, np.pi/2]) 
    
    # 线性插值生成初始轨迹
    x_init = np.linspace(start, goal, T_steps)
    u_init = np.zeros((T_steps - 1, CONTROL_DIM))
    
    # 2. 运行求解器
    final_x, final_u, success = trajopt_sco_solver(x_init, u_init, goal)
    
    if success:
        print("\nOptimization Successful!")
    else:
        print("\nOptimization Failed (Check constraints).")
        
    # 可视化最终轨迹 (需要运行 visualization.py 中的 dynamic_visualization_test)
    # 简化：只打印结果
    print(f"Final Path Length Cost: {linearize_and_quadraticize_path_cost(final_x)[0]}")
    # print(final_x)