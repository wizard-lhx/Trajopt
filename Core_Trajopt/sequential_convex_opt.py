# 02_Core_TrajOpt/sequential_convex_opt.py

import numpy as np
from typing import Tuple
# 导入模型和成本
from Kinematics.state_definitions import STATE_DIM
from Cost_Constraints.no_collision_cost import compute_signed_distance
from Cost_Constraints.trust_region import (
    compute_merit_function, evaluate_step, update_trust_region, 
    TAU_PLUS, TAU_MINUS, C_ACCEPT
)
from Environment.obstacles import OBSTACLES
from Core_Trajopt.qp_solver_interface import solve_qp
from Core_Trajopt.convexify import build_qp_subproblem

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
    
    # 注意：松弛变量数量在每次迭代时动态确定，取决于实际的碰撞约束数量
    
    # --- 辅助函数：计算当前 Merit Function 和约束违反程度 ---
    def calculate_current_violations(x_traj):
        # 1. 目标函数成本 (f_cost)
        diffs = x_traj[1:] - x_traj[:-1]
        f_cost = np.sum(diffs**2)

        # 3. 碰撞约束违反 (不等式约束 g)
        # 原始约束: sd(x_t, obs) ≥ d_safe
        # 违反量: max(0, d_safe - sd)
        collision_violations = []
        for t in range(T):
            for i, obs in enumerate(OBSTACLES):
                # 使用 compute_signed_distance 计算带符号距离
                sd, _, _, _ = compute_signed_distance(x_traj[t], i)
                
                # 计算违反量（不考虑松弛变量，用于评估真实约束满足情况）
                violation = max(0, D_SAFE - sd)
                collision_violations.append(violation)
        
        # 总碰撞违反量 (L1 范数)
        residual_ineq = np.sum(collision_violations)

        # 等式约束违反 (h)
        residual_eq = 0.0
        
        return f_cost, residual_ineq, residual_eq

    # 初始评估
    f_cost, residual_ineq, residual_eq = calculate_current_violations(x_curr)
    merit_old = compute_merit_function(f_cost, residual_ineq, residual_eq, mu)
    
    # --- 1. PenaltyIteration (外层循环：增加惩罚系数 μ) ---
    for pen_iter in range(MAX_ITER_PENALTY):
        
        # 检查是否满足约束
        if residual_ineq < CTOL and residual_eq < CTOL:
            print(f"SCO Success: Constraints satisfied (μ={mu:.2f})")
            return x_curr, True
        
        print(f"\n--- PENALTY ITERATION {pen_iter + 1}: μ={mu:.2f}, s={s:.2f} ---")
        
        # --- 2. ConvexifyIteration (内层循环：解决凸子问题) ---
        for conv_iter in range(MAX_ITER_CONVEXIFY):
            
            # --- 构造 QP 子问题 (使用模块化的凸化函数) ---
            H, c, A_eq, b_eq, A_ineq, b_ineq, M = build_qp_subproblem(
                x_curr, T, N, D_SAFE, mu
            )
            
            if A_ineq is not None:
                num_col_constraints = A_ineq.shape[0] // 2 if A_ineq.size > 0 else 0
            else:
                num_col_constraints = 0
            print(f"  [conv_iter={conv_iter}] Active collision constraints: {num_col_constraints}, total vars: {M}")
            
            merit_old = compute_merit_function(f_cost, residual_ineq, residual_eq, mu)
            
            # --- 3. TrustRegionIteration (最内层循环：尝试步长) ---
            for trust_iter in range(MAX_ITER_TRUST_REGION):
                
                # ====== 记录QP求解器输入参数 ======
                print(f"\n  === QP Call [{conv_iter}.{trust_iter}] ===")
                print(f"  Trust region s = {s:.4f}")
                # 调试信息（可选）
                # print(f"  H shape: {H.shape}, norm: {np.linalg.norm(H):.4f}")
                # print(f"  c shape: {c.shape}, norm: {np.linalg.norm(c):.4f}")
                # print(f"  A_eq shape: {A_eq.shape}, rank: {np.linalg.matrix_rank(A_eq)} (endpoint constraints)")
                # print(f"  b_eq: all zeros (fixing start and end points)")
                # if A_ineq.size > 0:
                #     print(f"  A_ineq shape: {A_ineq.shape}, rank: {np.linalg.matrix_rank(A_ineq)}")
                #     print(f"  b_ineq shape: {b_ineq.shape}, min: {b_ineq.min():.4f}, max: {b_ineq.max():.4f}")
                # else:
                #     print(f"  A_ineq: empty (no collision constraints)")
                #     A_ineq = None
                #     b_ineq = None
                
                # 求解 QP (获取增量 ΔX)
                # 传入 num_state=T*N，使信赖域约束只应用于状态变量
                # 传入 A_eq, b_eq 确保起点和终点不变
                delta_X, success = solve_qp(H, c, s, A_eq, b_eq, A_ineq, b_ineq, num_state=T*N)
                
                if not success:
                    print(f"  QP FAILED!")
                else:
                    print(f"  QP SUCCESS: ||ΔX|| = {np.linalg.norm(delta_X):.4f}")
                    # print(f"  Δx state norm: {np.linalg.norm(delta_X[:T*N]):.4f}")
                    # print(f"  Δslack norm: {np.linalg.norm(delta_X[T*N:]):.4f}")
                    # print(f"  slack values min/max: {delta_X[T*N:].min():.4f} / {delta_X[T*N:].max():.4f}")
                # ====================================
                
                if not success:
                    # QP求解失败，缩小信赖域重试
                    s = s * TAU_MINUS
                    print(f"  Shrinking trust region to s={s:.4f}")
                    if s < XTOL:
                        print(f"  Trust region too small, breaking")
                        break
                    continue  # 跳过本次迭代，用新的信赖域重试

                # 计算模型预测的改进 (ModelImprove)
                # ModelImprove = -(c^T * ΔX + 1/2 * ΔX^T * H * ΔX)
                model_improve = -(c @ delta_X + 0.5 * delta_X @ H @ delta_X)
                
                # 尝试新解：只更新状态变量（松弛变量在每次迭代重新确定）
                delta_x_state = delta_X[:T*N]
                x_new = x_curr + delta_x_state.reshape(T, N)
                
                f_cost_new, r_ineq_new, r_eq_new = calculate_current_violations(x_new)
                merit_new = compute_merit_function(f_cost_new, r_ineq_new, r_eq_new, mu)
                
                # 评估步长
                true_improve, ratio = evaluate_step(merit_old, merit_new, model_improve)
                
                # 更新信赖域 s 和决定是否接受步长
                s_new, accepted = update_trust_region(s, ratio)
                
                if accepted:
                    x_curr = x_new
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
    return x_curr, False


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
    
    # 3. 使用PyBullet可视化初始轨迹和最终轨迹
    print("\n=== Starting PyBullet Visualization ===")
    import pybullet as p
    import time
    from utils.bullet_collision import BulletCollisionChecker
    
    # 创建带GUI的碰撞检测器用于可视化
    viz_checker = BulletCollisionChecker(use_gui=True)
    
    # 创建障碍物
    for i, obs in enumerate(OBSTACLES):
        cx, cy, w, l, theta = obs
        viz_checker.create_obstacle(i, cx, cy, w, l, theta)
    
    # 创建车辆（将用于显示轨迹）
    car_viz_id = viz_checker.create_car_body()
    
    # 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=10,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=[2, 3, 0],
        physicsClientId=viz_checker.physics_client
    )
    
    print("\n显示初始轨迹（红色）...")
    # 绘制初始轨迹（红色线）
    initial_line_ids = []
    for i in range(len(x_init) - 1):
        line_id = p.addUserDebugLine(
            [x_init[i][0], x_init[i][1], 0.1],
            [x_init[i+1][0], x_init[i+1][1], 0.1],
            lineColorRGB=[1, 0, 0],  # 红色
            lineWidth=3,
            physicsClientId=viz_checker.physics_client
        )
        initial_line_ids.append(line_id)
    
    # 绘制初始轨迹点
    for i, state in enumerate(x_init):
        p.addUserDebugText(
            f"{i}",
            [state[0], state[1], 0.3],
            textColorRGB=[1, 0, 0],
            textSize=0.8,
            physicsClientId=viz_checker.physics_client
        )
    
    time.sleep(2)
    
    print("显示最终轨迹（绿色）...")
    # 绘制最终轨迹（绿色线）
    final_line_ids = []
    for i in range(len(final_x) - 1):
        line_id = p.addUserDebugLine(
            [final_x[i][0], final_x[i][1], 0.1],
            [final_x[i+1][0], final_x[i+1][1], 0.1],
            lineColorRGB=[0, 1, 0],  # 绿色
            lineWidth=3,
            physicsClientId=viz_checker.physics_client
        )
        final_line_ids.append(line_id)
    
    # 绘制最终轨迹点
    for i, state in enumerate(final_x):
        p.addUserDebugText(
            f"{i}",
            [state[0], state[1], 0.5],
            textColorRGB=[0, 1, 0],
            textSize=0.8,
            physicsClientId=viz_checker.physics_client
        )
    
    # 添加图例
    p.addUserDebugText(
        "红色 = 初始轨迹",
        [-1, 7, 1],
        textColorRGB=[1, 0, 0],
        textSize=1.2,
        physicsClientId=viz_checker.physics_client
    )
    p.addUserDebugText(
        "绿色 = 优化后轨迹",
        [-1, 6.5, 1],
        textColorRGB=[0, 1, 0],
        textSize=1.2,
        physicsClientId=viz_checker.physics_client
    )
    
    print("\n动画演示优化后的轨迹...")
    # 动画演示最终轨迹
    for i, state in enumerate(final_x):
        # 更新车辆位置
        viz_checker.update_car_pose(state)
        time.sleep(1)
    
    print("\n可视化完成。按Enter键退出...")
    input()
    
    # 清理
    viz_checker.cleanup()