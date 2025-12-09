# 02_Core_TrajOpt/qp_solver_interface.py

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

# 假设我们从 state_definitions 导入总变量维度 M (虽然在这个文件中暂时不用)
# from ..01_Kinematics.state_definitions import STATE_DIM, CONTROL_DIM 

def solve_qp(
    H: np.ndarray, 
    c: np.ndarray, 
    s_trust_region: float, 
    A_eq: Optional[np.ndarray] = None, 
    b_eq: Optional[np.ndarray] = None, 
    A_ineq: Optional[np.ndarray] = None, 
    b_ineq: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], bool]:
    """
    使用 CVXPY 封装对二次规划 (QP) 求解器的调用。

    QP 形式: min Δx (1/2 * Δx^T * H * Δx + c^T * Δx)
    约束:   ||Δx||_inf <= s_trust_region (信赖域)
             A_eq @ Δx == b_eq (可选的线性等式约束，例如运动学)
             A_ineq @ Δx <= b_ineq (可选的线性不等式约束，例如 L1 惩罚转换)

    参数:
    H (ndarray): Hessian 矩阵 (二次项的系数)。
    c (ndarray): 梯度向量 (线性项的系数)。
    s_trust_region (float): 信赖域大小。
    A_eq, b_eq (optional): 线性等式约束矩阵和向量。
    A_ineq, b_ineq (optional): 线性不等式约束矩阵和向量。

    返回:
    Tuple[Optional[np.ndarray], bool]: 最优增量 Δx 和是否求解成功。
    """
    
    M = c.shape[0]  # 总优化变量维度
    delta_x = cp.Variable(M)
    constraints = []

    # 1. 目标函数 (Quadratic Objective)
    # H 必须是对称的。cvxpy 接受 cp.quad_form(delta_x, H) 表示二次项。
    objective = cp.Minimize(0.5 * cp.quad_form(delta_x, H) + c.T @ delta_x)

    # 2. 约束条件

    # a. 信赖域约束 (Box/Infinity Norm Trust Region)
    # TrajOpt 通常使用 L-infinity 范数 (Box Constraint): ||Δx||_∞ <= s
    constraints += [cp.abs(delta_x) <= s_trust_region]
    
    # b. 线性等式约束 (如硬运动学约束)
    if A_eq is not None and b_eq is not None:
        constraints += [A_eq @ delta_x == b_eq]

    # c. 线性不等式约束 (如 L1 惩罚项转换后的约束)
    if A_ineq is not None and b_ineq is not None:
        constraints += [A_ineq @ delta_x <= b_ineq]

    # 3. 求解问题
    problem = cp.Problem(objective, constraints)
    
    try:
        # 使用 OSQP 或 ECOS 求解器
        problem.solve(solver=cp.OSQP, verbose=False) 
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # 求解成功
            return delta_x.value, True
        else:
            # 求解失败（如问题不可行）
            print(f"QP Warning: Solver status is {problem.status}. Not optimal.")
            return None, False
            
    except cp.SolverError as e:
        print(f"QP Error: Solver failed. {e}")
        return None, False


# --- 示例运行块 (测试) ---
if __name__ == '__main__':
    # 模拟 TrajOpt 步骤
    M = 3 # 假设只有 3 个变量 [x0, x1, x2]
    
    # 1. 目标函数 (最小化路径长度的近似)
    # 假设 Hessian H (路径长度) 和 梯度 c (当前不平滑造成的斜率)
    H_test = np.array([[2, -1, 0], 
                       [-1, 2, -1], 
                       [0, -1, 2]])
    c_test = np.array([-5.0, 1.0, 0.5]) # 强烈的负梯度，促使 x0 减小
    
    # 2. 信赖域和约束
    s_test = 0.5 # 信赖域大小
    
    # 线性等式约束示例: x0 + x1 + x2 = 1.0
    A_eq_test = np.array([[1.0, 1.0, 1.0]])
    b_eq_test = np.array([1.0])
    
    # 3. 求解
    delta_x_sol, success = solve_qp(H_test, c_test, s_test, A_eq=A_eq_test, b_eq=b_eq_test)
    
    print("\n--- QP Solver Test Results ---")
    if success:
        print("QP Solved Successfully.")
        print(f"Optimal Increment Δx: {delta_x_sol}")
        print(f"Δx Max Norm (should be <= {s_test}): {np.max(np.abs(delta_x_sol)):.4f}")
    else:
        print("QP Solver Failed.")