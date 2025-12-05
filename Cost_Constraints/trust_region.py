# 03_Costs_Constraints/trust_region.py

from typing import Tuple, Dict

# --- 信赖域更新参数 (来自 Algorithm 1) ---
# 这些参数决定了信赖域的收缩和扩张速度
TAU_PLUS = 2.0      # τ⁺: 扩张因子 (论文中未给出具体值，通常 > 1.0)
TAU_MINUS = 0.5     # τ⁻: 收缩因子 (论文中未给出具体值，通常 < 1.0)
C_ACCEPT = 0.1      # c: 步长接受参数 (论文中要求 TrueImprove / ModelImprove > c)
XTOL = 1e-4         # xtol: 信赖域收缩到此阈值时可能终止


def compute_merit_function(
    f_cost: float, 
    residual_ineq: float, 
    residual_eq: float, 
    mu: float
) -> float:
    """
    计算非凸问题的 Merit Function (优点函数)。
    
    Merit Function = Objective Cost + Penalty Coefficient * Sum(|Violations|)
    Merit(x) = f(x) + μ * Σ|g(x)|⁺ + μ * Σ|h(x)|
    
    注意：在 TrajOpt 中，Merit Function 使用 L1 惩罚项。
    
    参数:
    f_cost: 轨迹目标函数的实际值 f(x)。
    residual_ineq: 所有不等式约束的 L1 违反总和 Σ|g(x)|⁺。
    residual_eq: 所有等式约束的 L1 违反总和 Σ|h(x)|。
    mu: 当前惩罚系数。
    
    返回:
    float: Merit Function 的值。
    """
    return f_cost + mu * (residual_ineq + residual_eq)


def evaluate_step(
    merit_old: float, 
    merit_new: float, 
    model_improve: float
) -> Tuple[float, float]:
    """
    评估步长，计算真实改进和改进比率。

    参数:
    merit_old: 当前解 x_old 的 Merit Function 值。
    merit_new: 新解 x_new 的 Merit Function 值。
    model_improve: 凸模型预测的 Merit Function 改进 (ModelImprove)。

    返回:
    Tuple[true_improve, ratio]: 真实改进 (TrueImprove) 和改进比率 (ratio)。
    """
    # 真实改进 (TrueImprove)
    true_improve = merit_old - merit_new
    
    # 改进比率 (ratio)
    if model_improve > 0:
        ratio = true_improve / model_improve
    else:
        # 如果模型预测改进很小或为负，但真实改进为正，则给一个高比率接受
        ratio = 999.0 if true_improve > 0 else 0.0
        
    return true_improve, ratio


def update_trust_region(
    current_s: float, 
    ratio: float
) -> Tuple[float, bool]:
    """
    根据改进比率更新信赖域大小 s。
    
    参数:
    current_s: 当前信赖域大小 s。
    ratio: TrueImprove / ModelImprove 的比率。

    返回:
    Tuple[new_s, accepted]: 更新后的信赖域大小 new_s 和步长是否被接受 (accepted)。
    """
    accepted = False
    new_s = current_s
    
    if ratio > C_ACCEPT:
        # 1. 接受步长：真实改进足够大
        accepted = True
        
        # 扩大信赖域 (如果改进效果很好，可以迈更大的步)
        if ratio > 0.75: # 设定一个高阈值来触发更积极的扩张 (通常 > 0.75)
            new_s = max(current_s, TAU_PLUS * current_s) # 扩张
        
    elif ratio < C_ACCEPT:
        # 2. 拒绝步长：真实改进不足
        accepted = False
        
        # 缩小信赖域 (模型近似不准，下次走小步)
        new_s = TAU_MINUS * current_s
        
        # 如果信赖域收缩到 XTOL 以下，可以考虑退出 Trust Region 迭代
        if new_s < XTOL:
             pass # 在主循环中处理终止
             
    # 如果 C_ACCEPT <= ratio < 1.0 (改进但不够好)，则保持信赖域大小
    
    return new_s, accepted


# --- 示例运行块 ---
if __name__ == '__main__':
    # 场景 1: 模型预测准确，改进良好
    mu_penalty = 100.0
    merit_old = 50.0
    merit_new = 30.0
    model_improve = 20.0  # 模型预测改进了 20.0
    
    true_improve, ratio = evaluate_step(merit_old, merit_new, model_improve)
    
    print(f"--- 场景 1: 完美改进 ---")
    print(f"真实改进 (TrueImprove): {true_improve:.2f}")
    print(f"改进比率 (Ratio): {ratio:.2f}")

    current_s = 1.0
    new_s, accepted = update_trust_region(current_s, ratio)
    print(f"步长接受: {accepted}, 新信赖域 s: {new_s:.2f} (扩张)")


    # 场景 2: 模型预测太乐观 (近似不准)
    merit_new_bad = 45.0
    model_improve_bad = 20.0 # 模型仍预测改进 20
    
    true_improve_bad, ratio_bad = evaluate_step(merit_old, merit_new_bad, model_improve_bad)
    
    print(f"\n--- 场景 2: 预测过于乐观 (Ratio < {C_ACCEPT}) ---")
    print(f"真实改进 (TrueImprove): {true_improve_bad:.2f}") # 实际只改进了 5.0
    print(f"改进比率 (Ratio): {ratio_bad:.2f}") 

    current_s = 1.0
    new_s_bad, accepted_bad = update_trust_region(current_s, ratio_bad)
    print(f"步长接受: {accepted_bad}, 新信赖域 s: {new_s_bad:.2f} (收缩)")