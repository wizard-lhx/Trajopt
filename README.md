# 整体目录
```
TrajOpt_2D_Car_Project/
├── Kinematics/
│   ├── car_model.py                # 2D 小车模型 (独轮小车模型，实现在pybullet用上下左右键操作小车)
│   └── state_definitions.py        # 定义 State (x, y, theta) 和 Control (vx, omega)
├── Core_TrajOpt/
│   ├── sequential_convex_opt.py    # 实现主循环 (SCO 框架)
│   ├── convexify.py                # 将目标函数及约束条件凸化，并组成新的包括松弛变量的长优化向量
│   ├── qp_solver_interface.py      # 封装对 QP 求解器的调用 (如 OSQP/Gurobi/cvxpy)
│   └── penalty_utilities.py        # 加入松弛变量，实现 L1 惩罚项到线性约束的转换（没有使用到）
├── Costs_Constraints/
│   ├── trajectory_cost.py          # 目标函数：最小路径长度 (Σ||x_{t+1}-x_t||^2)
│   ├── no_collision_cost.py        # 核心：实现碰撞检测的线性化和惩罚
│   ├── kinematics_constraint.py    # 实现运动学等式约束的线性化（没有使用到）
│   └── trust_region.py             # 实现步长评估和信赖域更新逻辑
├── Environment/
│   └── obstacles.py                # 2D 障碍物定义 (多边形/圆形，用于碰撞检测)
├── test/
│   ├── distance3d_demo.py          # 测试 distance3d 包的gdk，mtv算法
│   └── pybullet_keyboard.py        # 测试 pybullet 包
├── utils/
│   └── bullet_collision.py         # 碰撞检测器
├── README.md                       # 项目说明
```
# 阅读顺序
从Kinematics（认识仿真器）到Core_Constraints（线性化约束使用gjk进行碰撞检测，线性化轨迹平滑度目标函数）再到Core_TrajOpt（SCP求解框架）。

# 问题
当前没有对论文中连续时间的碰撞进行检测造成两个离散状态之间插值可能会出现碰撞或者两个状态之间直接穿过障碍物，在dev分支有未完成的连续碰撞检测。
<video src="./videos/离散trajopt.webm" controls=""></video>
<video src="./videos/离散trajopt1.webm" controls=""></video>
<!-- [两个离散状态之间插值可能会出现碰撞](videos/离散trajopt.webm)
[两个离散状态之间直接穿过障碍物](videos/离散trajopt.webm) -->
- 修复凸化时用到的变量
- 只对delta_x应用置信域
- A_ineq 类型为 None 判断