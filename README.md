```
TrajOpt_2D_Car_Project/
├── 01_Kinematics/
│   ├── car_model.py                # 2D 小车运动学模型 (Unicycle model)
│   ├── visualization.py            # 动态可视化函数 (您上一步创建的)
│   └── state_definitions.py        # 定义 State (x, y, theta) 和 Control (vx, omega)
├── 02_Core_TrajOpt/
│   ├── sequential_convex_opt.py    # 实现 Algorithm 1 的主循环 (SCO 框架)
│   ├── qp_solver_interface.py      # 封装对 QP 求解器的调用 (如 Gurobi/cvxpy)
│   └── penalty_utilities.py        # 实现 L1 惩罚项到线性约束的转换 (松弛变量)
├── 03_Costs_Constraints/
│   ├── trajectory_cost.py          # 目标函数：最小路径长度 (Σ||x_{t+1}-x_t||^2)
│   ├── no_collision_cost.py        # 核心：实现碰撞检测的线性化和惩罚
│   ├── kinematics_constraint.py    # 实现运动学等式约束的线性化
│   └── trust_region.py             # 实现步长评估和信赖域更新逻辑
├── 04_Environment/
│   └── obstacles.py                # 2D 障碍物定义 (多边形/圆形，用于碰撞检测)
├── 05_Experiments/
│   ├── experiment_1_simple_bend.py # 简单避障场景 (一个障碍物)
│   └── experiment_2_corridor.py    # 复杂场景 (通道/多个障碍物)
├── results/
│   ├── saved_trajectories/         # 存储成功的 (x, y, theta) 轨迹数据
│   └── plots/                      # 存储最终的轨迹对比图
├── README.md                       # 项目说明、依赖安装指南
└── requirements.txt                # Python 依赖列表 (numpy, matplotlib, QP solver, etc.)
```
# 问题
无法实现2D的碰撞检测，所以先放弃了。