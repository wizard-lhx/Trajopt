# 01_Kinematics/state_definitions.py

# --- 状态和控制的维度 ---
STATE_DIM = 3     # 状态量: [x, y, theta]
CONTROL_DIM = 2   # 控制量: [vx, omega]

# --- 物理/仿真参数 ---
CAR_WIDTH = 0.5   # 车子的宽度 (m)
CAR_LENGTH = 1.0  # 车子的长度 (m)
TIME_STEP = 0.05  # 仿真步长 DT (s)