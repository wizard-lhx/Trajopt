import pybullet as p
import pybullet_data
import time
import math

# 1. 初始化 PyBullet
p.connect(p.GUI)  # 使用 p.DIRECT 可以不显示窗口，用于后台训练
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 2. 加载地面
plane_id = p.loadURDF("plane.urdf")

# 3. 创建简单的车子模型 (这里用一个长方体代替)
# 定义碰撞形状 (半长, 半宽, 半高)
col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.05])
# 定义可视化形状 (蓝色)
vis_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.05], rgbaColor=[0, 0, 1, 1])

# 创建多体 (质量设为 1kg)
# basePosition 稍微抬高一点以免卡进地面
car_id = p.createMultiBody(baseMass=1,
                           baseCollisionShapeIndex=col_shape_id,
                           baseVisualShapeIndex=vis_shape_id,
                           basePosition=[0, 0, 0.05])

# --- 关键步骤：实现 2D 约束 ---
# 我们需要创建一个约束，让车子永远不会翻倒 (Roll/Pitch = 0)，且 Z 轴高度恒定
# 这里的 cid 创建了一个 "Prismatic" 类型的约束，或者是 Generic 6DOF constraint 更好
# 但为了极其简单，我们可以每一帧通过代码强制重置姿态（Kinematic模式），
# 或者只依赖物理引擎的平面支撑 + 忽略 Z 轴力。

# 这里演示最稳健的方法：只在算法输入端控制，物理端靠重力压在平面上。
# 为了防止翻车（因为是简单的 Box），我们可以把摩擦力改小，或者加宽底盘。
p.changeDynamics(car_id, -1, lateralFriction=0.5)

print("Start Simulation. Use Arrow Keys to move.")
print("Up/Down: Linear Velocity")
print("Left/Right: Angular Velocity (Yaw)")

# 模拟的主循环
t = 0
dt = 1./240.
while p.isConnected():
    
    # 4. 获取控制输入 (模拟你的算法输出)
    # 假设你的算法输出了线速度 v 和角速度 omega
    keys = p.getKeyboardEvents()
    
    target_v = 0
    target_omega = 0
    
    # 简单的键盘控制逻辑作为示例
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        target_v = 1.0
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        target_v = -1.0
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        target_omega = 1.0
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        target_omega = -1.0

    # 5. 将控制量应用到模型
    # 获取当前车子的姿态
    pos, orn = p.getBasePositionAndOrientation(car_id)
    # 四元数转欧拉角 [roll, pitch, yaw]
    euler = p.getEulerFromQuaternion(orn)
    current_yaw = euler[2]

    # 计算全局坐标系下的速度分量
    # v_x_global = v_linear * cos(theta)
    # v_y_global = v_linear * sin(theta)
    vx = target_v * math.cos(current_yaw)
    vy = target_v * math.sin(current_yaw)

    # 直接设置底座速度 (Kinematic Control)
    # 格式: resetBaseVelocity(object, linearVelocity=[vx, vy, vz], angularVelocity=[wx, wy, wz])
    # 强制 vz=0, wx=0, wy=0 实现了完美的 2D 约束
    p.resetBaseVelocity(car_id, [vx, vy, 0], [0, 0, target_omega])

    # 6. 步进仿真
    p.stepSimulation()
    
    # 保持相机跟随 (可选)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=math.degrees(current_yaw)-90, cameraPitch=-45, cameraTargetPosition=pos)
    
    time.sleep(dt)