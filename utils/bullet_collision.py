"""
使用 PyBullet 进行碰撞检测的工具函数
"""

import numpy as np
import time
import pybullet as p
from typing import Tuple, Optional
from Kinematics.state_definitions import CAR_LENGTH, CAR_WIDTH, CAR_HEIGHT

class BulletCollisionChecker:
    """
    PyBullet 碰撞检测器
    用于 TrajOpt 中计算带符号距离和梯度
    """
    
    def __init__(self, use_gui: bool = False):
        """
        初始化 PyBullet 物理引擎（用于碰撞检测）
        
        参数:
            use_gui: 是否显示GUI（通常设为False用于后台计算）
        """
        # 连接到物理引擎
        if use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)  # 无GUI模式，更快
        
        # 存储创建的物体ID
        self.car_id = None
        self.obstacle_ids = {}
        self.swept_volume_ids = []  # 存储临时创建的swept volume多面体
        
        print(f"PyBullet collision checker initialized (client={self.physics_client})")
    
    def create_car_body(self, car_length: float = CAR_LENGTH, car_width: float = CAR_WIDTH):
        """
        创建车辆碰撞体
        """
        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[car_length/2, car_width/2, CAR_HEIGHT/2],
            physicsClientId=self.physics_client
        )
        
        self.car_id = p.createMultiBody(
            baseMass=0,  # 质量为0表示不受重力影响
            baseCollisionShapeIndex=col_shape,
            basePosition=[0, 0, 0.05],
            physicsClientId=self.physics_client
        )
        
        return self.car_id
    
    def create_obstacle(self, obs_id: int, cx: float, cy: float, 
                       width: float, length: float, theta: float):
        """
        创建障碍物碰撞体
        
        参数:
            obs_id: 障碍物唯一ID
            cx, cy: 障碍物中心坐标
            width, length: 宽度和长度
            theta: 旋转角度（弧度）
        """
        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length/2, width/2, 3],
            physicsClientId=self.physics_client
        )
        
        orn = p.getQuaternionFromEuler([0, 0, theta])
        
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            basePosition=[cx, cy, 0],
            baseOrientation=orn,
            physicsClientId=self.physics_client
        )
        
        self.obstacle_ids[obs_id] = body_id
        return body_id
    
    def update_car_pose(self, car_state: np.ndarray):
        """
        更新车辆位置和姿态
        
        参数:
            car_state: [x, y, theta]
        """
        if self.car_id is None:
            raise ValueError("Car body not created. Call create_car_body() first.")
        
        x, y, theta = car_state
        orn = p.getQuaternionFromEuler([0, 0, theta])
        
        p.resetBasePositionAndOrientation(
            self.car_id,
            [x, y, CAR_HEIGHT/2],
            orn,
            physicsClientId=self.physics_client
        )
    
    def compute_signed_distance(
        self, 
        car_state: np.ndarray, 
        obs_id: int,
        max_distance: float = 1.0
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        使用 PyBullet 计算带符号距离和接触信息
        
        参数:
            car_state: [x, y, theta] 车辆状态
            obs_id: 障碍物ID
            max_distance: 最大检测距离
        
        返回:
            sd: 带符号距离（正数=分离，负数=穿透）
            n_hat: 接触法向量（从障碍物指向车辆）
            pA_local: 车辆上接触点（局部坐标）
            pB_world: 障碍物上接触点（世界坐标）
        """
        # 更新车辆位置
        self.update_car_pose(car_state)
        
        # 获取障碍物ID
        if obs_id not in self.obstacle_ids:
            raise ValueError(f"Obstacle {obs_id} not found")
        
        obstacle_body_id = self.obstacle_ids[obs_id]
        
        # 使用 PyBullet 的最近点查询
        closest_points = p.getClosestPoints(
            bodyA=self.car_id,
            bodyB=obstacle_body_id,
            distance=max_distance,
            physicsClientId=self.physics_client
        )
        
        if len(closest_points) == 0:
            # 距离超过 max_distance，返回安全值
            return max_distance, np.zeros(2), np.zeros(2), np.zeros(2)
        
        # 取第一个接触点（通常只有一个）
        contact = closest_points[0]
        
        # 提取信息
        # contact[8] = distance (负数表示穿透)
        # contact[5] = positionOnA (车辆上的点，世界坐标)
        # contact[6] = positionOnB (障碍物上的点，世界坐标)
        # contact[7] = contactNormalOnB (从B指向A的法向量)
        
        sd = contact[8]  # 带符号距离
        pA_world = np.array(contact[5][:2])  # 只取x,y
        pB_world = np.array(contact[6][:2])
        n_hat = np.array(contact[7][:2])  # 法向量（从B指向A）
        
        # 归一化法向量
        n_norm = np.linalg.norm(n_hat)
        if n_norm > 1e-10:
            n_hat = n_hat / n_norm
        else:
            # 如果法向量为零，使用连线方向
            if sd >= 0:
                diff = pA_world - pB_world
                diff_norm = np.linalg.norm(diff)
                if diff_norm > 1e-10:
                    n_hat = diff / diff_norm
                else:
                    n_hat = np.array([1.0, 0.0])
            else:
                n_hat = np.array([1.0, 0.0])
        
        # 计算 pA 在车辆局部坐标系下的位置
        x, y, theta = car_state
        R_inv = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
        ])
        pA_local = R_inv @ (pA_world - car_state[:2])
        
        return sd, n_hat, pA_local, pB_world
    
    def get_car_vertices(self, car_state: np.ndarray) -> np.ndarray:
        """
        获取车辆在给定状态下的四个顶点坐标（世界坐标系）
        
        参数:
            car_state: [x, y, theta] 车辆状态
        
        返回:
            vertices: (4, 2) 四个顶点的世界坐标
        """
        x, y, theta = car_state
        
        # 车辆局部坐标系下的四个顶点（矩形）
        half_length = CAR_LENGTH / 2
        half_width = CAR_WIDTH / 2
        local_vertices = np.array([
            [half_length, half_width],    # 右前
            [half_length, -half_width],   # 右后
            [-half_length, -half_width],  # 左后
            [-half_length, half_width]    # 左前
        ])
        
        # 旋转矩阵
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        # 转换到世界坐标系
        world_vertices = (R @ local_vertices.T).T + car_state[:2]
        
        return world_vertices
    
    def support_map(self, car_state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        计算车辆在给定方向上的支撑顶点（support vertex）
        
        支撑映射定义：s_A(d) = argmax_{v ∈ A} d^T v
        即找到在方向d上投影最大的顶点
        
        参数:
            car_state: [x, y, theta] 车辆状态
            direction: (2,) 搜索方向向量（不需要归一化）
        
        返回:
            support_vertex: (2,) 支撑顶点的世界坐标
        """
        # 获取车辆的所有顶点
        vertices = self.get_car_vertices(car_state)
        
        # 计算每个顶点在方向上的投影
        projections = vertices @ direction
        
        # 找到投影最大的顶点
        max_idx = np.argmax(projections)
        support_vertex = vertices[max_idx]
        
        return support_vertex
    
    def create_swept_volume_convex_hull(
        self,
        car_state_t0: np.ndarray,
        car_state_t1: np.ndarray
    ) -> int:
        """
        创建两个车辆状态之间的swept volume凸包多面体
        
        参数:
            car_state_t0: 时刻t的车辆状态 [x, y, theta]
            car_state_t1: 时刻t+1的车辆状态 [x, y, theta]
        
        返回:
            swept_body_id: 创建的swept volume多面体的PyBullet ID
        """
        # 获取两个状态下车辆的顶点
        vertices_t0 = self.get_car_vertices(car_state_t0)
        vertices_t1 = self.get_car_vertices(car_state_t1)
        
        # 合并所有顶点（8个顶点：4个来自t0，4个来自t1）
        # 添加z坐标（在2D平面上，z=0和z=CAR_HEIGHT）
        vertices_3d = []
        for v in vertices_t0:
            vertices_3d.append([v[0], v[1], 0])
            vertices_3d.append([v[0], v[1], CAR_HEIGHT])
        for v in vertices_t1:
            vertices_3d.append([v[0], v[1], 0])
            vertices_3d.append([v[0], v[1], CAR_HEIGHT])
        
        vertices_3d = np.array(vertices_3d)
        
        # 使用PyBullet创建凸包碰撞体
        col_shape = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=vertices_3d,
            physicsClientId=self.physics_client
        )
        
        swept_body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        
        self.swept_volume_ids.append(swept_body_id)
        
        return swept_body_id
    
    def compute_swept_volume_distance(
        self,
        car_state_t0: np.ndarray,
        car_state_t1: np.ndarray,
        obs_id: int,
        max_distance: float = 1.0
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        计算两个车辆状态之间swept volume凸包与障碍物的最近距离
        
        参数:
            car_state_t0: 时刻t的车辆状态 [x, y, theta]
            car_state_t1: 时刻t+1的车辆状态 [x, y, theta]
            obs_id: 障碍物ID
            max_distance: 最大检测距离
        
        返回:
            sd_min: 最小带符号距离
            n_hat: 接触法向量
            pA_local_t0: 接触点在t0车辆局部坐标系下的位置
            pB_world: 障碍物上接触点
            alpha: 最近点的估计插值参数（用于梯度加权）
        """
        # 创建swept volume凸包
        swept_body_id = self.create_swept_volume_convex_hull(car_state_t0, car_state_t1)
        
        # 获取障碍物ID
        if obs_id not in self.obstacle_ids:
            raise ValueError(f"Obstacle {obs_id} not found")
        
        obstacle_body_id = self.obstacle_ids[obs_id]
        
        # 查询swept volume与障碍物的最近点
        closest_points = p.getClosestPoints(
            bodyA=swept_body_id,
            bodyB=obstacle_body_id,
            distance=max_distance,
            physicsClientId=self.physics_client
        )
        
        # 清理临时创建的swept volume
        p.removeBody(swept_body_id, physicsClientId=self.physics_client)
        self.swept_volume_ids.remove(swept_body_id)
        
        if len(closest_points) == 0:
            # 距离超过max_distance
            return max_distance, np.zeros(2), np.zeros(2), np.zeros(2), 0.5
        
        contact = closest_points[0]
        
        sd = contact[8]
        pA_world = np.array(contact[5][:2])  # swept volume上的接触点
        pB_world = np.array(contact[6][:2])
        n_hat = np.array(contact[7][:2])
        
        # 归一化法向量
        n_norm = np.linalg.norm(n_hat)
        if n_norm > 1e-10:
            n_hat = n_hat / n_norm
        else:
            diff = pA_world - pB_world
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                n_hat = diff / diff_norm
            else:
                n_hat = np.array([1.0, 0.0])
        
        # 根据TrajOpt论文：使用支撑映射在-n方向找到p0和p1
        # p0 = support_A(t)(-n)  在时刻t的车辆上，沿-n方向的支撑顶点
        # p1 = support_A(t+1)(-n) 在时刻t+1的车辆上，沿-n方向的支撑顶点
        minus_n = -n_hat
        p0 = self.support_map(car_state_t0, minus_n)
        p1 = self.support_map(car_state_t1, minus_n)
        
        # 计算插值参数alpha
        # 接触点pA_world应该在p0和p1之间，计算其位置参数
        # pA ≈ α*p0 + (1-α)*p1
        dist_to_t0 = np.linalg.norm(pA_world - p0)
        dist_to_t1 = np.linalg.norm(pA_world - p1)
        total_dist = dist_to_t0 + dist_to_t1
        
        if total_dist > 1e-6:
            alpha = dist_to_t1 / total_dist
        else:
            # p0和p1重合，使用0.5
            alpha = 0.5
        
        # 将接触点转换到t0局部坐标系
        x0, y0, theta0 = car_state_t0
        R_inv_t0 = np.array([
            [np.cos(-theta0), -np.sin(-theta0)],
            [np.sin(-theta0),  np.cos(-theta0)]
        ])
        pA_local_t0 = R_inv_t0 @ (pA_world - car_state_t0[:2])
        
        return sd, n_hat, pA_local_t0, pB_world, alpha
    
    def cleanup(self):
        """
        清理资源
        """
        # 清理所有临时swept volume
        for swept_id in self.swept_volume_ids:
            p.removeBody(swept_id, physicsClientId=self.physics_client)
        self.swept_volume_ids.clear()
        
        # time.sleep(60)  # 测试时使用，观察GUI
        if self.physics_client >= 0:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = -1


def test_bullet_collision():
    """
    测试 PyBullet 碰撞检测
    """
    print("=" * 70)
    print("测试 PyBullet 碰撞检测")
    print("=" * 70)
    
    # 创建碰撞检测器
    checker = BulletCollisionChecker(use_gui=True)
    
    # 创建车辆
    checker.create_car_body(car_length=1, car_width=0.5)
    
    # 创建障碍物（中心在(1, 0)，2x1大小）
    checker.create_obstacle(
        obs_id=0,
        cx=1.0,
        cy=0.0,
        width=1.0,
        length=2.0,
        theta=0.0
    )
    
    # 测试1: 车辆在原点（部分重叠）
    print("\n测试1: 车辆在原点")
    car_state1 = np.array([0.0, 0.0, 0.05])
    sd1, n_hat1, pA_local1, pB_world1 = checker.compute_signed_distance(car_state1, obs_id=0)
    print(f"  Signed Distance: {sd1:.4f}")
    print(f"  Normal: {n_hat1}")
    print(f"  pA_local: {pA_local1}")
    print(f"  pB_world: {pB_world1}")
    
    # 测试2: 车辆与障碍物重叠
    print("\n测试2: 车辆在(0.5, 0, 0) - 应该碰撞")
    car_state2 = np.array([0.5, 0.0, 0.05])
    sd2, n_hat2, pA_local2, pB_world2 = checker.compute_signed_distance(car_state2, obs_id=0)
    print(f"  Signed Distance: {sd2:.4f}")
    print(f"  Normal: {n_hat2}")
    print(f"  pA_local: {pA_local2}")
    print(f"  pB_world: {pB_world2}")
    
    # 测试3: 车辆远离障碍物
    print("\n测试3: 车辆在(5, 0, 0) - 应该分离")
    car_state3 = np.array([5.0, 0.0, 0.05])
    sd3, n_hat3, pA_local3, pB_world3 = checker.compute_signed_distance(car_state3, obs_id=0)
    print(f"  Signed Distance: {sd3:.4f}")
    print(f"  Normal: {n_hat3}")
    
    # 清理
    checker.cleanup()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_bullet_collision()
