import numpy as np
from scipy.spatial import Voronoi

def compute_RVO_velocity(current_pos, current_vel, radius, neighbors_pos, neighbors_vel, neighbors_radius, max_speed, time_horizon=2.0):
    """
    计算RVO避碰后的速度
    参数：
        current_pos: 当前小车位置 (2D向量)
        current_vel: 当前小车速度 (2D向量)
        radius: 当前小车半径
        neighbors_pos: 周围小车位置列表 [array(x,y),...]
        neighbors_vel: 周围小车速度列表 [array(vx,vy),...]
        neighbors_radius: 周围小车半径列表 [float,...]
        max_speed: 最大速度
        time_horizon: 避碰时间窗口
    返回：
        安全速度 (2D向量)
    """
    preferred_vel = current_vel.copy()
    candidate_vels = []

    # 生成候选速度（极坐标采样）
    for theta in np.linspace(0, 2*np.pi, 20):
        for r in np.linspace(0, max_speed, 5):
            candidate_vels.append(np.array([r*np.cos(theta), r*np.sin(theta)]))
    
    # 添加当前速度方向加强采样
    if np.linalg.norm(preferred_vel) > 0:
        dir = preferred_vel / np.linalg.norm(preferred_vel)
        for r in np.linspace(0, max_speed, 5):
            candidate_vels.append(dir * r)

    # RVO速度筛选
    safe_vels = []
    for v in candidate_vels:
        collision = False
        for n_pos, n_vel, n_radius in zip(neighbors_pos, neighbors_vel, neighbors_radius):
            # 计算相对参数
            relative_pos = n_pos - current_pos
            combined_radius = radius + n_radius
            relative_vel = v - n_vel
            
            # 计算碰撞时间
            time_to_collision = np.dot(relative_pos, relative_vel) / (np.linalg.norm(relative_vel)**2 + 1e-5)
            
            # 检查是否在危险距离内
            if time_to_collision > 0 and time_to_collision < time_horizon:
                dist = np.linalg.norm(relative_pos - relative_vel*time_to_collision)
                if dist < combined_radius:
                    collision = True
                    break
        if not collision:
            safe_vels.append(v)
    
    # 选择最优速度
    if safe_vels:
        # 选择最接近预设速度的
        norms = [np.linalg.norm(v - preferred_vel) for v in safe_vels]
        return safe_vels[np.argmin(norms)]
    else:
        # 无安全速度时保持原速（实际应更复杂处理）
        return current_vel * 0.8  # 减速