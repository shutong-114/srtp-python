#coding: utf-8
'''
0325更新: 添加凸多边形障碍物
0326更新: 修改凸多边形裁剪边选取逻辑, 并适配
0326_2更新：
1. 更正 FormationOptimizer 中初始位置参数接入（通过 anchor_3 库引用）
2. 增加小车间距限制并适配，同时进行图片中的锚点与有效宽度计算
3. 如果未达到目标点，将优化结果作为新目标插入路径集
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from a_star_lib_v_5 import a_star_path  # 导入A*算法库
from LF2_2 import safe_region  # 导入安全区域计算
from anchors_3 import FormationOptimizer  # 作为库引用

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#####################################
# 锚点与有效宽度计算辅助函数         #
#####################################
def is_point_in_polygon(point, polygon):
    """射线法判断点是否在任意多边形内（支持凹凸多边形）"""
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1) % n]
        if min(p1[1], p2[1]) < y <= max(p1[1], p2[1]):
            x_inters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
            if x < x_inters:
                inside = not inside
    return inside

def compute_effective_width(LF_vertices, center_point, n_vector, R, r):
    """
    计算沿前进方向两侧的有效区域最小宽度（垂直于 n_vector 方向）
    LF_vertices: 可行区域多边形顶点列表
    center_point: 编队中心 (x, y)
    n_vector: 前进方向向量
    R: 编队半径（用于扩展）
    r: 单车半径或车辆间距参考参数
    """
    n_vector = np.asarray(n_vector).flatten()
    center_point = np.asarray(center_point).flatten()
    # 归一化前进方向
    n_unit = n_vector[:2] / np.linalg.norm(n_vector[:2])
    # 与 n_vector 垂直的单位向量（左手法则）
    mu_vector = np.array([-n_unit[1], n_unit[0]])
    # 取中心点沿前进方向两侧偏移 -R 与 +R 的点，以及中心点本身
    points = [center_point - R * n_unit, center_point, center_point + R * n_unit]
    widths = []
    def ray_cast(p, direction, max_len=20.0, step=0.1):
        direction = np.asarray(direction).flatten()
        for t in np.arange(0, max_len + step, step):
            test_point = p + t * direction/np.linalg.norm(direction)
            if not is_point_in_polygon(test_point, LF_vertices):
                return max(0, t - step)
        return max_len
    for p in points:
        forward = ray_cast(p, mu_vector)
        backward = ray_cast(p, -mu_vector)
        widths.append(forward + backward)
    min_width = min(widths)
    print("有效宽度:", min_width)
    return min_width

class NavigationVectorCalculator:
    """
    计算航点（锚点）方向向量  
    根据当前航点及后续若干航点计算加权平均方向向量  
    """
    def __init__(self, waypoints, look_ahead=3):
        self.waypoints = waypoints
        self.look_ahead = look_ahead
        self.alpha_weights = [0.8, 0.5, 0.2]  # 递减权重

    def get_n_vector(self, current_idx):
        if current_idx >= len(self.waypoints)-1:
            return np.array([1, 0])
        current_center = np.mean(self.waypoints[current_idx], axis=0)
        vectors = []
        for i in range(1, self.look_ahead+1):
            if current_idx + i >= len(self.waypoints):
                break
            next_center = np.mean(self.waypoints[current_idx+i], axis=0)
            vec = next_center - current_center
            if np.linalg.norm(vec) == 0:
                continue
            vec_norm = vec / np.linalg.norm(vec)
            vectors.append(vec_norm * self.alpha_weights[i-1])
            current_center = next_center
        if not vectors:
            return np.array([1, 0])
        n_vector = np.sum(vectors, axis=0)
        return n_vector / np.linalg.norm(n_vector)

#####################################
# 优化问题求解封装（调用 FormationOptimizer） #
#####################################
def solve_optimization(LF_vertices, target_positions, initial_positions,
                       min_distance_threshold=0.5, alpha=1.0, beta=1.0):
    """
    利用 FormationOptimizer 进行优化计算，同时根据当前安全区域计算有效宽度并设置 Gamma 矩阵  
    target_positions: 4x2 数组（各车目标点）
    initial_positions: 4x2 数组（参考形状）
    """
    formation_icon = np.array(FORMATION_GEOMETRY)[:, :2]
    formation_radius = compute_formation_radius(formation_icon)
    vehicle_radius = max(p[2] for p in FORMATION_GEOMETRY)
    current_center = np.mean(initial_positions, axis=0)
    nav_calculator = NavigationVectorCalculator(MISSION_WAYPOINTS)
    n_vector = nav_calculator.get_n_vector(0)
    Weff = compute_effective_width(LF_vertices, current_center, n_vector, formation_radius, vehicle_radius)
    W_min = 4 * vehicle_radius
    W_max = formation_radius + 2 * vehicle_radius
    if Weff >= W_max:
        Gamma = np.eye(2)
    elif Weff <= W_min:
        Gamma = np.diag([3, 0.1])
    else:
        lambda2 = (Weff - W_min) / (W_max - W_min)
        lambda1 = 3 - 2 * lambda2
        Gamma = np.diag([lambda1, lambda2])
    optimizer = FormationOptimizer(formation_icon, sigma=0.2, lam=0.5, max_iter=500, tol=1e-6)
    relative_offset = np.array(initial_positions[2]) - np.array(initial_positions[0])
    result = optimizer.optimize(
        initial_positions=np.array(initial_positions),
        LF_vertices=LF_vertices,
        Gamma=Gamma,
        target_center=np.mean(target_positions, axis=0),
        free_anchor_idx1=0, free_anchor_idx2=2, relative_offset=relative_offset
    )
    return result if result is not None else np.array(initial_positions)

#####################################
# 车辆与编队控制器、辅助函数部分      #
#####################################
class Car:
    def __init__(self, id, init_pos, car_radius, max_speed=2.0):
        self.id = id
        self.position = np.array(init_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.radius = car_radius
        self.max_speed = max_speed
        self.arrival_threshold = 0.2
        self.forward_vector = (1, 0)

    def update_movement(self, target_pos, delta_t=0.1):
        direction = target_pos - self.position
        distance = np.linalg.norm(direction)
        if distance > self.arrival_threshold:
            self.velocity = direction / distance * self.max_speed
        else:
            self.velocity = np.zeros(2)
        self.position += self.velocity * delta_t
        return distance <= self.arrival_threshold

class FormationController:
    def __init__(self, vehicles, formation_offsets, waypoints, alpha_beta_groups, max_attempts=3):
        self.vehicles = vehicles
        self.formation_offsets = np.array(formation_offsets)
        self.waypoints = waypoints   # 每个航点为 4x2 数组
        self.current_waypoint_index = 0
        self.completed_centers = []
        self.safe_regions = []
        self.navigation_complete = False
        self.max_attempts = max_attempts
        self.attempts = 0
        # alpha-beta 参数仅用于初始传递（后续不动态更新）
        self.alpha_beta_groups = alpha_beta_groups
        self.current_group_index = 0
        self.alpha, self.beta = self.alpha_beta_groups[self.current_group_index]

        # 初始安全区域计算
        initial_center = np.mean([v.position for v in self.vehicles], axis=0)
        initial_car_info = [(initial_center[0] + offset[0],
                             initial_center[1] + offset[1],
                             offset[2]) for offset in FORMATION_GEOMETRY]
        initial_target = self.waypoints[0]  # 4x2 数组
        initial_target_center = np.mean(initial_target, axis=0)
        self.vehicles[0].forward_vector = (initial_target_center[0] - initial_center[0],
                                           initial_target_center[1] - initial_center[1])
        initial_safe = safe_region([initial_car_info, OBSTACLES],
                                   forward_vector=self.vehicles[0].forward_vector,
                                   formation_erosion=False)
        self.safe_regions.append(initial_safe)
        # 使用优化器计算起始目标点
        formation_icon = np.array(FORMATION_GEOMETRY)[:, :2]
        formation_radius = compute_formation_radius(formation_icon)
        vehicle_radius = max(p[2] for p in FORMATION_GEOMETRY)
        Gamma = compute_gamma_matrix(initial_safe, formation_radius, vehicle_radius)
        relative_offset = np.array(self.waypoints[0][2]) - np.array(self.waypoints[0][0])
        computed_target = solve_optimization(initial_safe,
                                             self.waypoints[0],
                                             np.array([info[:2] for info in FORMATION_GEOMETRY]),
                                             min_distance_threshold=MIN_DISTANCE,
                                             alpha=self.alpha, beta=self.beta)
        if computed_target is not None and np.linalg.norm(np.mean(computed_target, axis=0) - initial_target_center) > 1e-1:
            self.waypoints[0] = computed_target
            print("初始目标更新为：", np.mean(computed_target, axis=0))

    def update_formation(self, delta_t=0.1):
        if self.navigation_complete:
            return
        current_positions = [v.position for v in self.vehicles]
        current_center = np.mean(current_positions, axis=0)
        current_target = np.mean(self.waypoints[self.current_waypoint_index], axis=0)
        distance = np.linalg.norm(current_center - current_target)
        
        # 如果未到达当前目标，则尝试利用优化结果作为新的中间目标插入路径集
        if distance >= self.vehicles[0].arrival_threshold:
            current_car_info = [(current_center[0] + offset[0],
                                 current_center[1] + offset[1],
                                 offset[2]) for offset in FORMATION_GEOMETRY]
            new_safe = safe_region([current_car_info, OBSTACLES],
                                   forward_vector=self.vehicles[0].forward_vector,
                                   formation_erosion=False)
            formation_icon = np.array(FORMATION_GEOMETRY)[:, :2]
            formation_radius = compute_formation_radius(formation_icon)
            vehicle_radius = max(p[2] for p in FORMATION_GEOMETRY)
            Gamma = compute_gamma_matrix(new_safe, formation_radius, vehicle_radius)
            relative_offset = np.array(self.waypoints[self.current_waypoint_index][2]) - np.array(self.waypoints[self.current_waypoint_index][0])
            computed_target = solve_optimization(new_safe,
                                                 self.waypoints[self.current_waypoint_index],
                                                 np.array([info[:2] for info in FORMATION_GEOMETRY]),
                                                 min_distance_threshold=MIN_DISTANCE,
                                                 alpha=self.alpha, beta=self.beta)
            if computed_target is not None:
                computed_center = np.mean(computed_target, axis=0)
                current_target_center = np.mean(self.waypoints[self.current_waypoint_index], axis=0)
                # 当优化结果与原目标有明显偏差时，将其作为新目标插入路径集（阈值可调，示例中设为 0.1）
                if np.linalg.norm(computed_center - current_target_center) > 0.1:
                    print("插入新目标：", computed_center)
                    self.waypoints.insert(self.current_waypoint_index, computed_target)
                    # 重新设置当前目标为新插入的目标
                    current_target = np.mean(self.waypoints[self.current_waypoint_index], axis=0)
                    distance = np.linalg.norm(current_center - current_target)
        
        # 如果到达目标点，则更新为下一个目标
        if distance < self.vehicles[0].arrival_threshold:
            self.completed_centers.append(current_center)
            new_car_info = [(current_center[0] + offset[0],
                             current_center[1] + offset[1],
                             offset[2]) for offset in FORMATION_GEOMETRY]
            if self.current_waypoint_index < len(self.waypoints) - 1:
                next_target = self.waypoints[self.current_waypoint_index + 1]
                next_center = np.mean(next_target, axis=0)
                self.vehicles[0].forward_vector = (next_center[0] - current_center[0],
                                                   next_center[1] - current_center[1])
                new_safe = safe_region([new_car_info, OBSTACLES],
                                       forward_vector=self.vehicles[0].forward_vector,
                                       formation_erosion=False)
                self.safe_regions.append(new_safe)
                if len(self.safe_regions) > 1:
                    self.safe_regions.pop(0)
                formation_icon = np.array(FORMATION_GEOMETRY)[:, :2]
                formation_radius = compute_formation_radius(formation_icon)
                vehicle_radius = max(p[2] for p in FORMATION_GEOMETRY)
                Gamma = compute_gamma_matrix(new_safe, formation_radius, vehicle_radius)
                relative_offset = np.array(next_target[2]) - np.array(next_target[0])
                computed_target = solve_optimization(new_safe,
                                                     next_target,
                                                     np.array([info[:2] for info in FORMATION_GEOMETRY]),
                                                     min_distance_threshold=MIN_DISTANCE,
                                                     alpha=self.alpha, beta=self.beta)
                if computed_target is not None:
                    self.waypoints[self.current_waypoint_index + 1] = computed_target
            else:
                new_safe = safe_region([new_car_info, OBSTACLES],
                                       forward_vector=self.vehicles[0].forward_vector,
                                       formation_erosion=False)
                self.safe_regions.append(new_safe)
                if len(self.safe_regions) > 1:
                    self.safe_regions.pop(0)
                print("导航完成。")
                self.navigation_complete = True
            self.current_waypoint_index += 1

        # 更新各车辆的目标位置
        for i, vehicle in enumerate(self.vehicles):
            target = self.waypoints[self.current_waypoint_index][i]
            vehicle.update_movement(target, delta_t)

def compute_formation_radius(car_points):
    center = np.mean(car_points, axis=0)[:2]
    return max(np.linalg.norm(np.array(p[:2]) - center) for p in car_points)

def compute_gamma_matrix(LF_vertices, R, r):
    W_min = 4 * r
    W_max = R + 2 * r
    if LF_vertices is None:
        return np.eye(2)
    if R >= W_max:
        return np.eye(2)
    elif R <= W_min:
        return np.diag([3, 0.1])
    else:
        lambda2 = (R - W_min) / (W_max - W_min)
        lambda1 = 3 - 2 * lambda2
        return np.diag([lambda1, lambda2])

def animation_update(frame):
    if not controller.navigation_complete:
        controller.update_formation(delta_t=0.1)
    for plot, vehicle in zip(vehicle_plots, vehicles):
        plot.set_data([vehicle.position[0]], [vehicle.position[1]])
    status_text.set_text(f"当前任务航点: {controller.current_waypoint_index + 1}/{len(controller.waypoints)}")
    completed_centers_x = [center[0] for center in controller.completed_centers]
    completed_centers_y = [center[1] for center in controller.completed_centers]
    completed_centers_plot.set_data(completed_centers_x, completed_centers_y)
    if controller.safe_regions:
        safe_region_patch.set_xy(np.array(controller.safe_regions[-1]))
    else:
        safe_region_patch.set_xy(np.empty((0,2)))
    if controller.navigation_complete:
        completion_text.set_text("导航完成")
        completion_text.set_color("red")
        completion_text.set_fontsize(16)
    else:
        completion_text.set_text("")
    return vehicle_plots + [status_text, completed_centers_plot, safe_region_patch, completion_text]

def expand_to_formation(center, formation_offsets):
    return np.array([(center[0] + offset[0], center[1] + offset[1]) for offset in formation_offsets])

def get_formation_center(formation):
    return np.mean(formation, axis=0)

# 邻接矩阵（暂未使用，可扩展）
adj_matrix = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

#####################################
# 主程序入口部分                   #
#####################################
if __name__ == "__main__":
    # 编队相对偏移量（保持中心对称）
    FORMATION_GEOMETRY = [
        (-0.5, -0.5, 0.1),  # 左上
        (0.5, -0.5, 0.1),   # 右上
        (-0.5, 0.5, 0.1),   # 左下
        (0.5, 0.5, 0.1)     # 右下
    ]
    # 障碍物信息 (x, y, r)
    OBSTACLES = [(10, 4, 4), (10, 16, 4), (5, 12, 2), (4, 9, 1)]
    formation_radius = compute_formation_radius(FORMATION_GEOMETRY)
    vehicle_radius = 0
    MIN_DISTANCE = 0.1
    start_point = (1, 19)
    goal_point = (18, 18)
    num_targets = 7  # 中间目标点数量

    MISSION_WAYPOINTS = a_star_path(start_point, goal_point, num_targets, OBSTACLES, 0.2, vehicle_radius)
    if not MISSION_WAYPOINTS:
        MISSION_WAYPOINTS = [start_point, goal_point]
    else:
        MISSION_WAYPOINTS.append(goal_point)
    MISSION_WAYPOINTS = [expand_to_formation(waypoint, FORMATION_GEOMETRY) for waypoint in MISSION_WAYPOINTS]

    vehicles = [
        Car(id=i,
            init_pos=(start_point[0] + FORMATION_GEOMETRY[i][0],
                      start_point[1] + FORMATION_GEOMETRY[i][1]),
            car_radius=FORMATION_GEOMETRY[i][2]
        ) for i in range(4)
    ]
    # 定义 alpha-beta 参数组（仅用于初始参数传递）
    alpha_beta_groups = [
        (10000, 10000),
        (20, 20),
        (0.2, 0.2),
        (0, 0)
    ]
    controller = FormationController(
        vehicles=vehicles,
        formation_offsets=FORMATION_GEOMETRY,
        waypoints=MISSION_WAYPOINTS,
        alpha_beta_groups=alpha_beta_groups,
        max_attempts=4
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 20), ylim=(0, 20))
    ax.set_title("基于A*路径的编队导航")
    waypoints_centers = [get_formation_center(waypoint) for waypoint in MISSION_WAYPOINTS[:-1]]
    waypoints_x, waypoints_y = zip(*waypoints_centers)
    ax.scatter(waypoints_x, waypoints_y, marker='X', s=200, c='red', label='中间点')
    start_x, start_y = start_point
    ax.scatter(start_x, start_y, marker='*', s=200, c='purple', label='起点')
    goal_x, goal_y = goal_point
    ax.scatter(goal_x, goal_y, marker='*', s=200, c='red', label='终点')
    obs_label_drawn = False
    for obs in OBSTACLES:
        if isinstance(obs, tuple) and len(obs) == 3:
            center = (obs[0], obs[1])
            circle = plt.Circle(center, obs[2], color='red', fill=False, linestyle='--',
                                label='障碍物' if not obs_label_drawn else None)
            ax.add_patch(circle)
            obs_label_drawn = True
        elif isinstance(obs, list):
            poly = obs + [obs[0]]
            ax.plot([p[0] for p in poly], [p[1] for p in poly], 'r--',
                    label='障碍物' if not obs_label_drawn else None)
            obs_label_drawn = True
    vehicle_plots = [ax.plot([], [], 'o', markersize=8, color='blue')[0] for _ in vehicles]
    vehicle_plots[0].set_label('小车')
    completed_centers_plot, = ax.plot([], [], 'o', markersize=5, color='black', label='已完成中心点')
    safe_region_patch = plt.Polygon(np.array(controller.safe_regions[-1]), color='green', alpha=0.5, label='安全区域')
    ax.add_patch(safe_region_patch)
    status_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center')
    completion_text = ax.text(0.5, 0.90, '', transform=ax.transAxes, ha='center')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1), borderaxespad=0.)
    ani = FuncAnimation(fig, animation_update, frames=200, interval=50, blit=True)
    plt.show()
