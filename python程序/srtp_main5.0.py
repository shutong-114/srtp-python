#coding: utf-8
'''
0325更新: 添加凸多边形障碍物
0326更新: 修改凸多边形裁剪边选取逻辑, 并适配
0326_2更新：
1.更正solve_optimization 的 initial_position 参数接入
2.增加小车间距限制并适配
3. 当优化目标与当前目标差距大于一定阈值后，将优化目标插入目标列表，作为新目标，否则目标序号加一进入新目标
1131更新：新接入分布式优化算子anchors_4(带锚点版)
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from anchors_4 import DistributedFormationOptimizer
from a_star_lib_v_5 import a_star_path  # 导入A*算法库
from LF2_2 import safe_region  # 导入安全区域计算
from rvo import compute_RVO_velocity  # 导入RVO避碰方法
from scipy.spatial import ConvexHull

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------
# 小车类
# ----------------------------------
class Car:
    
    def __init__(self, id, init_pos, car_radius, max_speed=2.0):
        self.id = id
        self.position = np.array(init_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.radius = car_radius
        self.max_speed = max_speed
        self.arrival_threshold = 0.2
        self.forward_vector = (1,0)

    def update_movement(self, target_pos, all_cars, delta_t=0.1):
        """更新移动（加入RVO避碰）"""
        # 收集周围车辆信息（排除自己）
        neighbors_pos = []
        neighbors_vel = []
        neighbors_radius = []
        for car in all_cars:
            if car.id != self.id:
                neighbors_pos.append(car.position)
                neighbors_vel.append(car.velocity)
                neighbors_radius.append(car.radius)
        # 计算理想速度
        direction = target_pos - self.position
        distance = np.linalg.norm(direction)
        if distance > self.arrival_threshold:
            preferred_vel = direction / distance * self.max_speed
        else:
            preferred_vel = np.zeros(2)
        # RVO避碰计算
        new_vel = compute_RVO_velocity(
            self.position, preferred_vel,
            self.radius, 
            neighbors_pos, neighbors_vel, neighbors_radius,
            self.max_speed
        )

        # 更新速度和位置
        self.velocity = new_vel
        self.position += self.velocity * delta_t
        return np.linalg.norm(self.position - target_pos) <= self.arrival_threshold

# ----------------------------------
# 点在多边形内的判断（利用凸包边界的线性约束）
# ----------------------------------
def is_point_in_polygon(point, polygon):
    """
    利用凸包边界的线性约束判断点是否在多边形内（支持凹凸多边形）。
    算法：构造 polygon 的凸包，对凸包中每个平面方程（a, b, c），检查是否有 a*x+b*y+c <= 0。
    如果所有方程均满足，则认为该点在区域内（或在边界上）。
    """
    polygon = np.array(polygon)
    hull = ConvexHull(polygon)
    for eq in hull.equations:
        a, b, c = eq
        if a * point[0] + b * point[1] + c > 1e-9:  # 允许一定数值容差
            return False
    return True

# ----------------------------------
# 计算沿前进方向两侧的有效宽度
# ----------------------------------
def compute_effective_width(LF_vertices, center_point, n_vector, R, r):
    """计算沿前进方向两侧的有效区域最小宽度（垂直于n_vector方向）"""
    n_vector = np.asarray(n_vector).flatten()
    center_point = np.asarray(center_point).flatten()
    # 归一化方向向量
    n_unit = n_vector[:2] / np.linalg.norm(n_vector[:2])
    mu_vector = np.array([-n_unit[1], n_unit[0]])  # 与 n_vector 垂直的单位向量（左手）
    
    def ray_cast(p, direction, max_len=20.0, step=0.1):
        """从点 p 沿 direction 方向前进，计算落出多边形的最大长度"""
        direction = direction / np.linalg.norm(direction)
        direction = np.asarray(direction).flatten()
        for t in np.arange(0, max_len + step, step):
            test_point = p + t * direction 
            #print(test_point)
            if not is_point_in_polygon(test_point, LF_vertices):
                return max(step, t - step)
        return max_len
    
    forward = ray_cast(center_point, mu_vector)
    backward = ray_cast(center_point, -mu_vector)
    width = forward + backward
    #print(width)
    return width

# ----------------------------------
# 根据有效宽度计算拉伸矩阵 Gamma
# ----------------------------------
def compute_gamma_matrix(Weff, R, r):
    """动态计算拉伸矩阵"""
    W_min = 4 * r
    W_max = R + 2 * r
    
    if Weff >= W_max:
        return np.eye(2)
    elif Weff <= W_min:
        return np.diag([3, 0.1])
    else:
        lambda2 = (Weff - W_min) / (W_max - W_min)
        lambda1 = 3 - 2 * lambda2
        return np.diag([lambda1, lambda2])

# ----------------------------------
# 导航向量计算器
# ----------------------------------
class NavigationVectorCalculator:
    def __init__(self, waypoints, look_ahead=3):
        self.waypoints = waypoints
        self.look_ahead = look_ahead
        self.alpha_weights = [0.8, 0.5, 0.2]  # 递减权重

    def get_n_vector(self, current_idx):
        """计算锚点方向向量"""
        if current_idx >= len(self.waypoints)-1:
            return np.array([1,0])
        current_center = np.mean(self.waypoints[current_idx], axis=0)
        vectors = []
        for i in range(1, self.look_ahead+1):
            if current_idx + i >= len(self.waypoints):
                break
            next_center = np.mean(self.waypoints[current_idx+i], axis=0)
            vec = next_center - current_center
            vec_norm = vec / np.linalg.norm(vec)
            vectors.append(vec_norm * self.alpha_weights[i-1])
            current_center = next_center
        if not vectors:
            return np.array([1,0])
        n_vector = np.sum(vectors, axis=0)[:2]
        return n_vector / np.linalg.norm(n_vector)*MIN_DISTANCE

# ----------------------------------
# 调用优化模块计算目标点
# ----------------------------------
def solve_optimization(LF_vertices, target_positions, initial_positions, adj_matrix,
                       current_waypoint_index):
    # 计算编队参数
    formation_icon = np.array(FORMATION_GEOMETRY_2)[:, :2]
    formation_radius = compute_formation_radius(formation_icon)
    vehicle_radius = max(p[2] for p in FORMATION_GEOMETRY)
    
    nav_calculator = NavigationVectorCalculator(MISSION_WAYPOINTS)
    target_center = np.mean([p[:2] for p in target_positions], axis=0)
    n_vector = nav_calculator.get_n_vector(current_waypoint_index)
    target_info = [(target_positions[i][0],target_positions[i][1],0.1) for i in range(4)]
    Weff = compute_effective_width(safe_region([target_info, OBSTACLES], 
                                       forward_vector=vehicles[0].forward_vector,
                                       formation_erosion=False), target_center, n_vector,
                                   formation_radius, vehicle_radius)
    
    #Gamma = compute_gamma_matrix(Weff, formation_radius, vehicle_radius)
    
    optimizer = DistributedFormationOptimizer(formation_icon, sigma=0.2, max_iter=5000, tol=1e-2, eta=0.2)
    
    q0 = FORMATION_GEOMETRY[0][:2]
    q2 = FORMATION_GEOMETRY[2][:2]
    relative_offset = n_vector
    '''
    # 环状拓扑 - 每个小车只与相邻的两个小车通信
    adjacency_matrix_ring = np.array([
        [0, 1, 0, 1],  # 小车0连接1和3
        [1, 0, 1, 0],  # 小车1连接0和2
        [0, 1, 0, 1],  # 小车2连接1和3
        [1, 0, 1, 0]   # 小车3连接0和2
    ])
    '''
    result = optimizer.optimize_distributed(
        #initial_positions=np.array([p[:2] for p in initial_positions]),
        initial_positions=np.tile(np.mean(target_positions, axis=0),(4, 1)),
        LF_vertices=LF_vertices,
        #Gamma=Gamma,
        #target_center=np.mean(target_positions, axis=0),
        #free_anchor_idx1=0,
        #free_anchor_idx2=2,
        #relative_offset=relative_offset
        adjacency_matrix= adj_matrix,
        b=relative_offset
    )
    #print(Gamma)
    #print(optimizer.ref_shape)
    return result if result is not None else np.array([p[:2] for p in initial_positions])

# ----------------------------------
# 编队控制器
# ----------------------------------
class FormationController:
    def __init__(self, vehicles, formation_offsets, waypoints, alpha_beta_groups, max_attempts=3):
        self.vehicles = vehicles
        self.formation_offsets = np.array(formation_offsets)
        self.waypoints = waypoints  # 每个航点为4x2的车辆位置数组
        self.current_waypoint_index = 0
        self.completed_centers = []  # 已到达的中心点
        self.safe_regions = []       # 安全区域多边形顶点列表
        self.max_attempts = max_attempts
        self.attempts = 0
        self.navigation_complete = False
        
        ###确定锚点方向用的
        #self.alpha_beta_groups = alpha_beta_groups
        #self.current_group_index = 0
        #self.alpha, self.beta = self.alpha_beta_groups[self.current_group_index]
        
        # 在起点计算初始安全区域
        initial_center = np.mean([v.position for v in self.vehicles], axis=0)
        initial_car_info = [(initial_center[0] + offset[0], initial_center[1] + offset[1], offset[2]) 
                            for offset in FORMATION_GEOMETRY]
        initial_safe = safe_region([initial_car_info, OBSTACLES], 
                                   forward_vector=(self.waypoints[0][0][0]-initial_center[0],
                                                   self.waypoints[0][0][1]-initial_center[1]), 
                                   formation_erosion=False)
        self.safe_regions.append(initial_safe)
        
        # 对起始目标点进行优化
        computed_target = solve_optimization(initial_safe, 
                                             self.waypoints[0], 
                                             np.array([info[:2] for info in FORMATION_GEOMETRY]), 
                                             adj_matrix, 
                                             current_waypoint_index=self.current_waypoint_index,
                                             )
        computed_target = np.array([computed_target[i] for i in range(1,5)])
        if computed_target is not None:
            if np.linalg.norm(computed_target - self.waypoints[0]) > 1e-1:
                self.waypoints[0] = computed_target

    def update_formation(self, delta_t=0.1):
        if self.navigation_complete:
            return

        # 计算当前编队中心
        current_car_points = [v.position for v in self.vehicles]
        current_center = np.mean(current_car_points, axis=0)
        current_target = np.mean(self.waypoints[self.current_waypoint_index], axis=0)
        distance = np.linalg.norm(current_center - current_target)

        # 到达当前目标后
        if distance < self.vehicles[0].arrival_threshold:
            self.completed_centers.append(current_center)
            new_car_info = [(current_center[0] + offset[0], current_center[1] + offset[1], offset[2])
                            for offset in FORMATION_GEOMETRY]
            # 当还有后续目标时
            if self.current_waypoint_index < len(self.waypoints) - 1:
                next_target = self.waypoints[self.current_waypoint_index + 1]
                next_center = np.mean(next_target, axis=0)
                # 更新前进方向
                self.vehicles[0].forward_vector = (next_center[0] - current_center[0],
                                                     next_center[1] - current_center[1])
                new_safe = safe_region([new_car_info, OBSTACLES], 
                                       forward_vector=self.vehicles[0].forward_vector,
                                       formation_erosion=False)
                self.safe_regions.append(new_safe)
                if len(self.safe_regions) > 1:
                    self.safe_regions.pop(0)
                
                # 调用优化模块得到优化目标
                computed_target = solve_optimization(new_safe, 
                                                     next_target, 
                                                     current_car_points,
                                                     adj_matrix, 
                                                     current_waypoint_index=self.current_waypoint_index,
                                                     )#给出的结果是字典需要转化为列表
                
                computed_target = np.array([computed_target[i] for i in range(1,5)])
                # 使用几何中心比较，设定阈值
                OPTIMIZATION_THRESHOLD = 0.5
                orig_center = np.mean(next_target, axis=0)
                opt_center = np.mean(computed_target, axis=0)
                diff = np.linalg.norm(opt_center - orig_center)
                if diff > OPTIMIZATION_THRESHOLD:
                    # 如果差距较大，插入新的目标
                    self.waypoints.insert(self.current_waypoint_index + 1, computed_target)
                    
                else:
                    # 否则更新现有目标
                    self.waypoints[self.current_waypoint_index + 1] = computed_target
                self.current_waypoint_index += 1
            else:
                # 最后一个目标：更新安全区域后完成导航
                new_safe = safe_region([new_car_info, OBSTACLES], 
                                       forward_vector=self.vehicles[0].forward_vector,
                                       formation_erosion=False)
                self.safe_regions.append(new_safe)
                if len(self.safe_regions) > 1:
                    self.safe_regions.pop(0)
                print("导航完成。")
                self.navigation_complete = True

            

        # 更新各车辆的目标位置
        for i, vehicle in enumerate(self.vehicles):
            target = self.waypoints[self.current_waypoint_index][i]
            
            vehicle.update_movement(target, self.vehicles, delta_t)
        

# ----------------------------------
# 计算编队半径（根据初始车辆位置）
# ----------------------------------
def compute_formation_radius(car_points):
    center = np.mean(car_points, axis=0)[:2]
    return max(np.linalg.norm(np.array(p[:2]) - center) for p in car_points)

# ----------------------------------
# 动画更新函数
# ----------------------------------
def animation_update(frame):
    if not controller.navigation_complete:
        controller.update_formation(delta_t=0.1)
    for plot, vehicle in zip(vehicle_plots, vehicles):
        plot.set_data([vehicle.position[0]], [vehicle.position[1]])
    
    # 更新状态文本
    status_text.set_text(f"当前任务航点: {controller.current_waypoint_index + 1}/{len(controller.waypoints)}")
    
    # 更新已完成的中心点
    completed_centers_x = [center[0] for center in controller.completed_centers]
    completed_centers_y = [center[1] for center in controller.completed_centers]
    completed_centers_plot.set_data(completed_centers_x, completed_centers_y)
    
    # 更新安全区域的显示
    if controller.safe_regions:
        safe_region_patch.set_xy(np.array(controller.safe_regions[-1]))
    else:
        safe_region_patch.set_xy(np.empty((0,2)))
    
    # 如果导航完成，显示提示信息
    if controller.navigation_complete:
        # 在“当前任务航点”下方显示“导航完成”
        completion_text.set_text("导航完成")
        completion_text.set_color("red")
        completion_text.set_fontsize(16)
    else:
        # 如果导航未完成，清空提示信息
        completion_text.set_text("")
    
    return vehicle_plots + [status_text, completed_centers_plot, safe_region_patch, completion_text]

# ----------------------------------
# 工具函数：扩展中心点为编队各车辆位置；提取编队中心
# ----------------------------------
def expand_to_formation(center, formation_offsets):
    return np.array([(center[0] + offset[0], center[1] + offset[1]) for offset in formation_offsets])

def get_formation_center(formation):
    return np.mean(formation, axis=0)

# ----------------------------------
# 邻接矩阵（完全图）
# ----------------------------------
adj_matrix = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

# ----------------------------------
# 主函数入口
# ----------------------------------
if __name__ == "__main__":
    # 编队的相对偏移量（中心对称）
    FORMATION_GEOMETRY = [
        (-0.5, -0.5, 0.1),  # 左上
        (0.5, -0.5, 0.1),   # 右上
        (-0.5, 0.5, 0.1),   # 左下
        (0.5, 0.5, 0.1)     # 右下
    ]

    FORMATION_GEOMETRY_2 = [
        (0, 0, 0.1),  # 左上
        (-1, -1, 0.1),   # 右上
        (0, -1, 0.1),   # 左下
        (-1, 0, 0.1)     # 右下
    ]

    # 障碍物信息 (x, y, r)
    OBSTACLES = [(10, 4, 4), (10, 16, 4), (5,12,2), (4,9,1)]
    formation_radius = compute_formation_radius(FORMATION_GEOMETRY)
    vehicle_radius = 0
    MIN_DISTANCE = 1.5

    # A*路径规划参数
    start_point = (1, 19)
    goal_point = (18, 18)
    num_targets = 5  # 中间目标点数量

    # 生成中间目标点
    MISSION_WAYPOINTS = a_star_path(start_point, goal_point, num_targets, OBSTACLES, 0.5, 0)
    if not MISSION_WAYPOINTS:
        MISSION_WAYPOINTS = [start_point, goal_point]
    else:
        MISSION_WAYPOINTS = [np.array(wp) for wp in MISSION_WAYPOINTS]
        MISSION_WAYPOINTS.append(goal_point)
    MISSION_WAYPOINTS = [wp if isinstance(wp, np.ndarray) else np.array(wp) for wp in MISSION_WAYPOINTS]
    MISSION_WAYPOINTS = [expand_to_formation(waypoint, FORMATION_GEOMETRY) for waypoint in MISSION_WAYPOINTS]

    # 初始化车辆
    vehicles = [
        Car(id=i, 
            init_pos=(start_point[0] + FORMATION_GEOMETRY[i][0], start_point[1] + FORMATION_GEOMETRY[i][1]),
            car_radius=FORMATION_GEOMETRY[i][2]
        ) for i in range(4)
    ]
    
    # 定义 alpha 和 beta 参数组（仅用于初始传递，不动态更新）
    alpha_beta_groups = [
        (100, 10),
        (2, 2),
        (0.2, 0.4),
        (0, 0)
    ]

    # 创建编队控制器
    controller = FormationController(
        vehicles=vehicles,
        formation_offsets=FORMATION_GEOMETRY,
        waypoints=MISSION_WAYPOINTS,
        alpha_beta_groups=alpha_beta_groups,
        max_attempts=4 
    )

    # ---------------------------
    # 可视化与动画设置
    # ---------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 20), ylim=(0, 20))
    ax.set_title("基于A*路径的编队导航")

    # 绘制A*航点（红色叉号）
    waypoints_centers = [get_formation_center(waypoint) for waypoint in MISSION_WAYPOINTS[:-1]]
    waypoints_x, waypoints_y = zip(*waypoints_centers)
    ax.scatter(waypoints_x, waypoints_y, marker='X', s=200, c='red', label='中间点')

    # 绘制起点（紫色五角星）
    start_x, start_y = start_point
    ax.scatter(start_x, start_y, marker='*', s=200, c='purple', label='起点')

    # 绘制终点（红色五角星）
    goal_x, goal_y = goal_point
    ax.scatter(goal_x, goal_y, marker='*', s=200, c='red', label='终点')
    
    # 绘制障碍物（第一个障碍物添加图例）
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
        
    # 创建颜色列表（使用tab10色系，支持最多10种不同颜色）
    colors = plt.cm.tab10(np.linspace(0, 1, len(vehicles)))

    # 创建车辆绘图对象（每个小车不同颜色）
    vehicle_plots = [
        ax.plot([], [], 'o', markersize=8, color=colors[i], label=f'小车{i+1}')[0] 
        for i in range(len(vehicles))
    ]

    # 可选：如果小车数量超过10个，改用更丰富的色系
    if len(vehicles) > 10:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(vehicles)))
        vehicle_plots = [
            ax.plot([], [], 'o', markersize=8, color=colors[i])[0] 
            for i in range(len(vehicles))
        ]
    # 添加图例说明
    ax.plot([], [], 'o', markersize=8, color='black', label='小车集群')
    
    # 已完成中心点绘图对象
    completed_centers_plot, = ax.plot([], [], 'o', markersize=5, color='black', label='已完成中心点')

    # 安全区域绘图对象
    safe_region_patch = plt.Polygon(np.array(controller.safe_regions[-1]), color='green', alpha=0.5, label='安全区域')
    ax.add_patch(safe_region_patch)

    # 状态文本（当前任务航点）
    status_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center')

    # 导航完成提示文本
    completion_text = ax.text(0.5, 0.90, '', transform=ax.transAxes, ha='center')

    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1), borderaxespad=0.)

    ani = FuncAnimation(fig, animation_update, frames=200, interval=50, blit=True)
    plt.show()
