#coding: utf-8
'''
0325更新: 添加凸多边形障碍物
0326更新: 修改凸多边形裁剪边选取逻辑, 并适配
0326_2更新：
1.更正solve_optimization 的 initial_position 参数接入
2.增加小车间距限制并适配
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from a_star.a_star_lib_v_2 import a_star_path  # 导入A*算法库
from LF2_2 import safe_region  # 导入安全区域计算
from optimization.new_optimization_3 import solve_optimization  # 导入优化库，并保留各种注释

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Car:
    
    def __init__(self, id, init_pos, car_radius, max_speed=2.0):
        self.id = id
        self.position = np.array(init_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.radius = car_radius
        self.max_speed = max_speed
        self.arrival_threshold = 0.2
        self.forward_vector = (1,0)

    def update_movement(self, target_pos, delta_t=0.1):
        # 根据目标位置更新车辆位置
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
        self.waypoints = waypoints  # waypoints 是包含4x2数组的列表
        self.current_waypoint_index = 0
        self.completed_centers = []  # 记录已到达的中心点
        self.safe_regions = []       # 存储安全区域，多边形顶点列表
        self.max_attempts = max_attempts  # 最大尝试次数
        self.attempts = 0            # 当前尝试次数
        self.navigation_complete = False  # 导航是否完成

        # 定义 alpha 和 beta 的组
        self.alpha_beta_groups = alpha_beta_groups
        self.current_group_index = 0  # 当前使用的组索引
        self.alpha, self.beta = self.alpha_beta_groups[self.current_group_index]  # 初始化 alpha 和 beta

        # 在起点计算并存储初始安全区域
        initial_center = np.mean([v.position for v in self.vehicles], axis=0)
        initial_car_information = [(initial_center[0] + offset[0], initial_center[1] + offset[1], offset[2]) for offset in FORMATION_GEOMETRY]
        initial_car_points = [(info[0], info[1]) for info in initial_car_information]
        initial_target_position = self.waypoints[0]  # 初始目标点已经是4x2数组
        initial_target_center = np.mean(initial_target_position, axis=0)
        self.vehicles[0].forward_vector = (initial_target_center[0] - initial_center[0],
                                  initial_target_center[1] - initial_center[1])
        initial_safe = safe_region([initial_car_information, OBSTACLES], 
                                   forward_vector= self.vehicles[0].forward_vector, 
                                   formation_erosion=False)
        self.safe_regions.append(initial_safe)
        
        # 对起始目标点进行优化计算
        
        computed_target = solve_optimization(initial_safe, 
                                             initial_target_position, 
                                             np.array([info[:2] for info in FORMATION_GEOMETRY]), 
                                             adj_matrix, 
                                             min_distance_threshold=MIN_DISTANCE,
                                             alpha=self.alpha, 
                                             beta=self.beta)
        if computed_target is not None:
            # 如果优化后的目标点与原目标点不同，则更新目标点
            if np.linalg.norm(computed_target - initial_target_position) > 1e-1:
                self.waypoints[0] = computed_target
                print(np.mean(computed_target, axis = 0))

    def update_formation(self, delta_t=0.1):
        if self.navigation_complete:
            return  # 如果导航已完成，直接返回

        # 计算当前编队中心
        current_car_point = [v.position for v in self.vehicles]
        current_center = np.mean(current_car_point, axis=0)
        current_target = np.mean(self.waypoints[self.current_waypoint_index], axis=0)  # 当前目标中心
        distance = np.linalg.norm(current_center - current_target)

        # 当到达当前目标点时
        if distance < self.vehicles[0].arrival_threshold:
            self.completed_centers.append(current_center)
            new_car_information = [(current_center[0] + offset[0], current_center[1] + offset[1], offset[2]) for offset in FORMATION_GEOMETRY]
            new_car_points = [(info[0], info[1]) for info in new_car_information]
            

            # 使用优化库计算新的目标点，并更新目标列表
            if self.current_waypoint_index < len(self.waypoints) - 1:
                next_target = self.waypoints[self.current_waypoint_index + 1]  # 下一个目标点（4x2数组）
                next_center = np.mean(next_target, axis=0) #下一个目标中心(x, y)
                previous_distance = np.linalg.norm(current_center - next_center)  # 上一次优化前的距离

                self.vehicles[0].forward_vector = (next_center[0] - current_center[0], next_center[1] - current_center[1])
                new_safe = safe_region([new_car_information, OBSTACLES], 
                                       forward_vector=self.vehicles[0].forward_vector,
                                       formation_erosion=False)
                self.safe_regions.append(new_safe)
                if len(self.safe_regions) > 1:
                    self.safe_regions.pop(0)  # 仅保留最新的安全区域

                computed_target = solve_optimization(new_safe, 
                                                     next_target, 
                                                     np.array([info[:2] for info in FORMATION_GEOMETRY]), 
                                                     adj_matrix, 
                                                     min_distance_threshold=MIN_DISTANCE,
                                                     alpha=self.alpha, 
                                                     beta=self.beta)
                
                # 判断优化是否有效
                if computed_target is not None:
                    computed_center = np.mean(computed_target, axis=0)

                    while not self.is_optimization_effective(computed_center, next_center, previous_distance):
                        # 如果优化无效（陷入局部极小值），则采取相应措施
                        self.attempts += 1
                        if self.attempts >= self.max_attempts:
                            print("优化陷入局部极小值，导航失败。")
                            self.navigation_complete = True
                            print("当前计算得目标点\n", computed_target)
                            print("当前下一个目标点\n", next_target)
                            print(previous_distance)
                            #print(self.vehicles[0].forward_vector)
                            return
                        
                        print(f"优化陷入局部极小值，尝试次数: {self.attempts}/{self.max_attempts}")
                        # 切换到下一组 alpha 和 beta
                        if self.current_group_index < len(self.alpha_beta_groups) - 1:
                            self.current_group_index += 1
                            self.alpha, self.beta = self.alpha_beta_groups[self.current_group_index]
                        else:
                            # 如果已经是最后一组，保持当前值
                            pass
                        computed_target = solve_optimization(new_safe, 
                                                             next_target, 
                                                             np.array([info[:2] for info in FORMATION_GEOMETRY]), 
                                                             adj_matrix, 
                                                             min_distance_threshold=MIN_DISTANCE,
                                                             alpha=self.alpha, 
                                                             beta=self.beta)
                        computed_center = np.mean(computed_target, axis=0)
                            
                    if self.is_optimization_effective(computed_center, next_center, previous_distance):
                        # 如果优化有效，则更新目标点
                        if np.linalg.norm(computed_center - next_center) >= 0.1:
                            #print(f"insert{computed_center}")
                            self.waypoints.insert(self.current_waypoint_index + 1, computed_target)
                        else:
                            self.waypoints[self.current_waypoint_index + 1] = computed_target
                        self.attempts = 0  # 重置尝试次数
                        # 如果当前不是第一组，逐步回到前一组
                        if self.current_group_index > 0:
                            self.current_group_index = 0
                            self.alpha, self.beta = self.alpha_beta_groups[self.current_group_index]
                    

            # 更新目标索引
            
                self.current_waypoint_index += 1
            else:
                new_safe = safe_region([new_car_information, OBSTACLES], 
                                       forward_vector=self.vehicles[0].forward_vector,
                                       formation_erosion=False)
                self.safe_regions.append(new_safe)
                if len(self.safe_regions) > 1:
                    self.safe_regions.pop(0)  # 仅保留最新的安全区域
                print("导航完成。")
                self.navigation_complete = True

        # 更新各车辆的目标位置
        for i, vehicle in enumerate(self.vehicles):
            target = self.waypoints[self.current_waypoint_index][i]  # 当前目标点中第i个车辆的目标
            vehicle.update_movement(target, delta_t)

    def is_optimization_effective(self, computed_center, target_center, previous_distance, threshold=0.21):
        """
        判断优化是否有效，即小车中心是否显著靠近目标中心
        :param current_center: 当前小车中心的坐标 (x, y)
        :param target_center: 目标中心的坐标 (x, y)
        :param previous_distance: 上一次优化前的距离
        :param threshold: 距离变化的阈值，默认 0.1
        :return: True 表示优化有效, False 表示陷入局部极小值
        """
        current_distance = np.linalg.norm(computed_center - target_center)
        return current_distance < previous_distance - threshold
        
# 计算小车集群的半径（以初始车辆位置计算）
def compute_formation_radius(car_points):
    center = np.mean(car_points, axis=0)[:2]
    return max(np.linalg.norm(np.array(p[:2]) - center) for p in car_points)

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

def expand_to_formation(center, formation_offsets):
    """
    将中心点扩展为包含4个车辆坐标的编队
    :param center: 中心点坐标 (x, y)
    :param formation_offsets: 编队偏移量列表
    :return: 4x2 的数组, 表示4个车辆的坐标
    """
    return np.array([(center[0] + offset[0], center[1] + offset[1]) for offset in formation_offsets])

def get_formation_center(formation):
    """
    提取编队的几何中心坐标
    :param formation: 4x2 的数组，表示4个车辆的坐标
    :return: 几何中心坐标 (x, y)
    """
    return np.mean(formation, axis=0)

# 设置邻接矩阵（完全图）
adj_matrix = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

if __name__ == "__main__":
    # 编队的相对偏移量，保持中心对称
    FORMATION_GEOMETRY = [
        (-0.5, -0.5, 0.1),  # 左上
        (0.5, -0.5, 0.1),   # 右上
        (-0.5, 0.5, 0.1),   # 左下
        (0.5, 0.5, 0.1)     # 右下
    ]

    # 障碍物信息（采用 (x, y, r) 三元组形式）
    OBSTACLES = [(10, 4, 4), (10, 16, 4),(5,12,2),(4,9,1)]
    #OBSTACLES = [[(5,0), (5,10.2), (15,10.2), (15,0)], [(5,11), (5,20), (15,20), (15,11)]]
    #OBSTACLES = [(10, 4, 4), (10, 16, 4)]
    formation_radius = compute_formation_radius(FORMATION_GEOMETRY)
    # 小车自身半径参数，初始默认为0（可调节）
    vehicle_radius = 0
    MIN_DISTANCE = 0.1
    # A*路径规划参数
    start_point = (1, 19)
    goal_point = (18, 18)
    num_targets = 7  # 可调节中间目标点数量
    
    # 调用A*算法生成中间目标点
    MISSION_WAYPOINTS = a_star_path(start_point, goal_point, num_targets, OBSTACLES, 0.2, vehicle_radius)
    #MISSION_WAYPOINTS = [ [4,10.5], [8,10.5], [14, 10.5], [16, 10.5]]
    if not MISSION_WAYPOINTS:
        # 如果没有生成路径，使用起点和终点
        MISSION_WAYPOINTS = [start_point, goal_point]
    else:
        # 将终点添加到路径中
        MISSION_WAYPOINTS.append(goal_point)

    # 将中心点路径扩展为包含4个车辆坐标的路径
    MISSION_WAYPOINTS = [expand_to_formation(waypoint, FORMATION_GEOMETRY) for waypoint in MISSION_WAYPOINTS]
    #print(np.mean(MISSION_WAYPOINTS, axis = 1))
    # 初始化车辆
    vehicles = [
        Car(id = i, 
            init_pos=
                (start_point[0] + FORMATION_GEOMETRY[i][0], 
                start_point[1] + FORMATION_GEOMETRY[i][1]),
            car_radius = FORMATION_GEOMETRY[i][2]
            ) 
            for i in range(4)]
    
    # 定义三组 alpha 和 beta 值
    alpha_beta_groups = [
        (10000, 10000),  # 第一组
        (20, 20), # 第二组
        (0.2, 0.2),  # 第三组
        (0, 0)
    ]

    # 创建编队控制器
    controller = FormationController(
        vehicles=vehicles,
        formation_offsets=FORMATION_GEOMETRY,
        waypoints=MISSION_WAYPOINTS,
        alpha_beta_groups=alpha_beta_groups,  # 传入 alpha 和 beta 的组
        max_attempts=4 
    )

    # ---------------------------
    # 可视化与动画
    # ---------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 20), ylim=(0, 20))
    ax.set_title("基于A*路径的编队导航")

    # 绘制A*生成的航点（红色叉号）
    waypoints_centers = [get_formation_center(waypoint) for waypoint in MISSION_WAYPOINTS[:-1]]
    waypoints_x, waypoints_y = zip(*waypoints_centers)
    ax.scatter(waypoints_x, waypoints_y, marker='X', s=200, c='red', label='中间点')

    # 绘制起点（紫色五角星）
    start_x, start_y = start_point
    ax.scatter(start_x, start_y, marker='*', s=200, c='purple', label='起点')

    # 绘制终点（红色五角星）
    goal_x, goal_y = goal_point
    ax.scatter(goal_x, goal_y, marker='*', s=200, c='red', label='终点')
    
    # 绘制障碍物（仅为第一个障碍物添加图例），图例放在右上方外部
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
        
    # 创建动态车辆绘图对象，统一颜色为蓝色
    vehicle_plots = [ax.plot([], [], 'o', markersize=8, color='blue')[0] for _ in vehicles]

    # 只显示一个小车的图例
    vehicle_plots[0].set_label('小车')
    
    # 创建已完成中心点的绘图对象
    completed_centers_plot, = ax.plot([], [], 'o', markersize=5, color='black', label='已完成中心点')

    # 创建安全区域的绘图对象
    safe_region_patch = plt.Polygon(np.array(controller.safe_regions[-1]), color='green', alpha=0.5, label='安全区域')
    ax.add_patch(safe_region_patch)

    # 创建状态文本（当前任务航点）
    status_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center')

    # 创建导航完成提示文本（放在状态文本下方）
    completion_text = ax.text(0.5, 0.90, '', transform=ax.transAxes, ha='center')

    # 设置图例位置
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1), borderaxespad=0.)

    ani = FuncAnimation(fig, animation_update, frames=200, interval=50, blit=True)
    plt.show()
