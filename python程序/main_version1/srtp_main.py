import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from a_star_lib import a_star_path  # 导入A*算法库

# 设置中文字体，确保图中中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Car:
    """车辆运动控制模型"""
    def __init__(self, id, init_pos, max_speed=2.0):
        self.id = id
        self.position = np.array(init_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.max_speed = max_speed
        self.arrival_threshold = 0.1

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
    """基于编队中心的控制系统"""
    def __init__(self, vehicles, formation_offsets, waypoints):
        self.vehicles = vehicles
        self.formation_offsets = np.array(formation_offsets)
        self.waypoints = waypoints
        self.current_waypoint_index = 0

        # 验证编队偏移量总和是否为零
        sum_offsets = np.sum(self.formation_offsets, axis=0)
        if not np.allclose(sum_offsets, [0, 0]):
            print("警告：编队偏移量总和不为零，可能影响导航精度")

    def update_formation(self, delta_t=0.1):
        # 计算当前编队中心
        current_center = np.mean([v.position for v in self.vehicles], axis=0)
        current_target = np.array(self.waypoints[self.current_waypoint_index])
        distance = np.linalg.norm(current_center - current_target)
        if distance < self.vehicles[0].arrival_threshold:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                print(f"到达路径点 {self.current_waypoint_index}")
        # 更新所有车辆目标位置（目标 = 当前航点 + 车辆对应的编队偏移）
        for i, vehicle in enumerate(self.vehicles):
            target = np.array(self.waypoints[self.current_waypoint_index]) + self.formation_offsets[i]
            vehicle.update_movement(target, delta_t)

# ---------------------------
# 配置参数（可在主文件中调节）
# ---------------------------
# 编队相对偏移（保持中心对称）
FORMATION_GEOMETRY = [
    [-0.5, -0.5],  # 左上
    [0.5, -0.5],   # 右上
    [-0.5, 0.5],   # 左下
    [0.5, 0.5]     # 右下
]

# 障碍物信息（采用 (x, y, r) 三元组形式）
OBSTACLES = [(10,4,4), (10,16,4)]

# 计算小车集群的半径（以初始车辆位置计算）
def compute_formation_radius(car_points):
    center = np.mean(car_points, axis=0)
    return max(np.linalg.norm(np.array(p) - center) for p in car_points)

# 初始小车位置
CAR_POINTS = [(0,0), (1,0), (1,1), (0,1)]
formation_radius = compute_formation_radius(CAR_POINTS)
# 小车自身半径参数，初始默认为0（可调节）
vehicle_radius = 0

# A*路径规划参数
start_point = (0, 0)
goal_point  = (20, 20)
num_targets = 5  # 可调节中间目标点数量

# 调用A*算法生成中间目标点作为任务航点
MISSION_WAYPOINTS = a_star_path(start_point, goal_point, num_targets, OBSTACLES, formation_radius, vehicle_radius)
if not MISSION_WAYPOINTS:
    MISSION_WAYPOINTS = [start_point, goal_point]  # 若A*未能生成路径则直接采用起终点
print("生成的任务航点:", MISSION_WAYPOINTS)

# 初始化车辆（这里以4辆车为例，初始位置与CAR_POINTS一致）
vehicles = [
    Car(0, [0, 0]),
    Car(1, [1, 0]),
    Car(2, [0, 1]),
    Car(3, [1, 1])
]

# 创建编队控制器
controller = FormationController(
    vehicles=vehicles,
    formation_offsets=FORMATION_GEOMETRY,
    waypoints=MISSION_WAYPOINTS
)

# ---------------------------
# 可视化与动画
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 10))
ax.set(xlim=(0, 20), ylim=(0, 20))
ax.set_title("基于A*路径的编队导航")

# 绘制A*生成的航点（红色叉号）
ax.scatter(*zip(*MISSION_WAYPOINTS), marker='X', s=200, c='red')

# 绘制障碍物（仅为第一个障碍物添加图例），图例放在右上方外部
    first_obs = True
    for obs in obstacles:
        center = (obs[0], obs[1])
        if first_obs:
            circle = plt.Circle(center, obs[2], color='red', fill=False, linestyle='--', label='障碍物')
            first_obs = False
        else:
            circle = plt.Circle(center, obs[2], color='red', fill=False, linestyle='--')
        ax.add_patch(circle)

# 创建动态车辆绘图对象
vehicle_plots = [ax.plot([], [], 'o', markersize=5)[0] for _ in vehicles]
status_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center')

def animation_update(frame):
    controller.update_formation(delta_t=0.1)
    for plot, vehicle in zip(vehicle_plots, vehicles):
        plot.set_data([vehicle.position[0]], [vehicle.position[1]])
    status_text.set_text(
        f"当前任务航点: {controller.current_waypoint_index + 1}/{len(MISSION_WAYPOINTS)}")
    return vehicle_plots + [status_text]

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

ani = FuncAnimation(fig, animation_update, frames=200, interval=50, blit=True)
plt.show()
