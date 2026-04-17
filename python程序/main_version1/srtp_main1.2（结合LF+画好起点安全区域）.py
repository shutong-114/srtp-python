import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from a_star_lib import a_star_path  # 导入A*算法库
from LF_version.LF import safe_region  # 导入安全区域计算

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Car:
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
    def __init__(self, vehicles, formation_offsets, waypoints):
        self.vehicles = vehicles
        self.formation_offsets = np.array(formation_offsets)
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.completed_centers = []  # 记录已到达的中心点
        self.safe_regions = []       # 存储安全区域，多边形顶点列表

        # 在起点计算并存储初始安全区域
        initial_center = np.mean([v.position for v in self.vehicles], axis=0)
        initial_car_points = [(initial_center[0] + offset[0], initial_center[1] + offset[1]) for offset in FORMATION_GEOMETRY]
        self.safe_regions.append(safe_region([initial_car_points, OBSTACLES]))

    def update_formation(self, delta_t=0.1):
        current_center = np.mean([v.position for v in self.vehicles], axis=0)
        current_target = np.array(self.waypoints[self.current_waypoint_index])
        distance = np.linalg.norm(current_center - current_target)
        if distance < self.vehicles[0].arrival_threshold:
            self.completed_centers.append(current_center)
            new_car_points = [(current_center[0] + offset[0], current_center[1] + offset[1]) for offset in FORMATION_GEOMETRY]
            self.safe_regions.append(safe_region([new_car_points, OBSTACLES]))
            if len(self.safe_regions) > 1:
                self.safe_regions.pop(0)  # 仅保留最新的安全区域
            
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
            else:
                # 到达终点后也更新安全区域
                self.safe_regions.append(safe_region([new_car_points, OBSTACLES]))
        
        for i, vehicle in enumerate(self.vehicles):
            target = np.array(self.waypoints[self.current_waypoint_index]) + self.formation_offsets[i]
            vehicle.update_movement(target, delta_t)

def animation_update(frame):
    controller.update_formation(delta_t=0.1)
    for plot, vehicle in zip(vehicle_plots, vehicles):
        plot.set_data([vehicle.position[0]], [vehicle.position[1]])
    status_text.set_text(
        f"当前任务航点: {controller.current_waypoint_index + 1}/{len(MISSION_WAYPOINTS)}"
    )
    completed_centers_x = [center[0] for center in controller.completed_centers]
    completed_centers_y = [center[1] for center in controller.completed_centers]
    completed_centers_plot.set_data(completed_centers_x, completed_centers_y)
    
    if controller.safe_regions:
        safe_region_patch.set_xy(np.array(controller.safe_regions[-1]))
    else:
        safe_region_patch.set_xy(np.empty((0,2)))
    
    return vehicle_plots + [status_text, completed_centers_plot, safe_region_patch]

if __name__ == "__main__":
    FORMATION_GEOMETRY = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    OBSTACLES = [(10, 4, 4), (10, 16, 4)]
    start_point = (1, 1)
    goal_point  = (18, 18)
    MISSION_WAYPOINTS = a_star_path(start_point, goal_point, 5, OBSTACLES, 1, 0)
    if not MISSION_WAYPOINTS:
        MISSION_WAYPOINTS = [start_point, goal_point]
    else:
        MISSION_WAYPOINTS.append(goal_point)
    
    vehicles = [Car(i, (start_point[0] + FORMATION_GEOMETRY[i][0], start_point[1] + FORMATION_GEOMETRY[i][1])) for i in range(4)]
    controller = FormationController(vehicles, FORMATION_GEOMETRY, MISSION_WAYPOINTS)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 20), ylim=(0, 20))
    ax.set_title("基于A*路径的编队导航")
    
    waypoints_x, waypoints_y = zip(*MISSION_WAYPOINTS[:-1])
    ax.scatter(waypoints_x, waypoints_y, marker='X', s=200, c='red', label='中间点')
    ax.scatter(start_point[0], start_point[1], marker='*', s=200, c='purple', label='起点')
    ax.scatter(goal_point[0], goal_point[1], marker='*', s=200, c='red', label='终点')
    
    for obs in OBSTACLES:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', fill=False, linestyle='--')
        ax.add_patch(circle)
    
    vehicle_plots = [ax.plot([], [], 'o', markersize=8, color='blue')[0] for _ in vehicles]
    vehicle_plots[0].set_label('小车')
    completed_centers_plot, = ax.plot([], [], 'o', markersize=5, color='black', label='已完成中心点')
    
    safe_region_patch = plt.Polygon(np.array(controller.safe_regions[-1]), color='green', alpha=0.5, label='安全区域')
    ax.add_patch(safe_region_patch)
    
    status_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1), borderaxespad=0.)
    
    ani = FuncAnimation(fig, animation_update, frames=200, interval=50, blit=True)
    plt.show()
