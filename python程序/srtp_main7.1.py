# coding: utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull

from LF2_2 import safe_region
from a_star_lib_v_5 import a_star_path
from distributed_optimization_operator_repulsion import (
    DistributedFormationOptimizerWithRepulsion as DistributedFormationOptimizer,
)
from rvo import compute_RVO_velocity


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


FORMATION_GEOMETRY = [
    (-0.5, -0.5, 0.1),
    (0.5, -0.5, 0.1),
    (-0.5, 0.5, 0.1),
    (0.5, 0.5, 0.1),
]
FORMATION_GEOMETRY_2 = [
    (0.0, 0.0, 0.1),
    (-1.0, -1.0, 0.1),
    (0.0, -1.0, 0.1),
    (-1.0, 0.0, 0.1),
]
FORMATION_GEOMETRY_PENTAGON = []
FORMATION_GEOMETRY_PENTAGON_2 = []
ADJ_MATRIX_4 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
])
ADJ_MATRIX_5 = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
])
OBSTACLES = []
MISSION_CENTERS = []
MIN_DISTANCE = 1.5


class Car:
    def __init__(self, id, init_pos, car_radius, max_speed=2.0):
        self.id = id
        self.position = np.array(init_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.radius = car_radius
        self.max_speed = max_speed
        self.arrival_threshold = 0.2
        self.forward_vector = (1.0, 0.0)
        self.active_in_formation = True
        self.frozen = False
        self.display_only = False

    def update_movement(self, target_pos, all_cars, delta_t=0.1):
        if self.frozen or target_pos is None:
            self.velocity = np.zeros(2)
            return False

        neighbors_pos = []
        neighbors_vel = []
        neighbors_radius = []
        for car in all_cars:
            if car.id == self.id:
                continue
            neighbors_pos.append(car.position)
            neighbors_vel.append(car.velocity)
            neighbors_radius.append(car.radius)

        direction = np.asarray(target_pos, dtype=float) - self.position
        distance = np.linalg.norm(direction)
        if distance > self.arrival_threshold:
            preferred_vel = direction / distance * self.max_speed
        else:
            preferred_vel = np.zeros(2)

        new_vel = compute_RVO_velocity(
            self.position,
            preferred_vel,
            self.radius,
            neighbors_pos,
            neighbors_vel,
            neighbors_radius,
            self.max_speed,
        )
        self.velocity = new_vel
        self.position += self.velocity * delta_t
        return np.linalg.norm(self.position - target_pos) <= self.arrival_threshold


def is_point_in_polygon(point, polygon):
    polygon = np.array(polygon)
    hull = ConvexHull(polygon)
    for eq in hull.equations:
        a, b, c = eq
        if a * point[0] + b * point[1] + c > 1e-9:
            return False
    return True


def compute_effective_width(LF_vertices, center_point, n_vector, _radius, _vehicle_radius):
    n_vector = np.asarray(n_vector).flatten()
    center_point = np.asarray(center_point).flatten()
    if np.linalg.norm(n_vector[:2]) < 1e-9:
        return 0.0
    n_unit = n_vector[:2] / np.linalg.norm(n_vector[:2])
    mu_vector = np.array([-n_unit[1], n_unit[0]])

    def ray_cast(p, direction, max_len=20.0, step=0.1):
        direction = direction / np.linalg.norm(direction)
        direction = np.asarray(direction).flatten()
        for t in np.arange(0, max_len + step, step):
            test_point = p + t * direction
            if not is_point_in_polygon(test_point, LF_vertices):
                return max(step, t - step)
        return max_len

    forward = ray_cast(center_point, mu_vector)
    backward = ray_cast(center_point, -mu_vector)
    return forward + backward


class NavigationVectorCalculator:
    def __init__(self, waypoints, look_ahead=3):
        self.waypoints = waypoints
        self.look_ahead = look_ahead
        self.alpha_weights = [0.8, 0.5, 0.2]

    def get_n_vector(self, current_idx):
        if current_idx >= len(self.waypoints) - 1:
            return np.array([1.0, 0.0])
        current_center = np.asarray(self.waypoints[current_idx], dtype=float)
        vectors = []
        for i in range(1, self.look_ahead + 1):
            if current_idx + i >= len(self.waypoints):
                break
            next_center = np.asarray(self.waypoints[current_idx + i], dtype=float)
            vec = next_center - current_center
            if np.linalg.norm(vec) < 1e-9:
                continue
            vec_norm = vec / np.linalg.norm(vec)
            vectors.append(vec_norm * self.alpha_weights[i - 1])
            current_center = next_center
        if not vectors:
            return np.array([1.0, 0.0])
        n_vector = np.sum(vectors, axis=0)[:2]
        if np.linalg.norm(n_vector) < 1e-9:
            return np.array([1.0, 0.0])
        return n_vector / np.linalg.norm(n_vector) * MIN_DISTANCE


def compute_regular_polygon_offsets(count, radius, car_radius=0.1, start_angle=math.pi / 2):
    offsets = []
    for idx in range(count):
        angle = start_angle + idx * 2 * math.pi / count
        offsets.append((radius * math.cos(angle), radius * math.sin(angle), car_radius))
    return offsets


def expand_to_formation(center, formation_offsets):
    center = np.asarray(center, dtype=float)
    return np.array([(center[0] + offset[0], center[1] + offset[1]) for offset in formation_offsets], dtype=float)


def get_formation_center(formation):
    return np.mean(formation, axis=0)


def compute_formation_radius(car_points):
    center = np.mean(np.array(car_points)[:, :2], axis=0)
    return max(np.linalg.norm(np.array(point[:2]) - center) for point in car_points)


def compute_event_indices(num_intermediate_targets):
    total_waypoints = max(1, num_intermediate_targets + 1)
    join_idx = min(2, max(0, total_waypoints - 2))
    dropout_idx = max(join_idx + 1, total_waypoints - 2)
    dropout_idx = min(dropout_idx, total_waypoints - 1)
    return join_idx, dropout_idx


def enforce_min_distance(positions, min_dist=0.5, max_iterations=200):
    """Push apart any pair of positions that are closer than min_dist."""
    positions = np.array(positions, dtype=float)
    n = len(positions)
    for _ in range(max_iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    if dist < 1e-9:
                        diff = np.random.randn(2)
                        diff /= np.linalg.norm(diff)
                    else:
                        diff /= dist
                    correction = (min_dist - dist) / 2.0 * diff
                    positions[i] += correction
                    positions[j] -= correction
                    moved = True
        if not moved:
            break
    return positions


def build_wait_position(join_center, wait_offset=(1.5, 1.0)):
    return np.asarray(join_center, dtype=float) + np.asarray(wait_offset, dtype=float)


def solve_optimization(
    LF_vertices,
    target_positions,
    initial_positions,
    adj_matrix,
    current_waypoint_index,
    formation_icon,
    forward_vector,
):
    target_positions = np.asarray(target_positions, dtype=float)
    formation_icon = np.asarray(formation_icon, dtype=float)[:, :2]
    target_center = np.mean(target_positions, axis=0)
    active_count = len(target_positions)

    nav_calculator = NavigationVectorCalculator(MISSION_CENTERS if MISSION_CENTERS else [target_center])
    n_vector = nav_calculator.get_n_vector(current_waypoint_index)
    if np.linalg.norm(n_vector) < 1e-9:
        n_vector = np.array([1.0, 0.0]) * MIN_DISTANCE

    vehicle_radius = 0.1
    if active_count == 4:
        vehicle_radius = max(point[2] for point in FORMATION_GEOMETRY)
    elif FORMATION_GEOMETRY_PENTAGON:
        vehicle_radius = max(point[2] for point in FORMATION_GEOMETRY_PENTAGON)

    try:
        target_info = [(point[0], point[1], vehicle_radius) for point in target_positions]
        formation_radius = compute_formation_radius([(point[0], point[1], vehicle_radius) for point in target_positions])
        width_region = safe_region(
            [target_info, OBSTACLES],
            forward_vector=forward_vector,
            formation_erosion=False,
        )
        compute_effective_width(width_region, target_center, n_vector, formation_radius, vehicle_radius)

        optimizer = DistributedFormationOptimizer(
            formation_icon, sigma=0.2, max_iter=5000, tol=1e-2, eta=0.2,
            min_dist=0.5, repulsion_inner_iters=20, repulsion_every=1,
        )
        optimized = optimizer.optimize_distributed(
            initial_positions=np.tile(target_center, (active_count, 1)),
            LF_vertices=LF_vertices,
            adjacency_matrix=adj_matrix,
            b=n_vector,
        )
        if isinstance(optimized, dict):
            return np.array([optimized[idx] for idx in range(1, active_count + 1)], dtype=float)
        return np.asarray(optimized, dtype=float)
    except Exception:
        return np.asarray(initial_positions if len(initial_positions) == active_count else target_positions, dtype=float)


class FormationController:
    def __init__(
        self,
        vehicles,
        square_offsets,
        pentagon_offsets,
        waypoint_centers,
        waiting_vehicle_id=4,
        join_wait_offset=(1.5, 1.0),
        max_attempts=3,
    ):
        self.vehicles = vehicles
        self.square_offsets = list(square_offsets)
        self.pentagon_offsets = list(pentagon_offsets)
        self.waypoint_centers = [np.asarray(center, dtype=float) for center in waypoint_centers]
        self.current_waypoint_index = 0
        self.completed_centers = []
        self.safe_regions = []
        self.max_attempts = max_attempts
        self.attempts = 0
        self.navigation_complete = False
        self.state = "four_car_cruise"
        self.waiting_vehicle_id = waiting_vehicle_id
        self.join_wait_offset = np.asarray(join_wait_offset, dtype=float)
        self.join_waypoint_index, self.dropout_waypoint_index = compute_event_indices(len(self.waypoint_centers) - 1)
        self.formation_vehicle_ids = [0, 1, 2, 3]
        self.target_index_by_vehicle_id = {}
        self.current_target_positions = np.empty((0, 2))

        self._place_waiting_vehicle()
        self.recompute_current_targets()

    def _place_waiting_vehicle(self):
        waiting_car = self.vehicles[self.waiting_vehicle_id]
        waiting_car.position = build_wait_position(
            self.waypoint_centers[self.join_waypoint_index],
            self.join_wait_offset,
        )
        waiting_car.velocity = np.zeros(2)
        waiting_car.active_in_formation = False
        waiting_car.frozen = True
        waiting_car.display_only = False

    def get_phase_label(self):
        phase_labels = {
            "four_car_cruise": "4车巡航",
            "join_wait": "合流中",
            "five_car_cruise": "5车巡航",
            "dropout_done": "5车断联后4车巡航",
            "navigation_complete": "导航完成",
        }
        return phase_labels.get(self.state, self.state)

    def get_current_offsets(self):
        return self.pentagon_offsets if len(self.formation_vehicle_ids) == 5 else self.square_offsets

    def get_current_shape_icon(self):
        return FORMATION_GEOMETRY_PENTAGON_2 if len(self.formation_vehicle_ids) == 5 else FORMATION_GEOMETRY_2

    def get_current_adj_matrix(self):
        return ADJ_MATRIX_5 if len(self.formation_vehicle_ids) == 5 else ADJ_MATRIX_4

    def get_active_positions(self):
        return np.array([self.vehicles[vehicle_id].position for vehicle_id in self.formation_vehicle_ids], dtype=float)

    def compute_forward_vector(self):
        if self.current_waypoint_index < len(self.waypoint_centers) - 1:
            next_center = self.waypoint_centers[self.current_waypoint_index + 1]
            current_center = self.waypoint_centers[self.current_waypoint_index]
            vec = next_center - current_center
        else:
            vec = np.asarray(self.vehicles[self.formation_vehicle_ids[0]].forward_vector, dtype=float)
        if np.linalg.norm(vec) < 1e-9:
            return np.array([1.0, 0.0])
        return vec

    def build_current_car_info(self):
        return [(car.position[0], car.position[1], car.radius) for car in self.get_active_cars()]

    def get_active_cars(self):
        return [self.vehicles[vehicle_id] for vehicle_id in self.formation_vehicle_ids]

    def recompute_current_targets(self):
        if self.navigation_complete:
            return

        current_center = self.waypoint_centers[self.current_waypoint_index]
        offsets = self.get_current_offsets()
        base_targets = expand_to_formation(current_center, offsets)
        active_positions = self.get_active_positions()
        forward_vector = self.compute_forward_vector()

        for vehicle_id in self.formation_vehicle_ids:
            self.vehicles[vehicle_id].forward_vector = tuple(forward_vector)

        current_car_info = self.build_current_car_info()
        try:
            new_safe = safe_region(
                [current_car_info, OBSTACLES],
                forward_vector=forward_vector,
                formation_erosion=False,
            )
        except Exception:
            new_safe = np.array(current_car_info)[:, :2] if current_car_info else np.empty((0, 2))

        self.safe_regions.append(new_safe)
        if len(self.safe_regions) > 1:
            self.safe_regions.pop(0)

        optimized_targets = solve_optimization(
            new_safe,
            base_targets,
            active_positions,
            self.get_current_adj_matrix(),
            self.current_waypoint_index,
            self.get_current_shape_icon(),
            forward_vector,
        )
        optimized_targets = np.asarray(optimized_targets, dtype=float)
        if optimized_targets.shape != base_targets.shape:
            optimized_targets = base_targets

        self.current_target_positions = optimized_targets
        self.target_index_by_vehicle_id = {
            vehicle_id: idx for idx, vehicle_id in enumerate(self.formation_vehicle_ids)
        }

    def all_active_vehicles_arrived(self):
        if len(self.current_target_positions) == 0:
            return False
        current_center = np.mean(
            [self.vehicles[vid].position for vid in self.formation_vehicle_ids], axis=0
        )
        target_center = np.mean(self.current_target_positions, axis=0)
        threshold = self.vehicles[self.formation_vehicle_ids[0]].arrival_threshold
        return np.linalg.norm(current_center - target_center) < threshold

    def trigger_join_wait(self):
        self.state = "join_wait"
        join_car = self.vehicles[self.waiting_vehicle_id]
        join_car.active_in_formation = True
        join_car.frozen = False
        self.formation_vehicle_ids = [0, 1, 2, 3, self.waiting_vehicle_id]
        self.recompute_current_targets()

    def finalize_join(self):
        self.state = "five_car_cruise"
        self._advance_to_next_waypoint()

    def trigger_dropout(self):
        dropout_car = self.vehicles[self.waiting_vehicle_id]
        dropout_car.active_in_formation = False
        dropout_car.frozen = True
        dropout_car.velocity = np.zeros(2)
        self.formation_vehicle_ids = [0, 1, 2, 3]
        self.state = "dropout_done"
        self._advance_to_next_waypoint()

    def _advance_to_next_waypoint(self):
        """5.1-style advance: pre-compute targets for next waypoint, insert intermediate
        if the optimizer shifts the formation centre by more than OPTIMIZATION_THRESHOLD."""
        _OPTIMIZATION_THRESHOLD = 0.5

        if self.current_waypoint_index >= len(self.waypoint_centers) - 1:
            self.navigation_complete = True
            self.state = "navigation_complete"
            return

        # Forward vector from current waypoint toward the next (index not yet advanced)
        forward_vector = self.compute_forward_vector()
        for vid in self.formation_vehicle_ids:
            self.vehicles[vid].forward_vector = tuple(forward_vector)

        # Safe region at the current (arrived) position
        current_car_info = self.build_current_car_info()
        try:
            new_safe = safe_region(
                [current_car_info, OBSTACLES],
                forward_vector=forward_vector,
                formation_erosion=False,
            )
        except Exception:
            new_safe = np.array(current_car_info)[:, :2] if current_car_info else np.empty((0, 2))

        self.safe_regions.append(new_safe)
        if len(self.safe_regions) > 1:
            self.safe_regions.pop(0)

        next_idx = self.current_waypoint_index + 1
        next_center = self.waypoint_centers[next_idx]
        offsets = self.get_current_offsets()
        base_targets = expand_to_formation(next_center, offsets)
        active_positions = self.get_active_positions()

        optimized_targets = solve_optimization(
            new_safe,
            base_targets,
            active_positions,
            self.get_current_adj_matrix(),
            self.current_waypoint_index,
            self.get_current_shape_icon(),
            forward_vector,
        )
        optimized_targets = np.asarray(optimized_targets, dtype=float)
        if optimized_targets.shape != base_targets.shape:
            optimized_targets = base_targets

        # 5.1 insertion logic: if optimizer shifts centre too far, insert intermediate waypoint
        opt_center = np.mean(optimized_targets, axis=0)
        orig_center = np.mean(base_targets, axis=0)  # equals next_center
        if np.linalg.norm(opt_center - orig_center) > _OPTIMIZATION_THRESHOLD:
            self.waypoint_centers.insert(next_idx, opt_center.copy())
            if self.join_waypoint_index >= next_idx:
                self.join_waypoint_index += 1
            if self.dropout_waypoint_index >= next_idx:
                self.dropout_waypoint_index += 1

        self.current_waypoint_index = next_idx
        self.current_target_positions = optimized_targets
        self.target_index_by_vehicle_id = {
            vid: idx for idx, vid in enumerate(self.formation_vehicle_ids)
        }

    def handle_waypoint_arrival(self):
        current_center = np.mean(self.get_active_positions(), axis=0)
        self.completed_centers.append(current_center)

        if self.state == "join_wait":
            self.finalize_join()
            return

        if self.state == "four_car_cruise" and self.current_waypoint_index == self.join_waypoint_index:
            self.trigger_join_wait()
            return

        if self.state == "five_car_cruise" and self.current_waypoint_index == self.dropout_waypoint_index:
            self.trigger_dropout()
            return

        if self.current_waypoint_index < len(self.waypoint_centers) - 1:
            self._advance_to_next_waypoint()
            return

        self.navigation_complete = True
        self.state = "navigation_complete"

    def update_formation(self, delta_t=0.1):
        if self.navigation_complete:
            return

        if self.all_active_vehicles_arrived():
            self.handle_waypoint_arrival()
            return

        for vehicle_id in self.formation_vehicle_ids:
            vehicle = self.vehicles[vehicle_id]
            target = self.current_target_positions[self.target_index_by_vehicle_id[vehicle_id]]
            vehicle.update_movement(target, self.vehicles, delta_t)

        waiting_vehicle = self.vehicles[self.waiting_vehicle_id]
        if waiting_vehicle.frozen:
            waiting_vehicle.velocity = np.zeros(2)


def animation_update(_frame):
    if not controller.navigation_complete:
        controller.update_formation(delta_t=0.1)

    for plot, vehicle in zip(vehicle_plots, vehicles):
        plot.set_data([vehicle.position[0]], [vehicle.position[1]])

    status_text.set_text(
        f"当前任务航点: {controller.current_waypoint_index + 1}/{len(controller.waypoint_centers)}"
    )
    phase_text.set_text(f"当前阶段: {controller.get_phase_label()}")

    completed_centers_x = [center[0] for center in controller.completed_centers]
    completed_centers_y = [center[1] for center in controller.completed_centers]
    completed_centers_plot.set_data(completed_centers_x, completed_centers_y)

    if controller.safe_regions:
        safe_region_patch.set_xy(np.array(controller.safe_regions[-1]))
    else:
        safe_region_patch.set_xy(np.empty((0, 2)))

    if controller.navigation_complete:
        completion_text.set_text("导航完成")
        completion_text.set_color("red")
        completion_text.set_fontsize(16)
    else:
        completion_text.set_text("")

    return vehicle_plots + [
        status_text,
        phase_text,
        completed_centers_plot,
        safe_region_patch,
        completion_text,
    ]


if __name__ == "__main__":
    FORMATION_GEOMETRY = [
        (-0.5, -0.5, 0.1),
        (0.5, -0.5, 0.1),
        (-0.5, 0.5, 0.1),
        (0.5, 0.5, 0.1),
    ]
    FORMATION_GEOMETRY_2 = [
        (0.0, 0.0, 0.1),
        (-1.0, -1.0, 0.1),
        (0.0, -1.0, 0.1),
        (-1.0, 0.0, 0.1),
    ]
    FORMATION_GEOMETRY_PENTAGON = compute_regular_polygon_offsets(5, 0.85, 0.1)
    FORMATION_GEOMETRY_PENTAGON_2 = [(*offset[:2], 0.1) for offset in FORMATION_GEOMETRY_PENTAGON]

    OBSTACLES = [(10, 4, 4), (10, 16, 4), (5, 12, 2), (4, 9, 1)]
    start_point = (1, 19)
    goal_point = (18, 18)
    num_targets = 7

    MISSION_CENTERS = a_star_path(start_point, goal_point, num_targets, OBSTACLES, 0.5, 0)
    if not MISSION_CENTERS:
        MISSION_CENTERS = [np.array(start_point), np.array(goal_point)]
    else:
        MISSION_CENTERS = [np.array(wp) for wp in MISSION_CENTERS]
        MISSION_CENTERS.append(np.array(goal_point))

    wait_position = build_wait_position(MISSION_CENTERS[2], (1.5, 1.0))
    vehicles = [
        Car(
            id=i,
            init_pos=(start_point[0] + FORMATION_GEOMETRY[i][0], start_point[1] + FORMATION_GEOMETRY[i][1]),
            car_radius=FORMATION_GEOMETRY[i][2],
        )
        for i in range(4)
    ]
    vehicles.append(Car(id=4, init_pos=wait_position, car_radius=0.1))

    controller = FormationController(
        vehicles=vehicles,
        square_offsets=FORMATION_GEOMETRY,
        pentagon_offsets=FORMATION_GEOMETRY_PENTAGON,
        waypoint_centers=MISSION_CENTERS,
        waiting_vehicle_id=4,
        join_wait_offset=(1.5, 1.0),
        max_attempts=4,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 20), ylim=(0, 20))
    ax.set_title("基于A*路径的五车动态编队导航")

    waypoint_centers = [np.asarray(center, dtype=float) for center in MISSION_CENTERS[:-1]]
    if waypoint_centers:
        waypoints_x, waypoints_y = zip(*waypoint_centers)
        ax.scatter(waypoints_x, waypoints_y, marker="X", s=200, c="red", label="中间点")

    ax.scatter(start_point[0], start_point[1], marker="*", s=200, c="purple", label="起点")
    ax.scatter(goal_point[0], goal_point[1], marker="*", s=200, c="red", label="终点")

    obs_label_drawn = False
    for obs in OBSTACLES:
        center = (obs[0], obs[1])
        circle = plt.Circle(
            center,
            obs[2],
            color="red",
            fill=False,
            linestyle="--",
            label="障碍物" if not obs_label_drawn else None,
        )
        ax.add_patch(circle)
        obs_label_drawn = True

    colors = plt.cm.tab10(np.linspace(0, 1, len(vehicles)))
    vehicle_plots = [
        ax.plot([], [], "o", markersize=8, color=colors[i], label=f"小车{i + 1}")[0]
        for i in range(len(vehicles))
    ]
    ax.plot([], [], "o", markersize=8, color="black", label="小车集群")

    completed_centers_plot, = ax.plot([], [], "o", markersize=5, color="black", label="已完成中心点")
    initial_safe_region = (
        np.array(controller.safe_regions[-1])
        if controller.safe_regions and len(np.array(controller.safe_regions[-1])) >= 3
        else np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    )
    safe_region_patch = plt.Polygon(initial_safe_region, color="green", alpha=0.5, label="安全区域")
    ax.add_patch(safe_region_patch)

    status_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center")
    phase_text = ax.text(0.5, 0.91, "", transform=ax.transAxes, ha="center")
    completion_text = ax.text(0.5, 0.87, "", transform=ax.transAxes, ha="center")

    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1), borderaxespad=0.0)

    ani = FuncAnimation(fig, animation_update, frames=300, interval=50, blit=True)
    plt.show()
