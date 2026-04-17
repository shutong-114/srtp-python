import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


MODULE_PATH = pathlib.Path(__file__).with_name("srtp_main7.0.py")
OPTIMIZER_PATH = pathlib.Path(__file__).with_name("distributed_optimization_operator.py")


def load_module():
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_animation = types.ModuleType("matplotlib.animation")

    fake_pyplot.rcParams = {}
    fake_animation.FuncAnimation = object
    fake_matplotlib.pyplot = fake_pyplot
    fake_matplotlib.animation = fake_animation

    sys.modules.setdefault("matplotlib", fake_matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", fake_pyplot)
    sys.modules.setdefault("matplotlib.animation", fake_animation)

    spec = importlib.util.spec_from_file_location("srtp_main7_0", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_optimizer_module():
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.rcParams = {}
    fake_matplotlib.pyplot = fake_pyplot
    sys.modules.setdefault("matplotlib", fake_matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", fake_pyplot)

    spec = importlib.util.spec_from_file_location("distributed_optimization_operator_test", OPTIMIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SrtpMain70Tests(unittest.TestCase):
    def test_compute_event_indices_for_seven_targets(self):
        module = load_module()
        join_idx, dropout_idx = module.compute_event_indices(7)
        self.assertEqual(join_idx, 2)
        self.assertEqual(dropout_idx, 6)

    def test_initial_safe_region_is_built_from_active_vehicle_center(self):
        module = load_module()

        module.OBSTACLES = []
        module.MIN_DISTANCE = 1.5
        module.FORMATION_GEOMETRY = [
            (-0.5, -0.5, 0.1),
            (0.5, -0.5, 0.1),
            (-0.5, 0.5, 0.1),
            (0.5, 0.5, 0.1),
        ]
        module.FORMATION_GEOMETRY_2 = [
            (0.0, 0.0, 0.1),
            (-1.0, -1.0, 0.1),
            (0.0, -1.0, 0.1),
            (-1.0, 0.0, 0.1),
        ]
        module.FORMATION_GEOMETRY_PENTAGON = module.compute_regular_polygon_offsets(5, 0.85, 0.1)
        module.FORMATION_GEOMETRY_PENTAGON_2 = [(*offset[:2], 0.1) for offset in module.FORMATION_GEOMETRY_PENTAGON]
        recorded_centers = []

        def fake_safe_region(args, **_kwargs):
            recorded_centers.append(np.mean(np.array(args[0])[:, :2], axis=0))
            return np.array([
                [-5.0, -5.0],
                [5.0, -5.0],
                [5.0, 5.0],
                [-5.0, 5.0],
            ])

        module.safe_region = fake_safe_region
        module.solve_optimization = lambda _lf, target_positions, _initial, _adj, _idx, _icon, _forward: np.array(target_positions)

        waypoint_centers = [np.array([10.0, 10.0]), np.array([12.0, 10.0])]
        vehicles = [
            module.Car(i, (1.0 + module.FORMATION_GEOMETRY[i][0], 2.0 + module.FORMATION_GEOMETRY[i][1]), 0.1)
            for i in range(4)
        ]
        vehicles.append(module.Car(4, (0.0, 0.0), 0.1))
        controller = module.FormationController(
            vehicles=vehicles,
            square_offsets=module.FORMATION_GEOMETRY,
            pentagon_offsets=module.FORMATION_GEOMETRY_PENTAGON,
            waypoint_centers=waypoint_centers,
            waiting_vehicle_id=4,
            join_wait_offset=(1.0, 1.0),
        )

        self.assertTrue(
            np.allclose(
                recorded_centers[0],
                np.array([1.0, 2.0]),
                atol=1e-6,
            )
        )
        self.assertFalse(
            np.allclose(
                recorded_centers[0],
                waypoint_centers[0],
                atol=1e-6,
            )
        )

    def test_recompute_targets_builds_safe_region_around_current_active_center(self):
        module = load_module()

        module.OBSTACLES = []
        module.MIN_DISTANCE = 1.5
        module.FORMATION_GEOMETRY = [
            (-0.5, -0.5, 0.1),
            (0.5, -0.5, 0.1),
            (-0.5, 0.5, 0.1),
            (0.5, 0.5, 0.1),
        ]
        module.FORMATION_GEOMETRY_2 = [
            (0.0, 0.0, 0.1),
            (-1.0, -1.0, 0.1),
            (0.0, -1.0, 0.1),
            (-1.0, 0.0, 0.1),
        ]
        module.FORMATION_GEOMETRY_PENTAGON = module.compute_regular_polygon_offsets(5, 1.0, 0.1)
        module.FORMATION_GEOMETRY_PENTAGON_2 = [(*offset[:2], 0.1) for offset in module.FORMATION_GEOMETRY_PENTAGON]
        recorded_centers = []

        def fake_safe_region(args, **_kwargs):
            recorded_centers.append(np.mean(np.array(args[0])[:, :2], axis=0))
            return np.array([
                [-20.0, -20.0],
                [20.0, -20.0],
                [20.0, 20.0],
                [-20.0, 20.0],
            ])

        module.safe_region = fake_safe_region
        module.solve_optimization = lambda _lf, target_positions, _initial, _adj, _idx, _icon, _forward: np.array(target_positions)

        waypoint_centers = [np.array([0.0, 0.0]), np.array([8.0, 0.0])]
        vehicles = [module.Car(i, (0.0, 0.0), 0.1) for i in range(5)]
        controller = module.FormationController(
            vehicles=vehicles,
            square_offsets=module.FORMATION_GEOMETRY,
            pentagon_offsets=module.FORMATION_GEOMETRY_PENTAGON,
            waypoint_centers=waypoint_centers,
            waiting_vehicle_id=4,
            join_wait_offset=(1.0, 1.0),
        )

        shifted_center = np.array([6.0, 4.0])
        for idx, vehicle_id in enumerate(controller.formation_vehicle_ids):
            vehicles[vehicle_id].position = shifted_center + np.array(module.FORMATION_GEOMETRY[idx][:2])

        controller.recompute_current_targets()

        self.assertTrue(
            np.allclose(
                recorded_centers[-1],
                shifted_center,
                atol=1e-6,
            )
        )

    def test_optimizer_enforces_min_pairwise_distance_of_one(self):
        optimizer_module = load_optimizer_module()

        optimizer = optimizer_module.DistributedFormationOptimizer(
            np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]),
            sigma=0.1,
            max_iter=1,
            tol=1e-6,
        )

        updates = {
            1: {"p1": np.array([0.0, 0.0]), "p12": np.array([0.0, 0.0]), "y1": np.zeros(2), "z1": np.zeros(2), "w12": np.zeros(2)},
            2: {"p2": np.array([0.2, 0.0]), "p21": np.array([0.2, 0.0]), "y2": np.zeros(2), "z21": np.zeros(2), "w2": np.zeros(2)},
            3: {"p_i": np.array([0.1, 0.1]), "p_i1": np.array([0.1, 0.1]), "p_i2": np.array([0.1, 0.1]), "y_i": np.zeros(2), "z_i1": np.zeros(2), "w_i2": np.zeros(2)},
            4: {"p_i": np.array([0.15, 0.05]), "p_i1": np.array([0.15, 0.05]), "p_i2": np.array([0.15, 0.05]), "y_i": np.zeros(2), "z_i1": np.zeros(2), "w_i2": np.zeros(2)},
        }

        adjusted = optimizer._enforce_min_pairwise_distance(
            updates,
            min_distance=1.0,
            LF_vertices=np.array([
                [-5.0, -5.0],
                [5.0, -5.0],
                [5.0, 5.0],
                [-5.0, 5.0],
            ]),
        )

        positions = np.array([
            adjusted[1]["p1"],
            adjusted[2]["p2"],
            adjusted[3]["p_i"],
            adjusted[4]["p_i"],
        ])
        min_distance = min(
            np.linalg.norm(positions[i] - positions[j])
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )
        self.assertGreaterEqual(min_distance, 0.999)

    def test_controller_transitions_join_and_dropout(self):
        module = load_module()

        module.OBSTACLES = []
        module.MIN_DISTANCE = 1.5
        module.FORMATION_GEOMETRY = [
            (-0.5, -0.5, 0.1),
            (0.5, -0.5, 0.1),
            (-0.5, 0.5, 0.1),
            (0.5, 0.5, 0.1),
        ]
        module.FORMATION_GEOMETRY_2 = [
            (0.0, 0.0, 0.1),
            (-1.0, -1.0, 0.1),
            (0.0, -1.0, 0.1),
            (-1.0, 0.0, 0.1),
        ]
        module.FORMATION_GEOMETRY_PENTAGON = module.compute_regular_polygon_offsets(5, 0.85, 0.1)
        module.FORMATION_GEOMETRY_PENTAGON_2 = [(*offset[:2], 0.1) for offset in module.FORMATION_GEOMETRY_PENTAGON]
        module.ADJ_MATRIX_4 = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ])
        module.ADJ_MATRIX_5 = np.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ])
        module.safe_region = lambda *_args, **_kwargs: np.array([
            [-5.0, -5.0],
            [5.0, -5.0],
            [5.0, 5.0],
            [-5.0, 5.0],
        ])
        module.solve_optimization = lambda _lf, target_positions, _initial, _adj, _idx, _icon, _forward: np.array(target_positions)

        waypoint_centers = [np.array([float(i), 0.0]) for i in range(8)]
        vehicles = [
            module.Car(i, (0.0, 0.0), 0.1)
            for i in range(5)
        ]

        controller = module.FormationController(
            vehicles=vehicles,
            square_offsets=module.FORMATION_GEOMETRY,
            pentagon_offsets=module.FORMATION_GEOMETRY_PENTAGON,
            waypoint_centers=waypoint_centers,
            waiting_vehicle_id=4,
            join_wait_offset=(1.0, 1.0),
        )

        self.assertEqual(controller.state, "four_car_cruise")
        self.assertFalse(vehicles[4].active_in_formation)
        self.assertTrue(vehicles[4].frozen)

        controller.current_waypoint_index = controller.join_waypoint_index
        controller.current_target_positions = module.expand_to_formation(
            waypoint_centers[controller.join_waypoint_index], module.FORMATION_GEOMETRY
        )
        for vehicle_id in controller.formation_vehicle_ids:
            vehicles[vehicle_id].position = np.array(controller.current_target_positions[controller.target_index_by_vehicle_id[vehicle_id]])
        controller.update_formation(delta_t=0.1)

        self.assertEqual(controller.state, "join_wait")
        self.assertIn(4, controller.formation_vehicle_ids)
        self.assertTrue(vehicles[4].active_in_formation)
        self.assertFalse(vehicles[4].frozen)

        for vehicle_id in controller.formation_vehicle_ids:
            vehicles[vehicle_id].position = np.array(controller.current_target_positions[controller.target_index_by_vehicle_id[vehicle_id]])
        controller.update_formation(delta_t=0.1)

        self.assertEqual(controller.state, "five_car_cruise")
        self.assertEqual(controller.current_waypoint_index, controller.join_waypoint_index + 1)

        controller.current_waypoint_index = controller.dropout_waypoint_index
        controller.state = "five_car_cruise"
        controller.recompute_current_targets()
        for vehicle_id in controller.formation_vehicle_ids:
            vehicles[vehicle_id].position = np.array(controller.current_target_positions[controller.target_index_by_vehicle_id[vehicle_id]])
        dropout_position = vehicles[4].position.copy()
        controller.update_formation(delta_t=0.1)

        self.assertEqual(controller.state, "dropout_done")
        self.assertEqual(controller.formation_vehicle_ids, [0, 1, 2, 3])
        self.assertTrue(vehicles[4].frozen)
        self.assertFalse(vehicles[4].active_in_formation)
        self.assertTrue(np.allclose(vehicles[4].position, dropout_position))

    def test_waypoint_advances_when_shifted_optimized_targets_are_reached(self):
        module = load_module()

        module.OBSTACLES = []
        module.MIN_DISTANCE = 1.5
        module.FORMATION_GEOMETRY = [
            (-0.5, -0.5, 0.1),
            (0.5, -0.5, 0.1),
            (-0.5, 0.5, 0.1),
            (0.5, 0.5, 0.1),
        ]
        module.FORMATION_GEOMETRY_2 = [
            (0.0, 0.0, 0.1),
            (-1.0, -1.0, 0.1),
            (0.0, -1.0, 0.1),
            (-1.0, 0.0, 0.1),
        ]
        module.FORMATION_GEOMETRY_PENTAGON = module.compute_regular_polygon_offsets(5, 0.85, 0.1)
        module.FORMATION_GEOMETRY_PENTAGON_2 = [(*offset[:2], 0.1) for offset in module.FORMATION_GEOMETRY_PENTAGON]
        module.safe_region = lambda *_args, **_kwargs: np.array([
            [-5.0, -5.0],
            [5.0, -5.0],
            [5.0, 5.0],
            [-5.0, 5.0],
        ])
        module.solve_optimization = (
            lambda _lf, target_positions, _initial, _adj, _idx, _icon, _forward:
            np.array(target_positions) + np.array([2.0, 0.0])
        )

        waypoint_centers = [np.array([float(i), 0.0]) for i in range(4)]
        vehicles = [module.Car(i, (0.0, 0.0), 0.1) for i in range(5)]
        controller = module.FormationController(
            vehicles=vehicles,
            square_offsets=module.FORMATION_GEOMETRY,
            pentagon_offsets=module.FORMATION_GEOMETRY_PENTAGON,
            waypoint_centers=waypoint_centers,
            waiting_vehicle_id=4,
            join_wait_offset=(1.0, 1.0),
        )

        optimized_targets = controller.current_target_positions.copy()
        for vehicle_id in controller.formation_vehicle_ids:
            vehicles[vehicle_id].position = np.array(
                optimized_targets[controller.target_index_by_vehicle_id[vehicle_id]]
            )

        controller.update_formation(delta_t=0.1)

        self.assertEqual(controller.current_waypoint_index, 1)
        self.assertEqual(controller.state, "four_car_cruise")

    def test_join_wait_continues_to_move_vehicles_until_merged(self):
        module = load_module()

        module.OBSTACLES = []
        module.MIN_DISTANCE = 1.5
        module.FORMATION_GEOMETRY = [
            (-0.5, -0.5, 0.1),
            (0.5, -0.5, 0.1),
            (-0.5, 0.5, 0.1),
            (0.5, 0.5, 0.1),
        ]
        module.FORMATION_GEOMETRY_2 = [
            (0.0, 0.0, 0.1),
            (-1.0, -1.0, 0.1),
            (0.0, -1.0, 0.1),
            (-1.0, 0.0, 0.1),
        ]
        module.FORMATION_GEOMETRY_PENTAGON = module.compute_regular_polygon_offsets(5, 0.85, 0.1)
        module.FORMATION_GEOMETRY_PENTAGON_2 = [(*offset[:2], 0.1) for offset in module.FORMATION_GEOMETRY_PENTAGON]
        module.safe_region = lambda *_args, **_kwargs: np.array([
            [-5.0, -5.0],
            [5.0, -5.0],
            [5.0, 5.0],
            [-5.0, 5.0],
        ])
        module.solve_optimization = lambda _lf, target_positions, _initial, _adj, _idx, _icon, _forward: np.array(target_positions)

        waypoint_centers = [np.array([float(i), 0.0]) for i in range(8)]
        vehicles = [module.Car(i, (0.0, 0.0), 0.1) for i in range(5)]
        controller = module.FormationController(
            vehicles=vehicles,
            square_offsets=module.FORMATION_GEOMETRY,
            pentagon_offsets=module.FORMATION_GEOMETRY_PENTAGON,
            waypoint_centers=waypoint_centers,
            waiting_vehicle_id=4,
            join_wait_offset=(1.0, 1.0),
        )

        controller.current_waypoint_index = controller.join_waypoint_index
        controller.current_target_positions = module.expand_to_formation(
            waypoint_centers[controller.join_waypoint_index], module.FORMATION_GEOMETRY
        )
        controller.target_index_by_vehicle_id = {vid: idx for idx, vid in enumerate(controller.formation_vehicle_ids)}
        for vehicle_id in controller.formation_vehicle_ids:
            vehicles[vehicle_id].position = np.array(
                controller.current_target_positions[controller.target_index_by_vehicle_id[vehicle_id]]
            )

        controller.update_formation(delta_t=0.1)
        self.assertEqual(controller.state, "join_wait")

        before_positions = [vehicles[vid].position.copy() for vid in controller.formation_vehicle_ids]
        controller.update_formation(delta_t=0.1)
        after_positions = [vehicles[vid].position.copy() for vid in controller.formation_vehicle_ids]

        self.assertTrue(any(not np.allclose(before, after) for before, after in zip(before_positions, after_positions)))

    def test_full_navigation_completes_and_marks_each_waypoint_once(self):
        module = load_module()

        module.OBSTACLES = []
        module.MIN_DISTANCE = 1.5
        module.FORMATION_GEOMETRY = [
            (-0.5, -0.5, 0.1),
            (0.5, -0.5, 0.1),
            (-0.5, 0.5, 0.1),
            (0.5, 0.5, 0.1),
        ]
        module.FORMATION_GEOMETRY_2 = [
            (0.0, 0.0, 0.1),
            (-1.0, -1.0, 0.1),
            (0.0, -1.0, 0.1),
            (-1.0, 0.0, 0.1),
        ]
        module.FORMATION_GEOMETRY_PENTAGON = module.compute_regular_polygon_offsets(5, 0.85, 0.1)
        module.FORMATION_GEOMETRY_PENTAGON_2 = [(*offset[:2], 0.1) for offset in module.FORMATION_GEOMETRY_PENTAGON]
        module.ADJ_MATRIX_4 = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ])
        module.ADJ_MATRIX_5 = np.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ])
        module.safe_region = lambda *_args, **_kwargs: np.array([
            [-50.0, -50.0],
            [50.0, -50.0],
            [50.0, 50.0],
            [-50.0, 50.0],
        ])
        module.solve_optimization = lambda _lf, target_positions, _initial, _adj, _idx, _icon, _forward: np.array(target_positions)

        waypoint_centers = [np.array([float(i), 0.0]) for i in range(8)]
        vehicles = [
            module.Car(i, (0.0 + module.FORMATION_GEOMETRY[i][0], 0.0 + module.FORMATION_GEOMETRY[i][1]), 0.1)
            for i in range(4)
        ]
        vehicles.append(module.Car(4, module.build_wait_position(waypoint_centers[2], (1.0, 1.0)), 0.1))

        controller = module.FormationController(
            vehicles=vehicles,
            square_offsets=module.FORMATION_GEOMETRY,
            pentagon_offsets=module.FORMATION_GEOMETRY_PENTAGON,
            waypoint_centers=waypoint_centers,
            waiting_vehicle_id=4,
            join_wait_offset=(1.0, 1.0),
        )

        for _ in range(1500):
            controller.update_formation(delta_t=0.1)
            if controller.navigation_complete:
                break

        self.assertTrue(controller.navigation_complete)
        self.assertEqual(controller.state, "navigation_complete")
        self.assertEqual(len(controller.completed_waypoint_indices), len(waypoint_centers))
        self.assertEqual(controller.completed_waypoint_indices, list(range(len(waypoint_centers))))
        self.assertEqual(len(controller.completed_centers), len(waypoint_centers))


if __name__ == "__main__":
    unittest.main()
