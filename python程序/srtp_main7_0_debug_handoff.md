# srtp_main7.0 Debug Handoff

## Current Goal
- Keep waypoint progression sequential: no inserting extra intermediate waypoints.
- Still use optimization to compute formation targets for the current waypoint.
- Use square reference shape for 4-car phase and regular pentagon reference shape for 5-car phase.
- Robot 5 waits near waypoint 3, joins at waypoint 3, drops out at the penultimate waypoint.

## Root Causes Found So Far

### 1. Stuck at second waypoint
- Symptom: controller stayed on waypoint index 1 forever.
- Root cause: arrival check used the original mission waypoint center, while the optimizer had shifted the actual current target center elsewhere.
- Fix applied: `has_reached_current_waypoint_center()` now checks the center of `self.current_target_positions`, which matches the original `srtp_main` semantics where `self.waypoints[self.current_waypoint_index]` is the optimized current waypoint target.

### 2. Stuck in `join_wait`
- Symptom: after entering `join_wait`, all cars froze and never merged.
- Root cause: `update_formation()` checked `all_active_vehicles_arrived()` inside the `join_wait` branch, but if that returned `False`, no vehicle movement happened in that branch.
- Fix applied: `update_formation()` now keeps moving vehicles during `join_wait` unless the merge has completed in that frame.

## Important Files
- Main logic: `D:\\SRTP\\2024srtp\\python程序\\srtp_main7.0.py`
- Regression tests: `D:\\SRTP\\2024srtp\\python程序\\test_srtp_main7_0.py`
- Optimizer interface already supports reference shape injection:
  `D:\\SRTP\\2024srtp\\python程序\\distributed_optimization_operator.py`

## Important Code Locations
- Optimizer tuning constants near top of `srtp_main7.0.py`
- Optimized target generation in `recompute_current_targets()`
- Arrival logic in `has_reached_current_waypoint_center()`
- Join/drop state transitions in `trigger_join_wait()`, `finalize_join()`, `trigger_dropout()`
- Main state machine in `update_formation()`

## Tests Added
- Event index calculation
- Join and dropout state transitions
- Waypoint advances when optimized target center is reached
- Join-wait phase continues moving until merge completes

## Useful Debugging Pattern
- Run headless simulation and log:
  - `current_waypoint_index`
  - `state`
  - active vehicle ids
  - active formation center
  - `current_target_positions.mean(axis=0)`
  - each car's distance to its assigned target
- This quickly distinguishes:
  - bad arrival logic
  - frozen state-machine branch
  - optimizer producing unreachable target geometry

## Next Likely Risk Areas
- After merge completes, check whether the first 5-car cruise waypoint advances smoothly.
- After dropout, verify robot 5 remains a static obstacle while 4-car formation resumes without deadlock.
- If five-car geometry still looks visibly non-pentagonal, tune:
  - `OPTIMIZER_SIGMA`
  - `OPTIMIZER_MAX_ITER`
  - `OPTIMIZER_TOL`
  - pentagon radius in `compute_regular_polygon_offsets(5, radius, ...)`
