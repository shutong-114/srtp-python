import math
from heapq import heappush, heappop

grid_size = 1.0  # 支持浮点数
grid_boundary = ((0, 20), (0, 20))  # 物理边界

def a_star_path(start, goal, num_targets, obstacles, formation_radius=0, vehicle_radius=0):
    """生成路径并采样中间目标点"""
    # 动态计算网格边界
    (x_min_phys, x_max_phys), (y_min_phys, y_max_phys) = grid_boundary
    x_min_grid = math.floor(x_min_phys / grid_size)
    x_max_grid = math.ceil(x_max_phys / grid_size)
    y_min_grid = math.floor(y_min_phys / grid_size)
    y_max_grid = math.ceil(y_max_phys / grid_size)

    # 转换物理坐标到网格坐标（整数）
    start_grid = (int(math.floor(start[0] / grid_size)), int(math.floor(start[1] / grid_size)))
    goal_grid = (int(math.floor(goal[0] / grid_size)), int(math.floor(goal[1] / grid_size)))

    # 检查网格坐标是否越界
    if not (x_min_grid <= start_grid[0] < x_max_grid and y_min_grid <= start_grid[1] < y_max_grid):
        return []
    if not (x_min_grid <= goal_grid[0] < x_max_grid and y_min_grid <= goal_grid[1] < y_max_grid):
        return []

    # 调用A*算法
    path_grid = a_star(start_grid, goal_grid, obstacles, formation_radius, vehicle_radius)
    if not path_grid:
        return []

    # 网格坐标转物理坐标（中心点）
    path_continuous = [
        (x * grid_size + grid_size/2, y * grid_size + grid_size/2)
        for (x, y) in path_grid
    ]
    return sample_path(path_continuous, num_targets)

def a_star(start, goal, obstacles, formation_radius, vehicle_radius):
    """动态网格边界的A*实现"""
    # 计算动态网格边界
    (x_min_phys, x_max_phys), (y_min_phys, y_max_phys) = grid_boundary
    x_min_grid = math.floor(x_min_phys / grid_size)
    x_max_grid = math.ceil(x_max_phys / grid_size)
    y_min_grid = math.floor(y_min_phys / grid_size)
    y_max_grid = math.ceil(y_max_phys / grid_size)

    open_heap = []
    heappush(open_heap, (0, start[0], start[1]))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()

    while open_heap:
        current_f, cx, cy = heappop(open_heap)
        current = (cx, cy)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current)
        # 8邻域探索
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (cx + dx, cy + dy)
                # 动态边界检查
                if not (x_min_grid <= neighbor[0] < x_max_grid and y_min_grid <= neighbor[1] < y_max_grid):
                    continue
                if is_blocked(neighbor[0], neighbor[1], obstacles, formation_radius, vehicle_radius):
                    continue
                # 物理距离计算
                move_cost = math.hypot(dx*grid_size, dy*grid_size)
                tentative_g = g_score.get(current, float('inf')) + move_cost
                if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heappush(open_heap, (f, neighbor[0], neighbor[1]))
    return []

def heuristic(a, b):
    """考虑grid_size的启发式函数"""
    return math.hypot((a[0]-b[0])*grid_size, (a[1]-b[1])*grid_size)

def is_blocked(x_grid, y_grid, obstacles, formation_radius, vehicle_radius):
    """动态计算网格中心物理坐标"""
    cx = x_grid * grid_size + grid_size/2
    cy = y_grid * grid_size + grid_size/2
    for ox, oy, r in obstacles:
        dx = cx - ox
        dy = cy - oy
        margin = r + formation_radius + vehicle_radius
        if dx**2 + dy**2 <= margin**2:
            return True
    return False

def sample_path(path, num_targets):
    """采样逻辑保持不变"""
    if len(path) < 2 or num_targets <= 0:
        return []
    cumulative = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        cumulative.append(cumulative[-1] + math.hypot(dx, dy))
    total = cumulative[-1]
    if total == 0:
        return []
    interval = total / (num_targets + 1)
    targets = []
    current_dist = interval
    current_seg = 0
    while len(targets) < num_targets and current_dist < total:
        while current_seg < len(cumulative)-1 and cumulative[current_seg+1] < current_dist:
            current_seg += 1
        seg_start = cumulative[current_seg]
        seg_end = cumulative[current_seg+1]
        seg_length = seg_end - seg_start
        if seg_length == 0:
            continue
        t = (current_dist - seg_start) / seg_length
        x = path[current_seg][0] + t*(path[current_seg+1][0] - path[current_seg][0])
        y = path[current_seg][1] + t*(path[current_seg+1][1] - path[current_seg][1])
        targets.append((x, y))
        current_dist += interval
    return targets[:num_targets]