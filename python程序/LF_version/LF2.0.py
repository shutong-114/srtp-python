import math
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，确保中文显示（如标题、图例中的中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 可改为 ['Microsoft YaHei'] 等支持中文的字体
plt.rcParams['axes.unicode_minus'] = False

def compute_car_circle(car_points):
    """
    根据小车各顶点计算几何中心及最大半径（小车整体半径）
    """
    n = len(car_points)
    cx = sum(p[0] for p in car_points) / n
    cy = sum(p[1] for p in car_points) / n
    center = np.array([cx, cy])
    r = max(math.hypot(p[0]-cx, p[1]-cy) for p in car_points)
    return center, r

def point_to_segment_distance(point, P1, P2):
    """
    计算点到线段 P1-P2 的最小距离和最近点
    """
    P1, P2, point = np.array(P1), np.array(P2), np.array(point)
    edge = P2 - P1
    length_squared = np.dot(edge, edge)
    if length_squared == 0:
        return np.linalg.norm(point - P1), P1
    t = np.dot(point - P1, edge) / length_squared
    if t < 0:
        closest = P1
    elif t > 1:
        closest = P2
    else:
        closest = P1 + t * edge
    return np.linalg.norm(point - closest), closest

def clip_polygon(polygon, inside, compute_intersection):
    """
    Sutherland-Hodgman 多边形裁剪算法
    polygon: 顶点列表，每个顶点为 (x,y)
    inside: 判断一个点是否在半平面内的函数
    compute_intersection: 计算边界交点的函数，参数为两个顶点
    返回裁剪后的多边形顶点列表
    """
    if not polygon:
        return []
    new_poly = []
    n = len(polygon)
    for i in range(n):
        curr = polygon[i]
        prev = polygon[i-1]
        curr_inside = inside(curr)
        prev_inside = inside(prev)
        if prev_inside and curr_inside:
            new_poly.append(curr)
        elif prev_inside and not curr_inside:
            new_poly.append(compute_intersection(prev, curr))
        elif not prev_inside and curr_inside:
            new_poly.append(compute_intersection(prev, curr))
            new_poly.append(curr)
    return new_poly

def clip_with_line(polygon, d, c):
    """
    对多边形用直线 p·d = c 裁剪，保留满足 p·d <= c 的部分
    d: 法向量
    c: 半平面偏移量
    """
    def inside(p):
        return (p[0]*d[0] + p[1]*d[1]) <= c + 1e-9
    def compute_intersection(p1, p2):
        dp = (p2[0]-p1[0], p2[1]-p1[1])
        denom = dp[0]*d[0] + dp[1]*d[1]
        if abs(denom) < 1e-9:
            return p1
        t = (c - (p1[0]*d[0] + p1[1]*d[1])) / denom
        return (p1[0] + t*dp[0], p1[1] + t*dp[1])
    return clip_polygon(polygon, inside, compute_intersection)

def clip_with_boundary(polygon, boundary):
    """
    按矩形边界裁剪多边形
    boundary: 字典，包含 'axis' ('x' 或 'y')、'val'（数值）和 'ineq' ('ge'或'le')
    """
    axis = 0 if boundary['axis']=='x' else 1
    val = boundary['val']
    ineq = boundary['ineq']
    def inside(p):
        if ineq=='ge':
            return p[axis] >= val - 1e-9
        else:
            return p[axis] <= val + 1e-9
    def compute_intersection(p1, p2):
        if abs(p2[axis]-p1[axis]) < 1e-9:
            return p1
        t = (val - p1[axis]) / (p2[axis]-p1[axis])
        return (p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1]))
    return clip_polygon(polygon, inside, compute_intersection)

def sort_polygon_vertices(poly):
    """
    根据多边形顶点质心对顶点按顺时针顺序排序
    """
    if not poly:
        return poly
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    poly_sorted = sorted(poly, key=lambda p: -math.atan2(p[1]-cy, p[0]-cx))
    return poly_sorted

def clip_polygon_offset(polygon, n, thresh):
    """
    对多边形用半平面 n·p >= thresh 裁剪
    n: 内法单位向量
    thresh: 阈值
    """
    def inside(p):
        return (p[0]*n[0] + p[1]*n[1]) >= thresh - 1e-9
    def compute_intersection(p1, p2):
        dp = (p2[0]-p1[0], p2[1]-p1[1])
        denom = dp[0]*n[0] + dp[1]*n[1]
        if abs(denom) < 1e-9:
            return p1
        t = (thresh - (p1[0]*n[0] + p1[1]*n[1])) / denom
        return (p1[0] + t*dp[0], p1[1] + t*dp[1])
    return clip_polygon(polygon, inside, compute_intersection)

def erode_polygon(polygon, car_center, r, car_radiaus=0):
    """
    对凸多边形整体内缩 r 距离（各边沿内法方向平移 r）
    """
    halfplanes = []
    n = len(polygon)
    for i in range(n):
        P = polygon[i]
        Q = polygon[(i+1)%n]
        mid = ((P[0]+Q[0])/2, (P[1]+Q[1])/2)
        edge = (Q[0]-P[0], Q[1]-P[1])
        cand1 = (edge[1], -edge[0])
        cand2 = (-edge[1], edge[0])
        if (car_center[0]-mid[0])*cand1[0] + (car_center[1]-mid[1])*cand1[1] > 0:
            n_vec = cand1
        else:
            n_vec = cand2
        norm = math.hypot(n_vec[0], n_vec[1])
        if norm == 0:
            continue
        n_unit = (n_vec[0]/norm, n_vec[1]/norm)
        thresh = n_unit[0]*P[0] + n_unit[1]*P[1] + r + car_radiaus
        halfplanes.append((n_unit, thresh))
    
    new_poly = [(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)]
    for n_unit, thresh in halfplanes:
        new_poly = clip_polygon_offset(new_poly, n_unit, thresh)
        if not new_poly:
            break
    return new_poly

def clip_with_convex_polygon(polygon, obstacle_poly, car_center, r_c):
    """
    使用最近的凸多边形边裁剪安全区域，并向外偏移 r_c
    对于障碍物的每条边，先计算小车到该边的实际线段距离，
    选择距离最近的边(或边组)后，将边按法向量向外偏移 r_c 后裁剪。
    """
    min_dist = float('inf')
    selected_edges = []
    n = len(obstacle_poly)
    for i in range(n):
        P1 = np.array(obstacle_poly[i])
        P2 = np.array(obstacle_poly[(i+1)%n])
        dist, closest_point = point_to_segment_distance(car_center, P1, P2)
        edge = P2 - P1
        normal_1 = np.array([edge[1], -edge[0]])
        normal_2 = -normal_1
        if np.dot(normal_1, car_center - closest_point) > 0:
            normal = normal_1
        else:
            normal = normal_2
        normal = normal / np.linalg.norm(normal)
        P1_offset = P1 + normal * r_c
        c_offset = np.dot(P1_offset, normal)
        if dist < min_dist - 1e-6:
            min_dist = dist
            selected_edges = [(normal, c_offset)]
        elif abs(dist - min_dist) < 1e-6:
            selected_edges.append((normal, c_offset))
    for normal, c in selected_edges:
        polygon = clip_with_line(polygon, normal, c)
        if not polygon:
            return []
    return polygon

def safe_region(params):
    """
    params: 列表，包含两个部分：
       params[0]: 小车顶点坐标列表，例如 [(x,y), ...]
       params[1]: 障碍物列表，每个障碍物为三元组 (ox, oy, r)
    返回：经过障碍物裁剪后，再向内侵蚀小车整体半径距离的安全区域顶点，
          顶点按顺时针顺序排列。
    """
    car_points = params[0]
    obstacles = params[1]
    car_center, r_c = compute_car_circle(car_points)

    polygon = [(0,0), (20,0), (20,20), (0,20)]  # 初始安全区域（地图边界）

    for obs in obstacles:
        if isinstance(obs, tuple) and len(obs) == 3:  # 处理圆形障碍物
            B = np.array([obs[0], obs[1]])
            obs_radius = obs[2]
            A = car_center

            dist2 = np.sum((B - A) ** 2)  # 计算距离的平方
            t = 1 - obs_radius / np.sqrt(dist2)  # 修正 math.sqrt() 为 np.sqrt()

            M_prime = A + t * (B - A)
            d = B - A
            c_val = np.dot(M_prime, d)
            polygon = clip_with_line(polygon, d, c_val)

        elif isinstance(obs, list) and len(obs) > 2:  # 处理多边形障碍物
            polygon = clip_with_convex_polygon(polygon, obs, car_center, r_c)

    # 裁剪安全区域，确保它不会超出地图边界
    boundaries = [
        {'axis':'x', 'val':0, 'ineq':'ge'},
        {'axis':'x', 'val':20, 'ineq':'le'},
        {'axis':'y', 'val':0, 'ineq':'ge'},
        {'axis':'y', 'val':20, 'ineq':'le'},
    ]
    for b in boundaries:
        polygon = clip_with_boundary(polygon, b)

    # 排序顶点
    polygon = sort_polygon_vertices(polygon)

    # 进一步缩小安全区域，考虑小车半径
    eroded_polygon = erode_polygon(polygon, car_center, r_c)
    eroded_polygon = sort_polygon_vertices(eroded_polygon)

    return eroded_polygon

if __name__ == "__main__":
    # 小车顶点及障碍物信息
    car_points = [(0,0), (1,0), (1,1), (0,1)]
    # 障碍物列表既包含圆形也包含多边形
    obstacles = [
        (10, 4, 4),  # 圆形障碍物
        [(0, 10), (2, 10), (2, 12), (0, 12)]  # 矩形障碍物（凸多边形）
    ]
    
    safe_poly = safe_region([car_points, obstacles])
    print("安全区域顶点（顺时针）:")
    for pt in safe_poly:
        print(pt)
    
    # 绘制图形直接在 __main__ 部分
    fig, ax = plt.subplots()
    if safe_poly:
        xs = [p[0] for p in safe_poly] + [safe_poly[0][0]]
        ys = [p[1] for p in safe_poly] + [safe_poly[0][1]]
        ax.fill(xs, ys, color='green', alpha=0.5, label='安全区域')
        ax.plot(xs, ys, color='black')
    
    # 绘制障碍物：支持圆形和多边形
    obs_label_drawn = False
    for obs in obstacles:
        if isinstance(obs, tuple) and len(obs) == 3:
            center = (obs[0], obs[1])
            circle = plt.Circle(center, obs[2], color='red', fill=False, linestyle='--',
                                label='障碍物' if not obs_label_drawn else None)
            ax.add_patch(circle)
            obs_label_drawn = True
        elif isinstance(obs, list):
            poly = obs + [obs[0]]
            ax.plot([p[0] for p in poly], [p[1] for p in poly], 'r--', label='障碍物' if not obs_label_drawn else None)
            obs_label_drawn = True
    
    # 绘制小车初始位置
    car_x = [p[0] for p in car_points] + [car_points[0][0]]
    car_y = [p[1] for p in car_points] + [car_points[0][1]]
    ax.plot(car_x, car_y, color='blue', label='小车位置')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax.set_xlim(0,20)
    ax.set_ylim(0,20)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('小车安全区域（绿色填充）')
    plt.grid(True)
    plt.show()
