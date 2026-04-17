#coding:utf-8
'''
更新内容：
1.建立对凸多边形的LF计算
2.更新小车信息为（x, y, r）三元组
3.完善安全区域裁切机制，并为是否考虑编队半径提供开关formation_erosion

0326更新: 修改凸多边形裁剪边选取逻辑, 并适配
'''

import math
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，确保中文显示（如标题、图例中的中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 等
plt.rcParams['axes.unicode_minus'] = False

def compute_car_circle(car_points):
    """
    根据小车各顶点计算几何中心及编队外接圆半径（formation radius）。
    对于扩展顶点 (x, y, r)，公式为：
      formation_radius = max( math.hypot(p[0]-cx, p[1]-cy) + p[2] for p in car_points )
    其中 (cx, cy) 为几何中心。
    """
    n = len(car_points)
    cx = sum(p[0] for p in car_points) / n
    cy = sum(p[1] for p in car_points) / n
    center = np.array([cx, cy])
    formation_radius = max(math.hypot(p[0]-cx, p[1]-cy) + p[2] for p in car_points)
    return center, formation_radius

def point_to_segment_distance(point, P1, P2):
    """
    计算点到线段 P1-P2 的最小距离及最近点
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
    polygon: 顶点列表，每个顶点为 (x, y)
    inside: 判断点是否在半平面内的函数
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
        return (p[0]*d[0] + p[1]*d[1]) >= c + 1e-9
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
    boundary: 字典，包含 'axis' ('x' 或 'y')、'val'（数值）和 'ineq' ('ge' 或 'le')
    """
    axis = 0 if boundary['axis'] == 'x' else 1
    val = boundary['val']
    ineq = boundary['ineq']
    def inside(p):
        if ineq == 'ge':
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

def clip_with_convex_polygon(polygon, obstacle_poly, car_center, erosion_radius, forward_vector=(1,0)):
    """
    使用最近的凸多边形边裁剪安全区域，并向外平移 erosion_radius。
    对于障碍物的每条边，先计算小车到该边的实际线段距离，
    选择距离最近的边；当出现距离相同时，取与 forward_vector 内积绝对值最大的边，
    如果仍然相同，则默认保留先计算的一条。
    """
    tol = 1e-6
    min_dist = float('inf')
    # 改为仅保存一个候选边，格式为 (normal, c_offset, dot_val)
    candidate = None
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
        P1_offset = P1 + normal * erosion_radius
        c_offset = np.dot(P1_offset, normal)
        if dist < min_dist - tol:
            min_dist = dist
            candidate = (normal, c_offset, abs(np.dot(normal, forward_vector)))
        elif abs(dist - min_dist) < tol:
            cur_dot = abs(np.dot(normal, forward_vector))
            # 如果当前边的内积绝对值大，则更新候选边
            if cur_dot < candidate[2]:
                candidate = (normal, c_offset, cur_dot)
            # 否则保持原候选（先计算的）
    if candidate is None:
        return polygon
    normal, c_offset, _ = candidate
    polygon = clip_with_line(polygon, normal, c_offset)
    return polygon


def safe_region(params, forward_vector:tuple = (1,0), formation_erosion=False):
    """
    params: 列表，包含以下部分：
       params[0]: 小车顶点列表，格式为 [(x, y, r), ...]，其中 r 为小车自身半径（各顶点可独立设置）
       params[1]: 障碍物列表，每个障碍物为：
                  - 圆形：(ox, oy, r)
                  - 多边形：顶点列表 [(x, y), ...]
    
    formation_erosion: 布尔值，是否启用编队外接圆侵蚀。
                       启用时，安全区域的侵蚀半径 = 编队外接圆半径 + 小车自身半径；
                       否则仅采用小车自身半径。
    
    算法流程：
      1. 计算小车几何中心及编队外接圆半径 formation_r；同时遍历所有顶点得到小车自身半径 car_r。
      2. 根据 formation_erosion 参数选择侵蚀半径：
            - True：erosion_radius = formation_r + car_r
            - False：erosion_radius = car_r
      3. 初始安全区域设为地图边界内缩 erosion_radius 后的矩形。
      4. 对每个障碍物进行裁剪，其中：
            - 圆形障碍物采用有效半径 (障碍物半径 + erosion_radius) 进行裁剪。
            - 多边形障碍物裁剪时向外平移 erosion_radius。
    """
    car_points = params[0]
    obstacles = params[1]

    # 计算小车几何中心及编队外接圆半径
    car_center, formation_r = compute_car_circle(car_points)
    # 计算小车自身半径（所有顶点中最大的 r）
    car_r = max(p[2] for p in car_points)
    # 根据 formation_erosion 参数选择侵蚀半径
    erosion_radius = formation_r + car_r if formation_erosion else car_r

    # 初始安全区域设为地图边界内缩 erosion_radius 后的区域
    polygon = [(erosion_radius, erosion_radius), 
               (20 - erosion_radius, erosion_radius), 
               (20 - erosion_radius, 20 - erosion_radius), 
               (erosion_radius, 20 - erosion_radius)]

    for obs in obstacles:
        if isinstance(obs, tuple) and len(obs) == 3:  # 圆形障碍物
            B = np.array([obs[0], obs[1]])
            obs_radius = obs[2]
            effective_radius = obs_radius + erosion_radius  # 有效障碍物半径
            A = car_center
            dist2 = np.sum((B - A) ** 2)
            if dist2 == 0:
                continue  # 避免除零
            t = 1 - effective_radius / np.sqrt(dist2)
            M_prime = A + t * (B - A)
            d = A - B
            c_val = np.dot(M_prime, d)
            polygon = clip_with_line(polygon, d, c_val)
        elif isinstance(obs, list) and len(obs) > 2:  # 多边形障碍物
            polygon = clip_with_convex_polygon(polygon, obs, car_center, erosion_radius, forward_vector)

    polygon = sort_polygon_vertices(polygon)
    return polygon

if __name__ == "__main__":
    # 示例：小车顶点列表，格式为 (x, y, r)；r 表示小车半径（各顶点取值，通常相同）
    car_points = [(0, 0, 0.2), (1, 0, 0.2), (1, 1, 0.2), (0, 1, 0.2)]
    # 障碍物列表：既包含圆形也包含多边形
    obstacles = [
        #(10, 4, 4),  # 圆形障碍物
        [(10, 10), (12, 10), (12, 12), (10, 12)]  # 多边形障碍物（矩形）
    ]
    
    # 设定是否启用编队外接圆侵蚀（True 表示使用编队外接圆半径，否则使用小车自身半径）
    formation_erosion = False
    
    safe_poly = safe_region(
        [
        car_points, 
        obstacles
        ], 
        forward_vector=(1,0),
        formation_erosion=formation_erosion
    )
    print("安全区域顶点（顺时针）:")
    for pt in safe_poly:
        print(pt)
    
    # 绘制图形
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
            ax.plot([p[0] for p in poly], [p[1] for p in poly], 'r--',
                    label='障碍物' if not obs_label_drawn else None)
            obs_label_drawn = True
    
    # 绘制小车初始位置（仅显示 x, y）
    car_x = [p[0] for p in car_points] + [car_points[0][0]]
    car_y = [p[1] for p in car_points] + [car_points[0][1]]
    ax.plot(car_x, car_y, color='blue', label='小车位置')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('小车安全区域（绿色填充）')
    plt.grid(True)
    plt.show()
