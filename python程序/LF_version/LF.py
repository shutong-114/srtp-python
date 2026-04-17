import math
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
    center = (cx, cy)
    r = max(math.hypot(p[0]-cx, p[1]-cy) for p in car_points)
    return center, r

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
            inter = compute_intersection(prev, curr)
            new_poly.append(inter)
        elif not prev_inside and curr_inside:
            inter = compute_intersection(prev, curr)
            new_poly.append(inter)
            new_poly.append(curr)
    return new_poly

def clip_with_line(polygon, d, c):
    """
    对多边形用直线 p·d = c 裁剪，保留满足 p·d <= c 的部分
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
    'ge' 表示保留 p[axis] >= val 的部分；'le' 表示保留 p[axis] <= val 的部分。
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
    # 按 -atan2 排序得到顺时针顺序
    poly_sorted = sorted(poly, key=lambda p: -math.atan2(p[1]-cy, p[0]-cx))
    return poly_sorted

def clip_polygon_offset(polygon, n, thresh):
    """
    对多边形用半平面 n·p >= thresh 裁剪
    n: 内法单位向量
    thresh: 阈值（原边界 n·P 加上偏移距离）
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

def erode_polygon(polygon, car_center, r, car_radiaus = 0):
    """
    对凸多边形整体内缩 r 距离（各边沿内法方向平移 r）
    方法：对原凸多边形每条边构造内缩的半平面，然后取交集。
    """
    halfplanes = []
    n = len(polygon)
    for i in range(n):
        P = polygon[i]
        Q = polygon[(i+1)%n]
        # 求该边的中点和边向量
        mid = ((P[0]+Q[0])/2, (P[1]+Q[1])/2)
        edge = (Q[0]-P[0], Q[1]-P[1])
        # 候选法向量（垂直于边）
        cand1 = (edge[1], -edge[0])
        cand2 = (-edge[1], edge[0])
        # 选择指向小车内部的一侧：即 (car_center - mid) 与法向量内积大于0
        if (car_center[0]-mid[0])*cand1[0] + (car_center[1]-mid[1])*cand1[1] > 0:
            n_vec = cand1
        else:
            n_vec = cand2
        norm = math.hypot(n_vec[0], n_vec[1])
        if norm == 0:
            continue
        n_unit = (n_vec[0]/norm, n_vec[1]/norm)
        # 原边界直线： n_unit·p = n_unit·P ，小车内部满足 n_unit·p >= n_unit·P
        # 平移 r 后，新直线： n_unit·p = n_unit·P + r
        thresh = n_unit[0]*P[0] + n_unit[1]*P[1] + r + car_radiaus
        halfplanes.append((n_unit, thresh))
    
    # 用足够大的初始多边形开始裁剪
    new_poly = [(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)]
    for n_unit, thresh in halfplanes:
        new_poly = clip_polygon_offset(new_poly, n_unit, thresh)
        if not new_poly:
            break
    return new_poly

def safe_region(params):
    """
    params: 列表，包含两个部分：
       params[0]: 小车顶点坐标列表，例如 [(x,y), ...]
       params[1]: 障碍物列表，每个障碍物为三元组 (ox, oy, r)
       
    返回：经过障碍物裁剪后，再向内侵蚀小车整体半径距离的安全区域顶点，
          顶点按顺时针顺序排列。
    """
    car_points = params[0]
    obstacles = params[1]  # 每个障碍物为 (ox, oy, r)

    # 1. 计算小车包围圆（得到小车几何中心和整体半径）
    car_center, r_c = compute_car_circle(car_points)
    r_c = 0 #测试
    
    # 2. 初始区域为边界 [0,20] x [0,20]
    polygon = [(0,0), (20,0), (20,20), (0,20)]
    
    # 3. 对每个障碍物，根据小车中心与障碍物中心构造新的半平面进行裁剪
    #    中点由公式 M' = X_c + (1/2 - 1/2*((r_o^2 - r_c^2)/||X_o-X_c||^2))*(X_o-X_c)
    for obs in obstacles:
        B = (obs[0], obs[1])
        obs_radius = obs[2]
        A = car_center
        dist2 = (B[0]-A[0])**2 + (B[1]-A[1])**2
        #t = 0.5 - 0.5 * ((obs_radius**2 - r_c**2) / dist2)
        t = 1 - obs_radius/math.sqrt(dist2)
        M_prime = (A[0] + t*(B[0]-A[0]), A[1] + t*(B[1]-A[1]))
        # d 为从小车中心指向障碍物中心的向量
        d = (B[0]-A[0], B[1]-A[1])
        # 半平面为： p·d <= M_prime·d，保证小车在安全侧
        c_val = M_prime[0]*d[0] + M_prime[1]*d[1]
        polygon = clip_with_line(polygon, d, c_val)
    
    # 4. 用 [0,20] x [0,20] 边界裁剪，确保区域限定在该范围内
    boundaries = [
        {'axis':'x', 'val':0,  'ineq':'ge'},
        {'axis':'x', 'val':20, 'ineq':'le'},
        {'axis':'y', 'val':0,  'ineq':'ge'},
        {'axis':'y', 'val':20, 'ineq':'le'},
    ]
    for b in boundaries:
        polygon = clip_with_boundary(polygon, b)
    
    # 5. 对裁剪结果顶点排序（顺时针）
    polygon = sort_polygon_vertices(polygon)
    
    # 6. 对安全区域整体进行内侵蚀：各边平移距离为小车整体半径 r_c
    eroded_polygon = erode_polygon(polygon, car_center, r_c)
    eroded_polygon = sort_polygon_vertices(eroded_polygon)
    
    return eroded_polygon

# 示例调用及绘图
if __name__ == "__main__":
    # 小车顶点及障碍物信息
    car_points = [(0,0), (1,0), (1,1), (0,1)]
    # 障碍物信息，每个障碍物为三元组 (x, y, r)
    obstacles = [(10,4,4), (10,16,4)]
    
    safe_poly = safe_region([car_points, obstacles])
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
    
    plt.show()
