import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import warnings

# ========== 设置中文字体支持 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ========== 核心函数（与script_8.py完全一致） ==========

def compute_relative_positions(X):
    """计算相对位置 r_1^{mn} (在F1坐标系中)"""
    N = len(X)
    r1_mn = np.zeros((N, N, 2))
    for m in range(N):
        for n in range(N):
            if m != n:
                r1_mn[m, n] = X[m] - X[n]
    return r1_mn

def J_col(X, A, d, EPS):
    """碰撞避免代价函数 - 论文公式(11)(12)"""
    N = len(X)
    total = 0.0
    r1_mn = compute_relative_positions(X)
    
    for m in range(N):
        for n in range(m + 1, N):
            diff = r1_mn[m, n]
            dist_sq = np.sum(diff**2) + EPS
            dist = np.sqrt(dist_sq)
            
            if dist < A:
                value = (dist_sq - A**2) / (dist_sq - d**2)
                total += (min(0.0, value))**2
    return total

def compute_shape_transformation(reference_formation, anchor_pair, anchor_vector):
    """
    根据锚点向量计算形状变换矩阵
    """
    i, j = anchor_pair
    
    # 参考队形中的锚点向量
    v_ref = reference_formation[j] - reference_formation[i]
    
    # 计算缩放因子
    ref_norm = np.linalg.norm(v_ref)
    target_norm = np.linalg.norm(anchor_vector)
    
    if ref_norm < 1e-6:
        return np.eye(2)
    
    scale = target_norm / ref_norm
    
    # 计算旋转角度
    v_ref_norm = v_ref / ref_norm
    v_target_norm = anchor_vector / target_norm
    
    # 计算旋转角度
    cos_theta = np.dot(v_ref_norm, v_target_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # 确定旋转方向（叉积符号）
    cross = np.cross(v_ref_norm, v_target_norm)
    if cross < 0:
        theta = -theta
    
    # 构造旋转矩阵
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # 变换矩阵C = 缩放 * 旋转
    C = scale * R
    
    return C

def compute_aligned_target(X, target_formation):
    """
    计算与当前队形对齐的目标队形
    """
    # 计算两个点集的中心
    center_X = np.mean(X, axis=0)
    center_T = np.mean(target_formation, axis=0)
    
    # 最优平移：将目标队形平移到当前队形的中心
    optimal_translation = center_X - center_T
    aligned_target = target_formation + optimal_translation
    
    return aligned_target, optimal_translation

def J_formation_to_target(X, target_formation, formation_weight=1.0):
    """
    形状队形成本函数 - 直接逼近目标队形
    """
    # 计算对齐后的目标队形
    aligned_target, _ = compute_aligned_target(X, target_formation)
    
    # 计算平方误差
    errors = X - aligned_target
    
    return formation_weight * np.sum(errors**2)

def compute_shape_error_to_target(X, target_formation):
    """
    计算当前队形与目标队形的形状误差
    """
    # 计算对齐后的目标队形
    aligned_target, optimal_translation = compute_aligned_target(X, target_formation)
    
    # 计算形状误差（平均欧氏距离）
    shape_error = np.mean(np.linalg.norm(X - aligned_target, axis=1))
    
    return shape_error, optimal_translation

def gradient_J_formation_finite_difference(X, target_formation, formation_weight=1.0, h=1e-5):
    """
    形状队形成本的有限差分梯度
    """
    N = len(X)
    grad = np.zeros_like(X)
    
    def J_total(positions):
        return J_formation_to_target(positions, target_formation, formation_weight)
    
    for i in range(N):
        for j in range(2):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[i, j] += h
            X_minus[i, j] -= h
            
            grad[i, j] = (J_total(X_plus) - J_total(X_minus)) / (2 * h)
    
    return grad

# ========== 优化器类（保持script_8.py的逻辑不变） ==========

class SimplifiedShapeOptimizer:
    """
    基于锚点向量的简化机器人队形优化器
    直接逼近目标队形，允许任意平移
    """
    
    def __init__(self, A=0.9, d=0.5, alpha=0.001, beta=0.9, 
                 robot_radius=0.5, max_iters=3000, eps=1e-8,
                 polygon_vertices=None, barrier_weight=10.0,
                 formation_strength=1.0, anchor_pair=(0, 1)):
        """
        初始化优化器参数
        
        参数:
        ----------
        A : float
            激活半径 (米)
        d : float
            碰撞避免半径 (米)
        alpha : float
            学习率
        beta : float
            动量参数
        robot_radius : float
            机器人半径 (米)
        max_iters : int
            最大迭代次数
        eps : float
            防止除零的小值
        polygon_vertices : numpy.ndarray, 形状 (M, 2), 可选
            凸多边形顶点坐标，按顺时针或逆时针顺序排列
        barrier_weight : float
            区域约束惩罚权重
        formation_strength : float
            队形保持强度系数，控制队形约束的权重
        anchor_pair : tuple, 默认 (0, 1)
            锚点机器人对索引
        """
        self.A = A
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.robot_radius = robot_radius
        self.max_iters = max_iters
        self.EPS = eps
        self.barrier_weight = barrier_weight
        self.formation_strength = formation_strength
        self.anchor_pair = anchor_pair
        
        # 多边形区域设置
        if polygon_vertices is not None:
            self.set_polygon_region(polygon_vertices)
        else:
            self.polygon_vertices = None
            self.polygon_edges = None
            self.polygon_normals = None
        
        # 存储历史记录
        self.target_formation = None  # 目标队形（固定形状）
        self.anchor_vector = None
        self.optimal_positions = None
        self.trajectory = None
        self.history = None
        self.initial_positions = None
    
    def set_polygon_region(self, vertices):
        """
        设置凸多边形区域
        
        参数:
        ----------
        vertices : numpy.ndarray, 形状 (M, 2)
            凸多边形顶点坐标，按顺时针或逆时针顺序排列
        """
        self.polygon_vertices = np.array(vertices)
        
        # 验证是否为凸多边形
        if len(self.polygon_vertices) < 3:
            raise ValueError("多边形至少需要3个顶点")
        
        # 计算凸包以确保顶点顺序正确
        hull = ConvexHull(self.polygon_vertices)
        self.polygon_vertices = self.polygon_vertices[hull.vertices]
        
        # 计算每条边的参数
        self._compute_polygon_edges()
        
        print(f"区域已设置为{len(self.polygon_vertices)}边形凸多边形")
        print(f"顶点坐标:")
        for i, v in enumerate(self.polygon_vertices):
            print(f"  顶点{i+1}: ({v[0]:.2f}, {v[1]:.2f})")
    
    def _compute_polygon_edges(self):
        """计算多边形的边和法向量"""
        n = len(self.polygon_vertices)
        self.polygon_edges = []
        self.polygon_normals = []
        
        # 假设顶点按逆时针顺序排列
        for i in range(n):
            p1 = self.polygon_vertices[i]
            p2 = self.polygon_vertices[(i + 1) % n]
            
            # 边向量
            edge = p2 - p1
            # 单位法向量（指向多边形内部，垂直于边向量）
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + self.EPS)
            
            self.polygon_edges.append(edge)
            self.polygon_normals.append(normal)
    
    def point_in_polygon(self, point):
        """
        判断点是否在凸多边形内部或边界上
        
        参数:
        ----------
        point : numpy.ndarray, 形状 (2,)
            要判断的点
        
        返回:
        ----------
        inside : bool
            点是否在多边形内（包括边界）
        distances : list
            到每条边的有符号距离（正表示在内部）
        """
        if self.polygon_vertices is None:
            return True, []  # 没有区域约束
        
        distances = []
        inside = True
        
        for i in range(len(self.polygon_vertices)):
            p1 = self.polygon_vertices[i]
            normal = self.polygon_normals[i]
            
            # 计算点到边的有符号距离（正表示在多边形内部）
            vec_to_point = point - p1
            distance = np.dot(vec_to_point, normal)
            distances.append(distance)
            
            # 如果点到任何一条边的距离为负，则点在多边形外部
            if distance < -self.EPS:
                inside = False
        
        return inside, distances
    
    def J_col(self, X):
        """碰撞避免代价函数"""
        return J_col(X, self.A, self.d, self.EPS)
    
    def J_formation(self, X):
        """
        形状队形成本 - 直接逼近目标队形
        """
        if self.target_formation is None:
            raise ValueError("未设置目标队形")
        
        return self.formation_strength * J_formation_to_target(
            X, self.target_formation, formation_weight=1.0
        )
    
    def J_region(self, X):
        """
        凸多边形区域约束的二次罚函数（外罚函数）
        """
        if self.polygon_vertices is None:
            return 0.0
        
        N = len(X)
        penalty = 0.0
        
        for i in range(N):
            _, distances = self.point_in_polygon(X[i])
            
            for d in distances:
                # 如果距离为负（在外部），则施加惩罚
                if d < 0:
                    penalty += d**2
        
        return self.barrier_weight * penalty
    
    def compute_shape_error(self, X):
        """
        计算当前队形与目标形状的形状误差
        """
        if self.target_formation is None:
            raise ValueError("未设置目标队形")
        
        shape_error, optimal_translation = compute_shape_error_to_target(
            X, self.target_formation
        )
        
        return shape_error, optimal_translation
    
    def compute_aligned_target(self, X):
        """
        计算与当前队形对齐的目标队形
        """
        if self.target_formation is None:
            raise ValueError("未设置目标队形")
        
        aligned_target, optimal_translation = compute_aligned_target(X, self.target_formation)
        
        return aligned_target
    
    def gradient_J_col_finite_difference(self, X, h=1e-5):
        """
        J_col的有限差分梯度
        """
        N = len(X)
        grad = np.zeros_like(X)
        
        def J_total(positions):
            return self.J_col(positions)
        
        for i in range(N):
            for j in range(2):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i, j] += h
                X_minus[i, j] -= h
                
                grad[i, j] = (J_total(X_plus) - J_total(X_minus)) / (2 * h)
        
        return grad
    
    def gradient_J_formation_finite_difference(self, X, h=1e-5):
        """
        形状队形成本的有限差分梯度
        """
        N = len(X)
        grad = np.zeros_like(X)
        
        def J_total(positions):
            return self.J_formation(positions)
        
        for i in range(N):
            for j in range(2):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i, j] += h
                X_minus[i, j] -= h
                
                grad[i, j] = (J_total(X_plus) - J_total(X_minus)) / (2 * h)
        
        return grad
    
    def gradient_J_region(self, X):
        """
        计算区域约束二次罚函数的梯度
        """
        if self.polygon_vertices is None:
            return np.zeros_like(X)
        
        N = len(X)
        grad = np.zeros_like(X)
        
        for i in range(N):
            _, distances = self.point_in_polygon(X[i])
            
            for j, d in enumerate(distances):
                normal = self.polygon_normals[j]
                
                # 二次罚函数的梯度
                if d < 0:
                    # 在外部：梯度 = 2μ * d * n
                    grad[i] += 2 * self.barrier_weight * d * normal
        
        return grad
    
    def compute_total_gradient(self, X):
        """总梯度计算"""
        grad_formation = self.gradient_J_formation_finite_difference(X)
        grad_col = self.gradient_J_col_finite_difference(X)
        grad_region = self.gradient_J_region(X)
        return grad_formation + grad_col + grad_region
    
    def compute_total_cost(self, X):
        """总代价计算"""
        cost_formation = self.J_formation(X)
        cost_col = self.J_col(X)
        cost_region = self.J_region(X)
        return cost_formation + cost_col + cost_region
    
    def set_shape_parameters(self, reference_formation, anchor_vector):
        """
        设置形状参数（参考队形和锚点向量）
        直接计算目标队形
        
        参数:
        ----------
        reference_formation : numpy.ndarray, 形状 (N, 2)
            参考队形S（标准化队形）
        anchor_vector : numpy.ndarray, 形状 (2,)
            锚点对的期望相对向量
        """
        # 验证锚点对索引
        i, j = self.anchor_pair
        if i >= len(reference_formation) or j >= len(reference_formation):
            raise ValueError(f"锚点对索引超出范围，参考队形有{len(reference_formation)}个机器人")
        
        self.reference_formation = reference_formation
        self.anchor_vector = anchor_vector
        
        # 计算变换矩阵（旋转和缩放）
        transformation_matrix = compute_shape_transformation(
            reference_formation, self.anchor_pair, anchor_vector
        )
        
        # 直接计算目标队形（固定形状）
        self.target_formation = np.dot(reference_formation, transformation_matrix.T)
        
        print(f"形状参数已设置，包含{len(reference_formation)}个机器人")
        print(f"锚点对: R{i+1} 和 R{j+1}")
        angle = np.degrees(np.arctan2(anchor_vector[1], anchor_vector[0]))
        print(f"期望锚点向量: [{anchor_vector[0]:.2f}, {anchor_vector[1]:.2f}]，方向: {angle:.1f}度")
        
        # 计算目标队形中的实际锚点向量
        target_anchor_vector = self.target_formation[j] - self.target_formation[i]
        target_angle = np.degrees(np.arctan2(target_anchor_vector[1], target_anchor_vector[0]))
        print(f"目标队形锚点向量: [{target_anchor_vector[0]:.2f}, {target_anchor_vector[1]:.2f}]，方向: {target_angle:.1f}度")
        
        # 显示变换信息
        scale = np.linalg.norm(target_anchor_vector) / np.linalg.norm(
            reference_formation[j] - reference_formation[i]
        )
        print(f"形状变换 - 缩放因子: {scale:.4f}")
        print(f"优化目标: 逼近固定目标队形，可在区域内任意平移")
    
    def create_square_reference_formation(self, side_length=1.0):
        """
        创建正方形参考队形
        
        参数:
        ----------
        side_length : float
            正方形边长
        
        返回:
        ----------
        reference_formation : numpy.ndarray, 形状 (4, 2)
            正方形参考队形
        """
        N = 4
        half = side_length / 2
        
        # 正方形的四个角（以原点为中心）
        reference_formation = np.array([
            [-half, -half],  # 左下角
            [ half, -half],  # 右下角
            [ half,  half],  # 右上角
            [-half,  half]   # 左上角
        ])
        
        return reference_formation
    
    def create_square_formation_parameters(self, side_length=1.0, region_center=None, random_noise=True, noise_std=0.2):
        """
        生成正方形队形参数
        
        参数:
        ----------
        side_length : float
            正方形边长
        region_center : tuple, 可选
            区域中心位置，用于生成初始位置
        random_noise : bool
            是否添加随机扰动
        noise_std : float
            随机扰动的标准差
        
        返回:
        ----------
        initial_positions : numpy.ndarray
            初始位置
        reference_formation : numpy.ndarray
            参考队形
        """
        # 生成参考队形（以原点为中心）
        reference_formation = self.create_square_reference_formation(side_length)
        
        if region_center is None and self.polygon_vertices is not None:
            # 如果没有指定区域中心，使用多边形区域的中心
            region_center = np.mean(self.polygon_vertices, axis=0)
        
        # 生成初始位置（在区域内，带有随机扰动）
        if region_center is not None:
            # 将参考队形平移到区域中心附近
            initial_positions = reference_formation + region_center
        else:
            initial_positions = reference_formation.copy()
        
        # 添加随机扰动（模拟初始位置不精确）
        if random_noise:
            initial_positions = initial_positions + np.random.normal(0, noise_std, initial_positions.shape)
        
        return initial_positions, reference_formation
    
    def optimize(self, initial_positions, verbose=True):
        """
        执行队形优化
        
        参数:
        ----------
        initial_positions : numpy.ndarray, 形状 (N, 2)
            机器人的初始位置，N为机器人数量
        verbose : bool, 默认 True
            是否打印优化过程信息
        
        返回:
        ----------
        optimal_positions : numpy.ndarray, 形状 (N, 2)
            优化后的机器人位置
        trajectory : list
            优化过程中的位置历史
        history : dict
            优化过程的详细信息
        """
        if self.target_formation is None:
            raise ValueError("请先设置形状参数 (set_shape_parameters)")
        
        # 参数验证
        N = len(initial_positions)
        if N != len(self.target_formation):
            raise ValueError(f"初始位置数量({N})与目标队形数量({len(self.target_formation)})不匹配")
        
        # 检查初始位置是否在区域内
        if self.polygon_vertices is not None:
            outside_robots = []
            for i, pos in enumerate(initial_positions):
                inside, _ = self.point_in_polygon(pos)
                if not inside:
                    outside_robots.append(i+1)
            
            if outside_robots and verbose:
                print(f"注意: 机器人{outside_robots}初始位置在区域外部，将使用外罚函数优化")
        
        # 初始化
        X = initial_positions.copy().astype(float)
        delta_x_prev = np.zeros_like(X)
        trajectory = [X.copy()]
        
        # 初始化历史记录
        history = {
            'cost_formation': [self.J_formation(X)],
            'cost_col': [self.J_col(X)],
            'cost_region': [self.J_region(X)],
            'total_cost': [],
            'updates': [],
            'iterations': [],
            'inside_region': [],
            'shape_errors': []
        }
        
        # 计算在区域内的机器人比例
        if self.polygon_vertices is not None:
            inside_count = 0
            for i in range(N):
                inside, _ = self.point_in_polygon(X[i])
                if inside:
                    inside_count += 1
            history['inside_region'].append(inside_count / N)
        
        # 计算初始形状误差
        shape_error, _ = self.compute_shape_error(X)
        history['shape_errors'].append(shape_error)
        
        if verbose:
            print(f"开始优化 - 机器人数量: {N}")
            i, j = self.anchor_pair
            print(f"锚点对: R{i+1} 和 R{j+1}")
            angle = np.degrees(np.arctan2(self.anchor_vector[1], self.anchor_vector[0]))
            print(f"期望锚点方向: {angle:.1f}度")
            if self.polygon_vertices is not None:
                print(f"区域约束: 二次罚函数（外罚）, 权重: {self.barrier_weight}")
            print(f"形状约束强度: {self.formation_strength}")
            print(f"优化目标: 逼近固定目标队形，可在区域内任意平移")
            print(f"初始成本: J_formation={history['cost_formation'][-1]:.4f}, "
                  f"J_col={history['cost_col'][-1]:.4f}")
            if self.polygon_vertices is not None:
                print(f"J_region={history['cost_region'][-1]:.4f}")
            print(f"初始形状误差: {shape_error:.4f}")
        
        # 优化循环
        for iteration in range(self.max_iters):
            # 计算梯度
            total_grad = self.compute_total_gradient(X)
            
            # 动量更新
            delta_x = -(self.alpha * total_grad + self.beta * delta_x_prev)
            
            # 更新位置
            X = X + delta_x
            
            # 记录历史
            if iteration % 10 == 0:
                trajectory.append(X.copy())
                history['cost_formation'].append(self.J_formation(X))
                history['cost_col'].append(self.J_col(X))
                history['cost_region'].append(self.J_region(X))
                history['updates'].append(np.max(np.abs(delta_x)))
                history['iterations'].append(iteration)
                
                # 计算形状误差
                shape_error, _ = self.compute_shape_error(X)
                history['shape_errors'].append(shape_error)
                
                # 记录在区域内的机器人比例
                if self.polygon_vertices is not None:
                    inside_count = 0
                    for i in range(N):
                        inside, _ = self.point_in_polygon(X[i])
                        if inside:
                            inside_count += 1
                    history['inside_region'].append(inside_count / N)
            
            # 检查收敛条件
            max_update = np.max(np.abs(delta_x))
            if max_update < 1e-5:
                if not np.array_equal(trajectory[-1], X):
                    trajectory.append(X.copy())
                    history['cost_formation'].append(self.J_formation(X))
                    history['cost_col'].append(self.J_col(X))
                    history['cost_region'].append(self.J_region(X))
                if verbose:
                    print(f"优化收敛于迭代 {iteration}: 最大更新量 {max_update:.2e}")
                break
            
            # 进度报告
            if verbose and iteration % 200 == 0:
                region_inside = 0
                if self.polygon_vertices is not None:
                    for i in range(N):
                        inside, _ = self.point_in_polygon(X[i])
                        if inside:
                            region_inside += 1
                
                msg = f"迭代 {iteration:4d} | 更新量 {max_update:.4e} | "
                if self.polygon_vertices is not None:
                    msg += f"区域内: {region_inside}/{N} | "
                msg += f"J_formation={self.J_formation(X):.4f}, "
                msg += f"J_col={self.J_col(X):.4f}"
                if self.polygon_vertices is not None:
                    msg += f", J_region={self.J_region(X):.4f}"
                
                shape_error, _ = self.compute_shape_error(X)
                msg += f", 形状误差={shape_error:.4f}"
                
                print(msg)
            
            delta_x_prev = delta_x.copy()
        
        # 最终记录
        if not np.array_equal(trajectory[-1], X):
            trajectory.append(X.copy())
            history['cost_formation'].append(self.J_formation(X))
            history['cost_col'].append(self.J_col(X))
            history['cost_region'].append(self.J_region(X))
        
        # 计算总成本
        if self.polygon_vertices is not None:
            history['total_cost'] = [f + c + r for f, c, r in 
                                     zip(history['cost_formation'], history['cost_col'], history['cost_region'])]
        else:
            history['total_cost'] = [f + c for f, c in 
                                     zip(history['cost_formation'], history['cost_col'])]
        
        # 保存结果
        self.optimal_positions = X
        self.trajectory = trajectory
        self.history = history
        self.initial_positions = initial_positions
        
        if verbose:
            # 计算最终形状误差
            shape_error, optimal_translation = self.compute_shape_error(X)
            
            # 统计最终在区域内的机器人数量
            if self.polygon_vertices is not None:
                final_inside = 0
                for i in range(N):
                    inside, _ = self.point_in_polygon(X[i])
                    if inside:
                        final_inside += 1
                
                print(f"优化完成 - 最终成本: J_formation={history['cost_formation'][-1]:.4f}, "
                      f"J_col={history['cost_col'][-1]:.4f}, "
                      f"J_region={history['cost_region'][-1]:.4f}")
                print(f"最终形状误差: {shape_error:.4f}")
                print(f"最优平移向量: [{optimal_translation[0]:.4f}, {optimal_translation[1]:.4f}]")
                print(f"区域内机器人: {final_inside}/{N} ({final_inside/N*100:.1f}%)")
            else:
                print(f"优化完成 - 最终成本: J_formation={history['cost_formation'][-1]:.4f}, "
                      f"J_col={history['cost_col'][-1]:.4f}")
                print(f"最终形状误差: {shape_error:.4f}")
                print(f"最优平移向量: [{optimal_translation[0]:.4f}, {optimal_translation[1]:.4f}]")
        
        return X, trajectory, history
    
    def get_optimization_summary(self):
        """
        获取优化结果摘要
        """
        if self.optimal_positions is None:
            raise ValueError("请先执行optimize()方法")
        
        N = len(self.optimal_positions)
        
        # 计算位置变化
        position_changes = []
        for i in range(N):
            delta = np.linalg.norm(self.optimal_positions[i] - self.initial_positions[i])
            position_changes.append(delta)
        
        # 计算最终间距
        distances = []
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(self.optimal_positions[i] - self.optimal_positions[j])
                distances.append(dist)
        
        # 计算形状误差和最优平移
        shape_error, optimal_translation = self.compute_shape_error(self.optimal_positions)
        
        # 检查锚点方向
        i, j = self.anchor_pair
        actual_anchor_vector = self.optimal_positions[j] - self.optimal_positions[i]
        anchor_angle_error = np.degrees(np.arccos(
            np.clip(np.dot(actual_anchor_vector, self.anchor_vector) / 
            (np.linalg.norm(actual_anchor_vector) * np.linalg.norm(self.anchor_vector) + 1e-6), -1.0, 1.0)
        ))
        
        summary = {
            'initial_cost': {
                'formation': self.history['cost_formation'][0],
                'col': self.history['cost_col'][0],
                'region': self.history['cost_region'][0] if 'cost_region' in self.history else 0
            },
            'final_cost': {
                'formation': self.history['cost_formation'][-1],
                'col': self.history['cost_col'][-1],
                'region': self.history['cost_region'][-1] if 'cost_region' in self.history else 0
            },
            'position_changes': position_changes,
            'distances': distances,
            'min_distance': min(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0,
            'mean_distance': np.mean(distances) if distances else 0,
            'num_robots': N,
            'num_iterations': len(self.trajectory),
            'formation_strength': self.formation_strength,
            'anchor_pair': self.anchor_pair,
            'shape_error': shape_error,
            'optimal_translation': optimal_translation,
            'anchor_angle_error_deg': anchor_angle_error
        }
        
        # 添加区域约束信息
        if self.polygon_vertices is not None:
            violation_info = self.get_region_violation_info()
            summary['region_violation'] = violation_info
        
        return summary
    
    def get_region_violation_info(self, positions=None):
        """
        获取区域违反情况信息
        """
        if self.polygon_vertices is None:
            return {"has_constraint": False}
        
        if positions is None:
            if self.optimal_positions is None:
                raise ValueError("请先执行optimize()方法或提供positions参数")
            positions = self.optimal_positions
        
        N = len(positions)
        outside_robots = []
        min_distances = []
        
        for i in range(N):
            inside, distances = self.point_in_polygon(positions[i])
            if not inside:
                outside_robots.append(i)
            min_distances.append(min(distances) if distances else 0)
        
        return {
            "has_constraint": True,
            "num_robots": N,
            "num_outside": len(outside_robots),
            "outside_robots": outside_robots,
            "all_inside": (len(outside_robots) == 0),
            "min_distances": min_distances,
            "avg_min_distance": np.mean(min_distances) if min_distances else 0
        }


# ========== 分布式接口兼容类（仅用于主程序兼容性） ==========

class CompatibleShapeOptimizer(SimplifiedShapeOptimizer):
    """
    兼容分布式接口的优化器
    仅添加必要的接口方法，不改变原有逻辑
    """
    
    def __init__(self, shape_icon, sigma=0.1, max_iter=1000, tol=1e-2, eta=0.2, **kwargs):
        """
        初始化优化器（兼容分布式接口）
        
        参数:
        ----------
        shape_icon : numpy.ndarray
            参考形状点集（用作参考队形）
        sigma, max_iter, tol, eta : 分布式接口参数（仅用于兼容性）
        **kwargs : 集中式优化器参数
        """
        # 提取集中式优化器参数
        A = kwargs.get('A', 0.9)
        d = kwargs.get('d', 0.5)
        alpha = kwargs.get('alpha', 0.001)
        beta = kwargs.get('beta', 0.9)
        robot_radius = kwargs.get('robot_radius', 0.5)
        max_iters = max_iter  # 使用分布式接口的max_iter
        eps = 1e-8
        polygon_vertices = kwargs.get('polygon_vertices', None)
        barrier_weight = kwargs.get('barrier_weight', 10.0)
        formation_strength = kwargs.get('formation_strength', 1.0)
        anchor_pair = kwargs.get('anchor_pair', (0, 1))
        
        # 调用父类初始化
        super().__init__(
            A=A, d=d, alpha=alpha, beta=beta,
            robot_radius=robot_radius, max_iters=max_iters, eps=eps,
            polygon_vertices=polygon_vertices, barrier_weight=barrier_weight,
            formation_strength=formation_strength, anchor_pair=anchor_pair
        )
        
        # 保存参考队形
        self.shape_icon = shape_icon
        self.sigma = sigma
        self.tol = tol
        self.eta = eta
        
        # 收敛历史（用于兼容性）
        self.convergence_history = []
    
    def optimize_distributed(self, initial_positions, LF_vertices, adjacency_matrix, b, 
                            verbose=False, **kwargs):
        """
        分布式接口兼容方法
        
        参数:
        ----------
        initial_positions : numpy.ndarray, 形状 (N, 2)
            初始位置数组
        LF_vertices : numpy.ndarray, 形状 (M, 2)
            可行区域顶点
        adjacency_matrix : numpy.ndarray, 形状 (N, N)
            通信邻接矩阵（不使用）
        b : numpy.ndarray, 形状 (2,)
            锚点约束向量
        verbose : bool, 默认 False
            是否打印优化信息
        """
        # 设置多边形区域
        if LF_vertices is not None:
            self.set_polygon_region(LF_vertices)
        
        # 使用shape_icon作为参考队形
        reference_formation = self.shape_icon
        
        # 设置形状参数（锚点向量 = b）
        self.set_shape_parameters(reference_formation, b)
        
        # 执行优化
        optimal_positions, _, _ = self.optimize(
            initial_positions=initial_positions,
            verbose=verbose
        )
        
        # 转换为字典格式以兼容分布式接口
        result_dict = {}
        for i in range(len(optimal_positions)):
            result_dict[i+1] = optimal_positions[i]  # 键从1开始
        
        # 更新收敛历史
        if hasattr(self, 'history') and 'shape_errors' in self.history:
            self.convergence_history = self.history['shape_errors']
        
        return result_dict
    
    def get_formation_error(self):
        """计算队形误差（兼容分布式接口）"""
        if self.optimal_positions is None or self.target_formation is None:
            return 0.0
        
        shape_error, _ = compute_shape_error_to_target(
            self.optimal_positions, self.target_formation
        )
        return shape_error
    
    def get_consensus_error(self):
        """计算一致性误差（集中式返回0）"""
        return 0.0
    

def create_simplified_optimization_visualization(initial_positions, optimal_positions, optimizer):
    """
    创建简化优化结果可视化
    
    参数:
    ----------
    initial_positions : numpy.ndarray
        初始位置
    optimal_positions : numpy.ndarray
        优化后位置
    optimizer : SimplifiedShapeOptimizer
        优化器实例
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    N = len(initial_positions)
    i, j = optimizer.anchor_pair
    
    # 计算对齐的目标队形
    aligned_target = optimizer.compute_aligned_target(optimal_positions)
    
    titles = ["初始队形", "优化结果", "形状对比", 
              "代价函数变化", "形状误差变化", "区域检查"]
    
    # 第1行：队形对比
    positions_list = [initial_positions, optimal_positions, optimal_positions]
    
    for idx, (ax, title, positions) in enumerate(zip(axes[0, :], titles[:3], positions_list)):
        # 绘制多边形区域
        if optimizer.polygon_vertices is not None:
            polygon_patch = plt.Polygon(optimizer.polygon_vertices, closed=True, 
                                       facecolor='lightgreen', alpha=0.1, 
                                       edgecolor='green', linewidth=2)
            ax.add_patch(polygon_patch)
        
        # 绘制机器人
        colors = ['green' if k in optimizer.anchor_pair else 'blue' for k in range(N)]
        sizes = [350 if k in optimizer.anchor_pair else 250 for k in range(N)]
        markers = ['s' if k in optimizer.anchor_pair else 'o' for k in range(N)]
        
        for k, pos in enumerate(positions):
            ax.scatter(pos[0], pos[1], color=colors[k], s=sizes[k], marker=markers[k], 
                      edgecolor='black', linewidths=2, alpha=0.8, zorder=10)
            
            # 绘制机器人标签
            label_text = f'R{k+1}' + ('(锚点)' if k in optimizer.anchor_pair else '')
            label_color = 'darkgreen' if k in optimizer.anchor_pair else 'darkblue'
            ax.annotate(label_text, xy=pos, xytext=(0, 15), 
                       textcoords='offset points', ha='center',
                       fontsize=10, fontweight='bold', color=label_color, zorder=11)
        
        # 绘制正方形连接线
        if idx == 1 or idx == 2:  # 优化结果和形状对比
            # 绘制实际队形的四条边
            for k in range(4):
                p1 = positions[k]
                p2 = positions[(k+1) % 4]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, linewidth=1.5, label='实际队形')
        
        # 绘制锚点向量（在初始和优化结果中）
        if idx == 0 or idx == 1:
            anchor_pos = positions[i]
            
            # 绘制实际锚点向量
            actual_vector = positions[j] - positions[i]
            ax.arrow(anchor_pos[0], anchor_pos[1], 
                    actual_vector[0], actual_vector[1],
                    head_width=0.1, head_length=0.15, 
                    fc='blue', ec='blue', alpha=0.7, linewidth=2,
                    label='实际锚点向量')
            
            # 在初始队形中绘制期望锚点向量
            if idx == 0 and optimizer.anchor_vector is not None:
                ax.arrow(anchor_pos[0], anchor_pos[1], 
                        optimizer.anchor_vector[0], optimizer.anchor_vector[1],
                        head_width=0.15, head_length=0.2, 
                        fc='red', ec='red', alpha=0.7, linewidth=2, 
                        label='期望锚点向量')
        
        # 在形状对比图中绘制对齐的目标队形
        if idx == 2:
            # 绘制对齐的目标队形
            for k, pos in enumerate(aligned_target):
                ax.scatter(pos[0], pos[1], color='red', s=100, marker='x', 
                          linewidths=2, alpha=0.8, zorder=9, label='目标形状' if k==0 else "")
            
            # 绘制目标形状的四条边
            for k in range(4):
                p1 = aligned_target[k]
                p2 = aligned_target[(k+1) % 4]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', alpha=0.6, linewidth=1.5, label='目标形状')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 设置合适的显示范围
        all_pos = np.vstack([initial_positions, optimal_positions, aligned_target])
        if optimizer.polygon_vertices is not None:
            all_pos = np.vstack([all_pos, optimizer.polygon_vertices])
        margin = 1.0
        ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        ax.set_xlabel('X (米)', fontsize=11)
        ax.set_ylabel('Y (米)', fontsize=11)
        
        # 添加图例
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
        elif idx == 2:
            ax.legend(loc='upper right', fontsize=9)
    
    # 第2行：代价函数变化
    if optimizer.history is not None:
        # 代价函数变化
        ax1 = axes[1, 0]
        iterations = optimizer.history['iterations']
        
        ax1.plot(iterations, optimizer.history['cost_formation'][:len(iterations)], 
                'b-', linewidth=2, label='形状代价(J_formation)')
        ax1.plot(iterations, optimizer.history['cost_col'][:len(iterations)], 
                'g-', linewidth=2, label='碰撞代价(J_col)')
        
        if optimizer.polygon_vertices is not None:
            ax1.plot(iterations, optimizer.history['cost_region'][:len(iterations)], 
                    'r-', linewidth=2, label='区域代价(J_region)')
        
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('代价函数值', fontsize=12)
        ax1.set_title('代价函数变化', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # 形状误差变化
        ax2 = axes[1, 1]
        ax2.plot(iterations, optimizer.history['shape_errors'][:len(iterations)], 
                'm-', linewidth=2, label='形状误差')
        
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('形状误差 (米)', fontsize=12)
        ax2.set_title('形状误差变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        # 区域检查
        ax3 = axes[1, 2]
        if optimizer.polygon_vertices is not None and 'inside_region' in optimizer.history:
            inside_ratios = optimizer.history['inside_region'][:len(iterations)]
            ax3.plot(iterations, inside_ratios, 
                    'c-', linewidth=2, label='区域内机器人比例')
            ax3.set_ylim(0, 1.1)
            ax3.set_ylabel('区域内比例', fontsize=12)
        else:
            ax3.text(0.5, 0.5, '无区域约束', ha='center', va='center', 
                    fontsize=14, transform=ax3.transAxes)
        
        ax3.set_xlabel('迭代次数', fontsize=12)
        ax3.set_title('区域约束检查', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        if optimizer.polygon_vertices is not None:
            ax3.legend(loc='upper right', fontsize=10)
    
    # 添加整体标题
    shape_error, _ = optimizer.compute_shape_error(optimal_positions)
    plt.suptitle(f'基于锚点向量的简化队形优化结果 - 锚点对R{i+1}和R{j+1} (形状误差: {shape_error:.4f})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()


# ========== 测试函数（更新为使用CompatibleShapeOptimizer） ==========

def test_shape_optimization_inside_region():
    """
    测试：基于锚点向量的简化队形优化（机器人在区域内）- 使用兼容接口
    """
    print("\n" + "="*70)
    print("测试：基于锚点向量的简化队形优化（机器人在区域内）- 兼容接口")
    print("="*70)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 创建凸多边形区域
    polygon_vertices = np.array([
        [0.0, 0.0],   # 顶点1
        [10.0, 0.0],  # 顶点2
        [10.0, 8.0],  # 顶点3
        [0.0, 8.0]    # 顶点4
    ])
    
    # 2. 创建正方形参考队形
    side_length = 2.0
    reference_formation = np.array([
        [-side_length/2, -side_length/2],  # 左下角
        [ side_length/2, -side_length/2],  # 右下角
        [-side_length/2,  side_length/2],  # 左上角
        [ side_length/2,  side_length/2]   # 右上角
    ])
    
    # 3. 创建初始位置（在区域内，带有随机扰动）
    region_center = (5.0, 4.0)  # 区域中心附近
    initial_positions = reference_formation + region_center + np.random.normal(0, 0.3, reference_formation.shape)
    
    # 4. 设置锚点向量（45度方向）
    anchor_angle = np.radians(45)  # 45度
    anchor_length = side_length  # 锚点向量长度等于正方形边长
    anchor_vector = anchor_length * np.array([np.cos(anchor_angle), np.sin(anchor_angle)])
    
    # 5. 创建兼容接口的优化器
    optimizer = CompatibleShapeOptimizer(
        shape_icon=reference_formation,
        sigma=0.1,
        max_iter=1500,
        tol=1e-5,
        eta=0.2,
        # 集中式优化器参数
        A=0.9,
        d=0.5,
        alpha=0.001,
        beta=0.9,
        robot_radius=0.5,
        barrier_weight=15.0,
        formation_strength=1.0,
        anchor_pair=(0, 1),
        polygon_vertices=polygon_vertices
    )
    
    print(f"\n测试配置:")
    print(f"  机器人数量: 4")
    print(f"  队形: {side_length}x{side_length}米正方形")
    print(f"  锚点对: R{optimizer.anchor_pair[0]+1} 和 R{optimizer.anchor_pair[1]+1}")
    angle = np.degrees(np.arctan2(anchor_vector[1], anchor_vector[0]))
    print(f"  期望锚点方向: {angle:.1f}度")
    print(f"  区域: {len(polygon_vertices)}边形")
    print(f"  优化目标: 逼近固定目标队形，可在区域内任意平移")
    
    print(f"\n初始位置:")
    for i, pos in enumerate(initial_positions):
        inside, _ = optimizer.point_in_polygon(pos)
        status = "区域内" if inside else "区域外"
        print(f"  R{i+1}: [{pos[0]:.2f}, {pos[1]:.2f}] - {status}")
    
    # 6. 定义通信拓扑（环状，集中式优化中不使用，但为了接口兼容）
    adjacency_matrix = np.array([
        [0, 1, 0, 1],  # 小车0连接1和3
        [1, 0, 1, 0],  # 小车1连接0和2
        [0, 1, 0, 1],  # 小车2连接1和3
        [1, 0, 1, 0]   # 小车3连接0和2
    ])
    
    # 7. 执行优化（使用兼容接口）
    print("\n" + "="*70)
    print("开始优化（使用兼容接口）...")
    print("="*70)
    
    result_dict = optimizer.optimize_distributed(
        initial_positions=initial_positions,
        LF_vertices=polygon_vertices,
        adjacency_matrix=adjacency_matrix,
        b=anchor_vector,
        verbose=True
    )
    
    # 8. 将结果从字典转换为数组
    optimal_positions = np.array([result_dict[i+1] for i in range(len(result_dict))])
    
    # 9. 获取优化摘要
    summary = optimizer.get_optimization_summary()
    
    # 10. 显示结果
    print("\n" + "="*70)
    print("优化结果摘要")
    print("="*70)
    print(f"锚点对: R{summary['anchor_pair'][0]+1} 和 R{summary['anchor_pair'][1]+1}")
    print(f"形状误差: {summary['shape_error']:.4f}")
    print(f"锚点方向误差: {summary['anchor_angle_error_deg']:.2f}度")
    print(f"最优平移向量: [{summary['optimal_translation'][0]:.4f}, {summary['optimal_translation'][1]:.4f}]")
    print(f"初始成本: J_formation={summary['initial_cost']['formation']:.4f}, "
          f"J_col={summary['initial_cost']['col']:.4f}")
    print(f"最终成本: J_formation={summary['final_cost']['formation']:.4f}, "
          f"J_col={summary['final_cost']['col']:.4f}")
    print(f"最小间距: {summary['min_distance']:.4f}米")
    print(f"最大位置变化: {max(summary['position_changes']):.4f}米")
    print(f"机器人数量: {summary['num_robots']}")
    print(f"总迭代次数: {summary['num_iterations']}")
    
    # 11. 检查正方形边长
    sides = []
    for k in range(4):
        p1 = optimal_positions[k]
        p2 = optimal_positions[(k+1) % 4]
        sides.append(np.linalg.norm(p2 - p1))
    
    print(f"\n正方形边长检查:")
    print(f"  期望边长: {side_length:.2f}米")
    print(f"  实际边长: 平均={np.mean(sides):.4f}米, 标准差={np.std(sides):.4f}米")
    print(f"  边长误差: {abs(np.mean(sides) - side_length):.4f}米")
    
    # 12. 检查正方形内角
    angles = []
    for k in range(4):
        p1 = optimal_positions[k]
        p2 = optimal_positions[(k+1) % 4]
        p3 = optimal_positions[(k+2) % 4]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        angles.append(angle_deg)
    
    print(f"\n正方形内角检查:")
    print(f"  期望内角: 90度")
    print(f"  实际内角: 平均={np.mean(angles):.1f}度, 标准差={np.std(angles):.1f}度")
    print(f"  内角误差: {abs(np.mean(angles) - 90):.1f}度")
    
    # 13. 检查区域违反情况
    violation_info = optimizer.get_region_violation_info()
    if violation_info['has_constraint']:
        print(f"\n区域约束检查:")
        print(f"  所有机器人在区域内: {violation_info['all_inside']}")
        print(f"  外部机器人数量: {violation_info['num_outside']}/{violation_info['num_robots']}")
        print(f"  平均最小边距: {violation_info['avg_min_distance']:.4f}")
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    if summary['shape_error'] < 0.3 and summary['anchor_angle_error_deg'] < 5 and violation_info['all_inside']:
        print("✅ 测试通过：成功优化到目标形状！")
    else:
        print("⚠️ 测试部分通过：某些指标需改进")
    
    # 14. 创建可视化
    create_simplified_optimization_visualization(initial_positions, optimal_positions, optimizer)
    
    return optimizer, initial_positions, optimal_positions

def test_shape_optimization_all_outside():
    """
    测试：基于锚点向量的简化队形优化（所有机器人初始在区域外）- 使用兼容接口
    """
    print("\n" + "="*70)
    print("测试：基于锚点向量的简化队形优化（所有机器人初始在区域外）- 兼容接口")
    print("="*70)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 创建一个凸多边形区域
    polygon_vertices = np.array([
        [3.0, 3.0],   # 顶点1
        [7.0, 3.0],   # 顶点2
        [7.0, 7.0],   # 顶点3
        [3.0, 7.0]    # 顶点4
    ])
    
    # 2. 创建正方形参考队形
    side_length = 1.5
    reference_formation = np.array([
        [-side_length/2, -side_length/2],  # 左下角
        [ side_length/2, -side_length/2],  # 右下角
        [-side_length/2,  side_length/2],  # 左上角
        [ side_length/2,  side_length/2]   # 右上角
    ])
    
    # 3. 初始位置：所有机器人都在区域外
    initial_positions = np.array([
        [0.0, 0.0],   # R1: 区域外左下
        [-1.0, 0.0],  # R2: 区域外右下
        [1.0, 3.0],   # R3: 区域外右上
        [0.0, 1.0]    # R4: 区域外左上
    ])
    
    # 4. 设置锚点向量（60度方向）
    anchor_angle = np.radians(60)  # 60度
    anchor_length = side_length
    anchor_vector = anchor_length * np.array([np.cos(anchor_angle), np.sin(anchor_angle)])
    
    # 5. 创建兼容接口的优化器
    optimizer = CompatibleShapeOptimizer(
        shape_icon=reference_formation,
        sigma=0.1,
        max_iter=2000,
        tol=1e-5,
        eta=0.2,
        # 集中式优化器参数
        A=0.9,
        d=0.5,
        alpha=0.001,
        beta=0.9,
        robot_radius=0.5,
        barrier_weight=25.0,  # 增大区域约束权重
        formation_strength=1.2,  # 增大形状约束权重
        anchor_pair=(0, 1),
        polygon_vertices=polygon_vertices
    )
    
    print(f"\n测试配置:")
    print(f"  机器人数量: 4")
    print(f"  队形: {side_length}x{side_length}米正方形")
    print(f"  锚点对: R{optimizer.anchor_pair[0]+1} 和 R{optimizer.anchor_pair[1]+1}")
    angle = np.degrees(np.arctan2(anchor_vector[1], anchor_vector[0]))
    print(f"  期望锚点方向: {angle:.1f}度")
    print(f"  区域: {len(polygon_vertices)}边形")
    print(f"  优化目标: 逼近固定目标队形，可在区域内任意平移")
    
    print(f"\n初始位置（全部在区域外）:")
    for i, pos in enumerate(initial_positions):
        inside, distances = optimizer.point_in_polygon(pos)
        min_distance = min(distances) if distances else 0
        status = "区域内" if inside else "区域外"
        print(f"  R{i+1}: [{pos[0]:.2f}, {pos[1]:.2f}] - {status} - 最小边距: {min_distance:.4f}")
    
    # 6. 计算初始形状误差
    optimizer.set_shape_parameters(reference_formation, anchor_vector)
    shape_error_initial, _ = optimizer.compute_shape_error(initial_positions)
    print(f"初始形状误差: {shape_error_initial:.4f}米")
    
    # 7. 定义通信拓扑（环状，集中式优化中不使用，但为了接口兼容）
    adjacency_matrix = np.array([
        [0, 1, 0, 1],  # 小车0连接1和3
        [1, 0, 1, 0],  # 小车1连接0和2
        [0, 1, 0, 1],  # 小车2连接1和3
        [1, 0, 1, 0]   # 小车3连接0和2
    ])
    
    # 8. 执行优化（使用兼容接口）
    print("\n" + "="*70)
    print("开始优化（使用兼容接口）...")
    print("="*70)
    
    result_dict = optimizer.optimize_distributed(
        initial_positions=initial_positions,
        LF_vertices=polygon_vertices,
        adjacency_matrix=adjacency_matrix,
        b=anchor_vector,
        verbose=True
    )
    
    # 9. 将结果从字典转换为数组
    optimal_positions = np.array([result_dict[i+1] for i in range(len(result_dict))])
    
    # 10. 检查优化结果
    print("\n" + "="*70)
    print("优化结果检查")
    print("="*70)
    
    # 检查是否所有机器人都在区域内
    all_inside = True
    print("机器人最终位置:")
    for i, pos in enumerate(optimal_positions):
        inside, distances = optimizer.point_in_polygon(pos)
        min_distance = min(distances) if distances else 0
        status = "区域内" if inside else "区域外"
        print(f"  R{i+1}: [{pos[0]:.2f}, {pos[1]:.2f}] - {status} - 最小边距: {min_distance:.4f}")
        if not inside:
            all_inside = False
    
    # 11. 获取优化摘要
    summary = optimizer.get_optimization_summary()
    
    print(f"\n形状精度检查:")
    print(f"  最终形状误差: {summary['shape_error']:.4f}米")
    print(f"  锚点方向误差: {summary['anchor_angle_error_deg']:.2f}度")
    print(f"  最优平移向量: [{summary['optimal_translation'][0]:.4f}, {summary['optimal_translation'][1]:.4f}]")
    
    print(f"\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"✓ 所有机器人在区域内: {all_inside}")
    print(f"✓ 形状精度: {summary['shape_error']:.4f}米 {'(良好)' if summary['shape_error'] < 0.4 else '(需改进)'}")
    print(f"✓ 锚点方向误差: {summary['anchor_angle_error_deg']:.2f}度")
    
    if all_inside and summary['shape_error'] < 0.4:
        print(f"\n✅ 测试通过：外罚函数成功将区域外机器人拉回并形成目标形状！")
        print(f"   队形已自动平移到区域内最佳位置。")
    else:
        print(f"\n⚠️ 测试部分通过：需要调整参数或算法。")
    
    # 12. 创建可视化
    create_simplified_optimization_visualization(initial_positions, optimal_positions, optimizer)
    
    return optimizer, initial_positions, optimal_positions

def test_compatible_optimizer_simple():
    """
    简单的兼容接口测试
    """
    print("\n" + "="*70)
    print("简单的兼容接口测试")
    print("="*70)
    
    np.random.seed(42)
    
    # 1. 创建凸多边形区域
    polygon_vertices = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 5.0],
        [0.0, 5.0]
    ])
    
    # 2. 创建参考队形（正方形）
    shape_icon = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    
    # 3. 创建初始位置
    initial_positions = np.array([
        [1.0, 1.0],
        [2.0, 1.0],
        [1.5, 2.0],
        [2.5, 2.0]
    ])
    
    # 4. 锚点向量
    b = np.array([1.0, 0.0])  # 期望p1在p2右边1米处
    
    # 5. 通信拓扑（环状）
    adjacency_matrix = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    
    # 6. 创建兼容接口的优化器
    optimizer = CompatibleShapeOptimizer(
        shape_icon=shape_icon,
        sigma=0.1,
        max_iter=1000,
        tol=1e-3,
        eta=0.2,
        A=0.9,
        d=0.5,
        alpha=0.001,
        beta=0.9,
        barrier_weight=10.0,
        formation_strength=1.0,
        anchor_pair=(0, 1),
        polygon_vertices=polygon_vertices
    )
    
    print(f"优化器创建完成:")
    print(f"  - 参考队形: {shape_icon.shape}")
    print(f"  - 锚点对: R{optimizer.anchor_pair[0]+1} 和 R{optimizer.anchor_pair[1]+1}")
    print(f"  - 锚点向量: {b}")
    
    # 7. 执行优化
    print("\n开始优化...")
    result = optimizer.optimize_distributed(
        initial_positions=initial_positions,
        LF_vertices=polygon_vertices,
        adjacency_matrix=adjacency_matrix,
        b=b,
        verbose=True
    )
    
    # 8. 显示结果
    print("\n优化结果:")
    for i in range(1, 5):
        print(f"  机器人{i}: [{result[i][0]:.4f}, {result[i][1]:.4f}]")
    
    # 9. 计算性能指标
    formation_error = optimizer.get_formation_error()
    print(f"队形误差: {formation_error:.6f}")
    
    # 10. 获取优化摘要
    summary = optimizer.get_optimization_summary()
    print(f"形状误差: {summary['shape_error']:.4f}")
    print(f"最优平移: [{summary['optimal_translation'][0]:.4f}, {summary['optimal_translation'][1]:.4f}]")
    
    return optimizer, result

# ========== 主函数（更新为使用兼容接口） ==========

def main():
    """主函数"""
    np.random.seed(42)
    
    print("="*80)
    print("多机器人队形优化 - 兼容分布式接口的集中式优化器")
    print("="*80)
    
    # 询问用户要运行哪个测试
    print("\n请选择测试模式:")
    print("1. 兼容接口优化测试（机器人在区域内）")
    print("2. 兼容接口优化测试（所有机器人在区域外）")
    print("3. 简单兼容接口测试")
    print("4. 运行所有测试")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == "1":
        # 测试1：机器人在区域内
        test_shape_optimization_inside_region()
    elif choice == "2":
        # 测试2：所有机器人在区域外
        test_shape_optimization_all_outside()
    elif choice == "3":
        # 测试3：简单兼容接口测试
        test_compatible_optimizer_simple()
    elif choice == "4":
        # 运行所有测试
        print("\n" + "="*80)
        print("测试1: 兼容接口优化测试（机器人在区域内）")
        print("="*80)
        test_shape_optimization_inside_region()
        
        print("\n" + "="*80)
        print("测试2: 兼容接口优化测试（所有机器人在区域外）")
        print("="*80)
        test_shape_optimization_all_outside()
        
        print("\n" + "="*80)
        print("测试3: 简单兼容接口测试")
        print("="*80)
        test_compatible_optimizer_simple()
        
        print("\n" + "="*80)
        print("所有测试完成！")
        print("="*80)
    else:
        print("无效选择，将运行默认测试（兼容接口优化测试，机器人在区域内）")
        test_shape_optimization_inside_region()

if __name__ == "__main__":
    main()


