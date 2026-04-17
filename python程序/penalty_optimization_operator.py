import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import warnings

# ========== 设置中文字体支持 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ========== 核心数学函数 (保持原样，提供底层支持) ==========

def compute_relative_positions(X):
    """计算相对位置 r_1^{mn}"""
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
    """根据锚点向量计算形状变换矩阵 (旋转+缩放)"""
    i, j = anchor_pair
    v_ref = reference_formation[j] - reference_formation[i]
    
    ref_norm = np.linalg.norm(v_ref)
    target_norm = np.linalg.norm(anchor_vector)
    
    if ref_norm < 1e-6:
        return np.eye(2)
    
    scale = target_norm / ref_norm
    v_ref_norm = v_ref / ref_norm
    v_target_norm = anchor_vector / target_norm
    
    cos_theta = np.clip(np.dot(v_ref_norm, v_target_norm), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    if np.cross(v_ref_norm, v_target_norm) < 0:
        theta = -theta
        
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    return scale * R

def compute_aligned_target(X, target_formation):
    """计算与当前队形对齐(平移)的目标队形"""
    center_X = np.mean(X, axis=0)
    center_T = np.mean(target_formation, axis=0)
    optimal_translation = center_X - center_T
    return target_formation + optimal_translation, optimal_translation

def J_formation_to_target(X, target_formation, formation_weight=1.0):
    """形状队形成本函数"""
    aligned_target, _ = compute_aligned_target(X, target_formation)
    errors = X - aligned_target
    return formation_weight * np.sum(errors**2)

def compute_shape_error_to_target(X, target_formation):
    """计算平均形状误差"""
    aligned_target, optimal_translation = compute_aligned_target(X, target_formation)
    shape_error = np.mean(np.linalg.norm(X - aligned_target, axis=1))
    return shape_error, optimal_translation


# ========== 重构后的兼容版优化器类 ==========

class PenaltyOptimizer:
    """
    兼容分布式框架接口的罚函数优化器
    参数映射: sigma->alpha, eta->beta, tol->收敛阈值, shape_icon->reference_formation
    """
    def __init__(self, shape_icon, sigma=0.001, max_iter=3000, tol=1e-4, eta=0.9, **kwargs):
        # 1. 对齐基础参数
        self.ref_shape = shape_icon
        self.sigma = sigma      # 等同于原 alpha
        self.max_iter = max_iter
        self.tol = tol          # 更新容差
        self.eta = eta          # 等同于原 beta (动量)
        
        # 2. 接收外部罚函数特有超参数
        self.d_col = kwargs.get('d_col', 0.5)
        self.A_col = kwargs.get('A_col', 0.9)
        self.w_form = kwargs.get('w_form', 1.0)
        self.w_col = kwargs.get('w_col', 1.0)
        self.w_reg = kwargs.get('w_reg', 1000.0)
        self.robot_radius = kwargs.get('robot_radius', 0.5)
        self.EPS = kwargs.get('eps', 1e-2)
        self.anchor_pair = kwargs.get('anchor_pair', (0, 1))
        
        # 3. 内部状态初始化
        self.polygon_vertices = None
        self.polygon_edges = None
        self.polygon_normals = None
        self.target_formation = None
        self.anchor_vector = None
        
        # 用于存储优化过程记录
        self.optimal_positions = None
        self.trajectory = None
        self.history = None
        self.initial_positions = None

    def _set_polygon_region(self, vertices):
        """设置凸多边形区域并计算法向量"""
        if vertices is None:
            self.polygon_vertices = None
            return
            
        self.polygon_vertices = np.array(vertices)
        if len(self.polygon_vertices) < 3:
            raise ValueError("多边形至少需要3个顶点")
        
        hull = ConvexHull(self.polygon_vertices)
        self.polygon_vertices = self.polygon_vertices[hull.vertices]
        
        n = len(self.polygon_vertices)
        self.polygon_edges = []
        self.polygon_normals = []
        
        for i in range(n):
            p1 = self.polygon_vertices[i]
            p2 = self.polygon_vertices[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + self.EPS)
            self.polygon_edges.append(edge)
            self.polygon_normals.append(normal)

    def point_in_polygon(self, point):
        """判断点是否在多边形内"""
        if self.polygon_vertices is None:
            return True, []
        distances = []
        inside = True
        for i in range(len(self.polygon_vertices)):
            vec_to_point = point - self.polygon_vertices[i]
            distance = np.dot(vec_to_point, self.polygon_normals[i])
            distances.append(distance)
            if distance < -self.EPS:
                inside = False
        return inside, distances

    def J_col(self, X):
        return self.w_col * J_col(X, self.A_col, self.d_col, self.EPS)

    def J_formation(self, X):
        return self.w_form * J_formation_to_target(X, self.target_formation, formation_weight=1.0)

    def J_region(self, X):
        """外罚函数"""
        if self.polygon_vertices is None:
            return 0.0
        penalty = 0.0
        for i in range(len(X)):
            _, distances = self.point_in_polygon(X[i])
            for d in distances:
                if d < 0:
                    penalty += d**2
        return self.w_reg * penalty

    def compute_shape_error(self, X):
        return compute_shape_error_to_target(X, self.target_formation)

    def compute_aligned_target(self, X):
        aligned_target, _ = compute_aligned_target(X, self.target_formation)
        return aligned_target

    def gradient_J_col_finite_difference(self, X, h=1e-5):
        N = len(X)
        grad = np.zeros_like(X)
        for i in range(N):
            for j in range(2):
                X_plus, X_minus = X.copy(), X.copy()
                X_plus[i, j] += h
                X_minus[i, j] -= h
                grad[i, j] = (self.J_col(X_plus) - self.J_col(X_minus)) / (2 * h)
        return grad

    def gradient_J_formation_finite_difference(self, X, h=1e-5):
        N = len(X)
        grad = np.zeros_like(X)
        for i in range(N):
            for j in range(2):
                X_plus, X_minus = X.copy(), X.copy()
                X_plus[i, j] += h
                X_minus[i, j] -= h
                grad[i, j] = (self.J_formation(X_plus) - self.J_formation(X_minus)) / (2 * h)
        return grad

    def gradient_J_region(self, X):
        if self.polygon_vertices is None:
            return np.zeros_like(X)
        N = len(X)
        grad = np.zeros_like(X)
        for i in range(N):
            _, distances = self.point_in_polygon(X[i])
            for j, d in enumerate(distances):
                if d < 0:
                    grad[i] += 2 * self.w_reg * d * self.polygon_normals[j]
        return grad

    def compute_total_gradient(self, X):
        return (self.gradient_J_formation_finite_difference(X) + 
                self.gradient_J_col_finite_difference(X) + 
                self.gradient_J_region(X))

    def _prepare_target_by_anchor(self, b):
        """根据锚点向量b设置目标队形"""
        self.anchor_vector = b
        transformation_matrix = compute_shape_transformation(
            self.ref_shape, self.anchor_pair, b
        )
        self.target_formation = np.dot(self.ref_shape, transformation_matrix.T)

    def _clip_gradient(self, grad, max_norm=5.0):
        """
        对梯度进行模长裁剪，保持方向不变，限制步长。
        :param grad: 原始梯度矩阵 (m x 2)
        :param max_norm: 允许的最大模长（建议设在 1.0 到 10.0 之间）
        """
        # 计算每个机器人的梯度模长
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            # 等比例缩小梯度
            grad = grad * (max_norm / norm)
        return grad

    @staticmethod
    def create_square_reference_formation(side_length=1.0):
        half = side_length / 2
        return np.array([[-half, -half], [half, -half], [half, half], [-half, half]])

    def optimize(self, initial_positions, LF_vertices, b, verbose=True):
        """
        兼容的新版接口：
        :param initial_positions: 初始位置 (m x 2)
        :param LF_vertices:       区域边界 (n x 2)
        :param b:                 锚点向量 (1 x 2)
        :param verbose:           打印日志
        :return final_X:          最终收敛的位置点集
        """

        # 【新增：防御性检查】防止 b 向量为零导致缩放奇点
        if np.linalg.norm(b) < 1e-6:
            print("警告: 锚点向量 b 过小，可能导致计算崩溃。返回初始位置。")
            return {i + 1: pos for i, pos in enumerate(initial_positions)}

        # 1. 动态初始化约束与形状
        self._set_polygon_region(LF_vertices)
        self._prepare_target_by_anchor(b)
        
        N = len(initial_positions)
        X = initial_positions.copy().astype(float)
        delta_x_prev = np.zeros_like(X)
        trajectory = [X.copy()]
        
        # 记录初始化
        history = {
            'cost_formation': [self.J_formation(X)],
            'cost_col': [self.J_col(X)],
            'cost_region': [self.J_region(X)],
            'total_cost': [], 'updates': [], 'iterations': [],
            'inside_region': [], 'shape_errors': []
        }
        
        if self.polygon_vertices is not None:
            history['inside_region'].append(sum([self.point_in_polygon(x)[0] for x in X]) / N)
        history['shape_errors'].append(self.compute_shape_error(X)[0])

        if verbose:
            print(f"开始优化 | 锚点向量b: {b} | Sigma(步长): {self.sigma} | Eta(动量): {self.eta}")

        # 核心下降循环
        for iteration in range(self.max_iter):
            total_grad = self.compute_total_gradient(X)

            # 【新增：NaN 检查】防止数值爆炸
            if np.any(np.isnan(total_grad)):
                print(f"迭代 {iteration}: 梯度出现 NaN，停止优化。")
                break

            # C. 【核心修改】梯度裁剪
            # max_norm 的值可以根据你的 sigma 调整。
            # 如果 sigma 是 0.001，max_norm 设为 10.0，则单次最大位移被限制在 0.01 左右。
            total_grad = self._clip_gradient(total_grad, max_norm=10.0)
            
            # 使用映射后的 sigma(步长) 和 eta(动量)
            delta_x = -(self.sigma * total_grad + self.eta * delta_x_prev)

            # E. 【二次保险】对最终位移量 delta_x 再次进行硬裁剪
            # 防止动量积累导致突然的剧烈跳跃
            max_step = 0.5  # 限制单步最大移动 0.5 米
            step_norm = np.max(np.abs(delta_x))
            if step_norm > max_step:
                delta_x = delta_x * (max_step / step_norm)

            X = X + delta_x
            
            if iteration % 10 == 0:
                trajectory.append(X.copy())
                history['cost_formation'].append(self.J_formation(X))
                history['cost_col'].append(self.J_col(X))
                history['cost_region'].append(self.J_region(X))
                history['updates'].append(np.max(np.abs(delta_x)))
                history['iterations'].append(iteration)
                history['shape_errors'].append(self.compute_shape_error(X)[0])
                if self.polygon_vertices is not None:
                    history['inside_region'].append(sum([self.point_in_polygon(x)[0] for x in X]) / N)
            
            max_update = np.max(np.abs(delta_x))
            
            # 使用映射后的 tol
            if max_update < self.tol:
                if verbose:
                    print(f"优化收敛于迭代 {iteration}: 最大更新量 {max_update:.2e}")
                break
                
            delta_x_prev = delta_x.copy()

        # 收尾保存
        if not np.array_equal(trajectory[-1], X):
            trajectory.append(X.copy())
            
        self.optimal_positions = X
        self.trajectory = trajectory
        self.history = history
        self.initial_positions = initial_positions

        return {i + 1: pos for i, pos in enumerate(X)}

    def get_optimization_summary(self):
        """保持原有的摘要提取逻辑"""
        N = len(self.optimal_positions)
        shape_error, optimal_translation = self.compute_shape_error(self.optimal_positions)
        i, j = self.anchor_pair
        actual_anchor_vector = self.optimal_positions[j] - self.optimal_positions[i]
        anchor_angle_error = np.degrees(np.arccos(
            np.clip(np.dot(actual_anchor_vector, self.anchor_vector) / 
            (np.linalg.norm(actual_anchor_vector) * np.linalg.norm(self.anchor_vector) + 1e-6), -1.0, 1.0)
        ))
        
        return {
            'shape_error': shape_error,
            'optimal_translation': optimal_translation,
            'anchor_angle_error_deg': anchor_angle_error,
            'num_iterations': len(self.trajectory)
        }

    def get_region_violation_info(self):
        """区域违例检查"""
        if self.polygon_vertices is None:
            return {"has_constraint": False, "all_inside": True}
        outside_robots = []
        for i, pos in enumerate(self.optimal_positions):
            if not self.point_in_polygon(pos)[0]:
                outside_robots.append(i)
        return {"has_constraint": True, "all_inside": (len(outside_robots) == 0)}


# ========== 适配后的可视化和测试部分 ==========

def create_simplified_optimization_visualization(initial_positions, optimal_dict, optimizer):
    # 添加这一行：将字典转回数组以便绘图逻辑执行
    optimal_positions = np.array([optimal_dict[k] for k in sorted(optimal_dict.keys())])
    """可视化逻辑无缝对接 PenaltyOptimizer 内部属性"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    N = len(initial_positions)
    i, j = optimizer.anchor_pair
    aligned_target = optimizer.compute_aligned_target(optimal_positions)
    titles = ["初始队形", "优化结果", "形状对比", "代价函数变化", "形状误差变化", "区域检查"]
    positions_list = [initial_positions, optimal_positions, optimal_positions]
    
    for idx, (ax, title, positions) in enumerate(zip(axes[0, :], titles[:3], positions_list)):
        if optimizer.polygon_vertices is not None:
            polygon_patch = plt.Polygon(optimizer.polygon_vertices, closed=True, 
                                       facecolor='lightgreen', alpha=0.1, edgecolor='green', linewidth=2)
            ax.add_patch(polygon_patch)
        
        colors = ['green' if k in optimizer.anchor_pair else 'blue' for k in range(N)]
        sizes = [350 if k in optimizer.anchor_pair else 250 for k in range(N)]
        markers = ['s' if k in optimizer.anchor_pair else 'o' for k in range(N)]
        
        for k, pos in enumerate(positions):
            ax.scatter(pos[0], pos[1], color=colors[k], s=sizes[k], marker=markers[k], edgecolor='black', linewidths=2, alpha=0.8, zorder=10)
            label_text = f'R{k+1}' + ('(锚点)' if k in optimizer.anchor_pair else '')
            ax.annotate(label_text, xy=pos, xytext=(0, 15), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', zorder=11)
        
        if idx in [1, 2]:
            for k in range(4):
                ax.plot([positions[k][0], positions[(k+1) % 4][0]], [positions[k][1], positions[(k+1) % 4][1]], 'b-', alpha=0.6, linewidth=1.5)
        
        if idx in [0, 1]:
            actual_vector = positions[j] - positions[i]
            ax.arrow(positions[i][0], positions[i][1], actual_vector[0], actual_vector[1], head_width=0.1, head_length=0.15, fc='blue', ec='blue', alpha=0.7, linewidth=2)
            if idx == 0 and optimizer.anchor_vector is not None:
                ax.arrow(positions[i][0], positions[i][1], optimizer.anchor_vector[0], optimizer.anchor_vector[1], head_width=0.15, head_length=0.2, fc='red', ec='red', alpha=0.7, linewidth=2)
        
        if idx == 2:
            for k, pos in enumerate(aligned_target):
                ax.scatter(pos[0], pos[1], color='red', s=100, marker='x', linewidths=2, alpha=0.8, zorder=9)
            for k in range(4):
                ax.plot([aligned_target[k][0], aligned_target[(k+1) % 4][0]], [aligned_target[k][1], aligned_target[(k+1) % 4][1]], 'r--', alpha=0.6, linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    if optimizer.history is not None:
        ax1, ax2, ax3 = axes[1, 0], axes[1, 1], axes[1, 2]
        iterations = optimizer.history['iterations']
        ax1.plot(iterations, optimizer.history['cost_formation'][:len(iterations)], 'b-', label='形状代价')
        ax1.plot(iterations, optimizer.history['cost_col'][:len(iterations)], 'g-', label='碰撞代价')
        if optimizer.polygon_vertices is not None:
            ax1.plot(iterations, optimizer.history['cost_region'][:len(iterations)], 'r-', label='区域代价')
        ax1.set_title('代价函数变化'); ax1.legend()
        
        ax2.plot(iterations, optimizer.history['shape_errors'][:len(iterations)], 'm-', label='形状误差')
        ax2.set_title('形状误差变化'); ax2.legend()
        
        if optimizer.polygon_vertices is not None:
            ax3.plot(iterations, optimizer.history['inside_region'][:len(iterations)], 'c-', label='区域内比例')
        ax3.set_title('区域约束检查')

    plt.tight_layout()
    plt.show()

def test_shape_optimization_inside_region():
    print("\n" + "="*70)
    print("测试：兼容版接口 - 机器人在区域内")
    np.random.seed(42)
    
    # 构建输入参数
    LF_vertices = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]])
    shape_icon = PenaltyOptimizer.create_square_reference_formation(2.0)
    b = 2.0 * np.array([np.cos(np.radians(45)), np.sin(np.radians(45))]) # 锚点向量
    
    initial_positions = shape_icon + np.array([5.0, 4.0]) + np.random.normal(0, 0.3, shape_icon.shape)
    
    # === 使用兼容的新接口调用 (完全映射原有的超参数) ===
    optimizer = PenaltyOptimizer(
        shape_icon=shape_icon, 
        sigma=0.001,       # 原 alpha
        max_iter=1500,     # 原 max_iters
        tol=1e-4,          # 收敛精度
        eta=0.9,           # 原 beta
        w_reg=15.0,        # 原 barrier_weight
        w_form=1.0,        # 原 formation_strength
        A_col=0.9, d_col=0.5
    )
    
    # 核心执行方法，返回最终坐标
    final_X = optimizer.optimize(initial_positions, LF_vertices, b)
    
    create_simplified_optimization_visualization(initial_positions, final_X, optimizer)

def test_shape_optimization_all_outside():
    print("\n" + "="*70)
    print("测试：兼容版接口 - 机器人全在区域外")
    np.random.seed(42)
    
    LF_vertices = np.array([[1.0, 1.0], [7.0, 1.0], [7.0, 7.0], [1.0, 7.0]])
    shape_icon = PenaltyOptimizer.create_square_reference_formation(1.5)
    b = 1.5 * np.array([np.cos(np.radians(60)), np.sin(np.radians(60))])
    
    initial_positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    
    # === 增大罚函数权重的困难测试 ===
    optimizer = PenaltyOptimizer(
        shape_icon=shape_icon, 
        sigma=0.001, 
        max_iter=200000, 
        tol=1e-8, 
        eta=0.9,
        w_reg=25.0,  # 增大区域约束，对应原代码的25.0
        w_form=1.2,  # 增大形状约束，对应原代码的1.2
        A_col=0.9, d_col=0.5
    )
    
    final_X = optimizer.optimize(initial_positions, LF_vertices, b)
    
    create_simplified_optimization_visualization(initial_positions, final_X, optimizer)

def main():
    print("兼容分布式接口的 PenaltyOptimizer 测试台")
    test_shape_optimization_inside_region()
    test_shape_optimization_all_outside()

if __name__ == "__main__":
    main()