import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path

class FormationOptimizer:
    """
    基于ADMM的编队优化器，改进特性：
    1. 归一化凸包约束平面
    2. 迭代投影确保严格满足约束
    3. 移除缓存机制
    4. 精确相对位置与锚点约束处理
    """
    n = 2
    
    def __init__(self, shape_icon, sigma=0.1, lam=0.5, max_iter=1000, tol=1e-6):
        """
        初始化优化器
        :param shape_icon: (m x 2)参考形状点集
        :param sigma: ADMM步长(默认0.1)
        :param lam: 权重系数(默认0.5)
        :param max_iter: 最大迭代次数(默认1000)
        :param tol: 收敛容差(默认1e-6)
        """
        self.ref_shape = shape_icon
        self.m, self.n = self.ref_shape.shape  # m: 点数, n: 维度
        self.Gamma = np.eye(self.n)
        # 构造形状保持约束矩阵 A1
        self.A1 = self._build_shape_constraint_matrix()
        # 默认无锚点时 b = 0，对应 A1*q = 0
        self.b = np.zeros(self.A1.shape[0])
        self.A = self.A1.copy()  # 若后续扩展锚点，则 A = [A1; B]
        self.sigma = sigma
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def apply_compression(self, shape, Gamma):
        s1, s3 = shape[0],  shape[2]
        q0 = (2 * s1 + s3) / 3
        n1 = (s3 - s1) / np.linalg.norm(s3 - s1)
        n2 = (shape[3] - q0) / np.linalg.norm(shape[1] - q0)
        N = np.column_stack((n1, n2))        
        Gamma = np.array(Gamma)
        N_inv = np.linalg.inv(N)
        
        compressed_shape = (N @ (Gamma @ (N_inv @ shape.T))).T
        return compressed_shape
    
    def _build_shape_constraint_matrix(self):
        """
        构造形状保持约束矩阵 A1，形状为 ((m-2)*2, 2*m)
        参见论文中将约束转化为 A1*q = 0 的描述
        """
        s1 = self.ref_shape[0]
        s2 = self.ref_shape[1]
        #M1 = np.array([[np.linalg.norm(s2), 0],
        #               [0, np.linalg.norm(s2)]])
        M2 = np.array([[s2[0], -s2[1]],
                        [s2[1], s2[0]]])
        constraints = []
        for i in range(2, self.m):
            si = self.ref_shape[i]
            Mi = np.array([[si[0], -si[1]],
                           [si[1], si[0]]])
            row = np.zeros((2, 2 * self.m))
            row[:, :2] = Mi - M2
            row[:, 2:4] = -Mi
            row[:, 2*i:2*i+2] = M2
            constraints.append(row)
        return np.vstack(constraints)

    def set_anchor_constraint(self, free_anchor_idx1=None, free_anchor_idx2=None, relative_offset=None,
                              fixed_anchor_idx=None, fixed_anchor_value=None):
        """
        根据提供的锚点信息，扩展原有约束矩阵 A1 与向量 b。
        当固定锚点提供时（如 fixed_anchor_idx 与 fixed_anchor_value），添加固定约束；
        当提供自由锚点（free_anchor_idx1, free_anchor_idx2, relative_offset）时，
        添加 q[free_anchor_idx1] - q[free_anchor_idx2] = relative_offset 的约束。
        """
        additional_rows = []
        additional_b = []
        if fixed_anchor_idx is not None and fixed_anchor_value is not None:
            # 固定锚点：例如 q_i = fixed_anchor_value
            row = np.zeros((self.n, 2 * self.m))
            start = 2 * fixed_anchor_idx
            row[:, start:start + 2] = np.eye(self.n)
            additional_rows.append(row)
            additional_b.append(fixed_anchor_value)
        if free_anchor_idx1 is not None and free_anchor_idx2 is not None and relative_offset is not None:
            # 自由锚点： q[free_anchor_idx1] - q[free_anchor_idx2] = relative_offset
            row = np.zeros((self.n, 2 * self.m))
            start1 = 2 * free_anchor_idx1
            start2 = 2 * free_anchor_idx2
            row[:, start1:start1+2] = -np.eye(self.n)
            row[:, start2:start2+2] = np.eye(self.n)
            additional_rows.append(row)
            relative_offset=relative_offset/np.linalg.norm(relative_offset)*np.linalg.norm(self.ref_shape[2] - self.ref_shape[0])
            additional_b.append(relative_offset)
        if additional_rows:
            B = np.vstack(additional_rows)
            b_anchor = np.hstack(additional_b)
            # 更新 A 和 b：将 A1 与 B 叠加
            self.A = np.vstack([self.A1, B])
            self.b = np.hstack([self.b, b_anchor])

    # phi函数：区域投影算子，投影到由LF_vertices构成的凸多边形边界上
    def phi(self, u):
        """
        区域投影算子 phi，将输入 u（展平向量，代表 m 个机器人坐标）投影到可行区域 Omega 内。
        这里调用 _project_to_region_iterative 完成投影操作。
        """
        return self._project_to_region_iterative(u, self.LF_vertices)

    # psi函数：逐元素截断到 [-1, 1] 区间
    def psi(self, u):
        """
        投影算子 psi，将输入 u 逐元素限制到 [-1, 1] 区间
        """
        return np.clip(u, -1, 1)

    def _project_to_constraints(self, q_flat):
        """投影到形状及锚点约束空间，即求解 q 使得 A*q = b"""
        A_pinv = np.linalg.pinv(self.A)
        return q_flat - A_pinv @ (self.A @ q_flat - self.b)

    def _project_to_region_iterative(self, q_flat, LF_vertices, max_iters=100, tol=1e-6):
        """
        迭代区域投影核心算法：
        对于每个点，先计算由LF_vertices构成的凸包平面方程（归一化后），
        然后选取违反条件中距离最大的平面，将点投影到该平面上，
        直到所有平面距离均小于 tol 为止。
        """
        # 构造凸包
        hull = ConvexHull(LF_vertices, qhull_options='QJ')
        q = q_flat.reshape(-1, 2)
        proj_q = np.copy(q)
        
        # 归一化平面方程 [A, B, C] → [A/||(A,B)||, B/||(A,B)||, C/||(A,B)||]
        norms = np.linalg.norm(hull.equations[:, :-1], axis=1, keepdims=True)
        normalized_equations = hull.equations / norms
        
        for i in range(q.shape[0]):
            point = q[i].copy()  # 当前点
            for _ in range(max_iters):
                # 计算到所有平面的真实距离
                distances = np.dot(normalized_equations[:, :-1], point) + normalized_equations[:, -1]
                violating = distances > tol  # 违反条件的平面
                if not np.any(violating):
                    break  # 点已在区域内，退出迭代
                
                # 找到最远违规平面
                farthest_idx = np.argmax(distances[violating])
                # 得到violating平面中的所有满足条件的平面，然后取最远的那个
                viol_eq = normalized_equations[violating]
                A, B, C = viol_eq[farthest_idx]
                distance = distances[violating][farthest_idx]
                
                # 投影到该平面：point = point - distance * [A, B]
                point -= distance * np.array([A, B])
            proj_q[i] = point
        
        return proj_q.flatten()

    def optimize(self, initial_positions, LF_vertices, 
                 Gamma = np.eye(n),
                 target_center=None, 
                 free_anchor_idx1=None, free_anchor_idx2=None, relative_offset=None,
                 fixed_anchor_idx=None, fixed_anchor_value=None):
        """
        主优化流程
        :param initial_positions: 初始位置，形状为 (m,2)，即 m 个目标位置堆叠展平得到 p
        :param LF_vertices: 可行区域凸包顶点
        :param target_center: 若提供目标中心，则用于整体平移（可选）
        :param free_anchor_idx1, free_anchor_idx2, relative_offset: 自由锚点约束参数（可选）
        :param fixed_anchor_idx, fixed_anchor_value: 固定锚点约束参数（可选）
        :return: 优化后的点集 (m x 2)
        """
        #根据输入调整参考形状
        if not np.array_equal(Gamma, self.Gamma):
            self.ref_shape = self.apply_compression(self.ref_shape, Gamma)
            self.A1 = self._build_shape_constraint_matrix()
        # 保存可行区域信息供 phi 使用
        self.LF_vertices = LF_vertices
        # 初始位置 p 使用 m 个目标位置堆叠展平得到
        p_flat = initial_positions.flatten()
        
        # 设置锚点约束（如果有）
        self.set_anchor_constraint(free_anchor_idx1, free_anchor_idx2, relative_offset,
                                   fixed_anchor_idx, fixed_anchor_value)
        
        # 若提供 target_center，则对 p 平移，使初始解更靠近目标中心，否则直接使用 p_flat
        if target_center is not None:
            shift = target_center - np.mean(initial_positions, axis=0)
            p_flat = p_flat + np.tile(shift, self.m)

        q_k = p_flat.copy()
        y_k = np.zeros(self.A.shape[0])
        z_k = np.zeros_like(q_k)

        # ADMM 主循环
        for k in range(self.max_iter):
            # 更新 q：注意引入了 p_flat (初始形成) 和约束项 (A*q - b)
            q_next = self.phi(
                q_k - self.sigma * (
                    self.lam * (q_k - p_flat) +
                    (1 - self.lam) * self.psi(z_k + q_k - p_flat) +
                    self.A.T @ (y_k + (self.A @ q_k - self.b))
                )
            )
            q_next_flat = q_next
            # 双重投影：先投影到约束 A*q=b，再投q_next影到可行区域（LF_vertices）
            #q_next_flat = self._project_to_constraints(q_next.flatten())
            #q_next_flat = self._project_to_region_iterative(q_next_flat, LF_vertices)
            
            # 对偶变量更新
            y_k += self.A @ q_next_flat - self.b
            z_k = self.psi(z_k + q_next_flat - p_flat)
            
            if np.linalg.norm(q_next_flat - q_k) < self.tol:
                q_k = q_next_flat
                break
            q_k = q_next_flat
        
        return q_k.reshape(-1, 2)

def visualize(LF_vertices, initial_positions, optimal_positions, compressed_shape, target_center, ref_shape):
    """可视化优化结果"""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_positions)))
    
    hull = ConvexHull(LF_vertices, qhull_options='QJ')
    plt.fill(LF_vertices[hull.vertices, 0], LF_vertices[hull.vertices, 1],
             'lightgray', alpha=0.5, label='Feasible Region')
    
    plt.scatter(initial_positions[:, 0], initial_positions[:, 1],
                c=colors, marker='*', s=150, edgecolor='k', label='Initial')
    plt.scatter(optimal_positions[:, 0], optimal_positions[:, 1],
                c=colors, marker='o', s=100, label='Optimized')
    
    plt.scatter(compressed_shape[:,0], compressed_shape[:,1], color='green', label='Compressed Shape')
    
    hull_ref = ConvexHull(ref_shape, qhull_options='QJ')
    for simplex in hull_ref.simplices:
        plt.plot(optimal_positions[simplex, 0], optimal_positions[simplex, 1],
                 'g--', lw=1.5, alpha=0.7)
    
    if target_center is not None:
        plt.scatter(target_center[0], target_center[1], marker='X',
                    s=250, c='red', label='Target Center')
    plt.scatter(*np.mean(optimal_positions, axis=0), marker='P',
                s=250, c='blue', label='Actual Center')
    
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.title('Formation Optimization Result\n(Improved Projection Operator)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试配置
    np.random.seed(42)
    LF_vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    ref_shape = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    # 初始位置为 4 个目标位置
    initial_positions = np.array([[5, 5], [6, 5], [6, 6], [5, 6]])
    target_center = np.array([13, 4])
    
    
    # 创建优化器
    optimizer = FormationOptimizer(ref_shape, 
                                   sigma=0.2, 
                                   max_iter=500,
                                   )
    
    # 例如：若设定自由锚点约束 q0 - q1 = relative_offset
    result = optimizer.optimize(
        initial_positions=initial_positions,
        LF_vertices=LF_vertices,
        Gamma = np.array([[1,0],
                          [0,1]]),
        target_center=target_center,
        free_anchor_idx1=0,
        free_anchor_idx2=2,
        relative_offset=np.array([0, 1])
    )
    
    print("优化结果点集：\n", result)
    print("形状约束误差：", np.linalg.norm(optimizer.A @ result.flatten() - optimizer.b))
    
    visualize(LF_vertices, initial_positions, result, optimizer.ref_shape, target_center, ref_shape)