import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import networkx as nx

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = True  # 改为True并确保字体支持

class DistributedFormationOptimizer:
    """
    基于论文的完全分布式编队优化器
    A Distributed Projection-Based Algorithm with Local Estimators for Optimal Formation of Multi-robot System
    """
    
    def __init__(self, shape_icon, sigma=0.1, max_iter=1000, tol=1e-2, eta=0.2):
        """
        初始化分布式优化器
        :param shape_icon: 参考形状点集 (m x 2)
        :param sigma: 步长参数
        :param max_iter: 最大迭代次数
        :param tol: 收敛容差
        """
        self.ref_shape = shape_icon
        self.m, self.n = self.ref_shape.shape  # m: 机器人数量, n: 维度(2)
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        
        # 计算形状参数矩阵 M 和 M_i
        self._compute_shape_matrices()
        
        # 初始化局部估计器和状态变量
        self._initialize_local_estimators()
        
        # 通信拓扑
        self.L = None  # 拉普拉斯矩阵
        self.adjacency_matrix = None
        
        # 锚点约束
        self.anchor_b = None
    
    def _compute_shape_matrices(self):
        """计算形状参数矩阵 M 和 M_i (论文中的定义)"""
        s2 = self.ref_shape[1]
        # M = ||s2|| * I_2
        #self.M = np.eye(2) * np.linalg.norm(s2)
        
        self.M_i = {}
        for i in range(2, self.m+1):
            s_i = self.ref_shape[i-1]
            # M_i = [s_i^x, -s_i^y; s_i^y, s_i^x]
            self.M_i[i] = np.array([[s_i[0], -s_i[1]],
                                  [s_i[1], s_i[0]]])
    
    def _initialize_local_estimators(self):
        """初始化所有机器人的局部估计器"""
        # 存储所有机器人的状态
        self.positions = {}      # 真实位置 p_i
        self.estimates = {}      # 局部估计值
        self.dual_vars = {}      # 对偶变量
        
        # 初始化机器人1 (特殊处理)
        self.positions[1] = None
        self.estimates[1] = {'p12': None}  # 只估计机器人2
        self.dual_vars[1] = {
            'y1': np.zeros(2), 
            'z1': np.zeros(2), 
            'w12': np.zeros(2)
        }
        
        # 初始化机器人2 (特殊处理)  
        self.positions[2] = None
        self.estimates[2] = {'p21': None}  # 只估计机器人1
        self.dual_vars[2] = {
            'y2': np.zeros(2), 
            'z21': np.zeros(2), 
            'w2': np.zeros(2)
        }
        
        # 初始化其他机器人 (i=3,...,m)
        for i in range(3, self.m + 1):
            self.positions[i] = None
            self.estimates[i] = {
                'p_i1': None,  # 对机器人1的估计
                'p_i2': None   # 对机器人2的估计
            }
            self.dual_vars[i] = {
                'y_i': np.zeros(2), 
                'z_i1': np.zeros(2), 
                'w_i2': np.zeros(2)
            }
    
    def _build_distributed_constraints(self):
        """
        按照论文公式构建分布式约束矩阵 B_i
        B_i = [M, M_i - M, -M_i] (论文公式中的定义)
        """
        B_matrices = {}
        
        for i in range(3, self.m+1):
            # B_i = [M, M_i - M, -M_i]
            B_i = np.hstack([
                self.M_i[2], 
                self.M_i[i] - self.M_i[2], 
                -self.M_i[i]
            ])
            B_matrices[i] = B_i
            
        return B_matrices
    
    def set_communication_topology(self, adjacency_matrix):
        """
        设置通信拓扑
        :param adjacency_matrix: 邻接矩阵 (m x m)
        """
        self.adjacency_matrix = adjacency_matrix
        # 计算拉普拉斯矩阵 L = D - A
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        self.L = degree_matrix - adjacency_matrix
    
    def set_anchor_constraint(self, b):
        """
        设置锚点约束 p1 - p2 = b
        :param b: 锚点向量
        """
        self.anchor_b = b
    
    def _get_neighbor_estimates(self, robot_id):
        """
        获取邻居机器人的估计值
        :param robot_id: 当前机器人ID
        :return: 邻居对机器人1和2的估计值字典
        """
        neighbors = np.where(self.adjacency_matrix[robot_id-1] != 0)[0] + 1
        
        neighbor_estimates = {}
        for j in neighbors:
            if j == 1:
                # 机器人1的真实位置和估计
                neighbor_estimates[j] = {
                    'p_j1': self.positions[1],  # 机器人1知道自己的真实位置
                    'p_j2': self.estimates[1]['p12']  # 机器人1对机器人2的估计
                }
            elif j == 2:
                # 机器人2的真实位置和估计
                neighbor_estimates[j] = {
                    'p_j1': self.estimates[2]['p21'],  # 机器人2对机器人1的估计
                    'p_j2': self.positions[2]  # 机器人2知道自己的真实位置
                }
            else:
                # 其他机器人的估计
                neighbor_estimates[j] = {
                    'p_j1': self.estimates[j]['p_i1'],
                    'p_j2': self.estimates[j]['p_i2']
                }
        
        return neighbor_estimates
    
    def _update_robot1(self, initial_positions, LF_vertices, k):
        """
        按照论文公式(13)更新机器人1
        """
        i = 1
        p1 = self.positions[i]
        p12 = self.estimates[i]['p12']
        y1 = self.dual_vars[i]['y1']
        z1 = self.dual_vars[i]['z1']
        w12 = self.dual_vars[i]['w12']
        xi1 = initial_positions[i-1]
        
        # 获取邻居估计值
        neighbor_estimates = self._get_neighbor_estimates(i)
        
        # 计算一致性项 (公式中的求和项)
        consensus_p1 = np.zeros(2)   # 对于p1的一致性项
        consensus_p12 = np.zeros(2)  # 对于p12的一致性项
        
        for j, estimates in neighbor_estimates.items():
            weight = -self.L[i-1, j-1]
            
            if estimates['p_j1'] is not None:
                consensus_p1 += weight * (p1 - estimates['p_j1'])
            if estimates['p_j2'] is not None:
                consensus_p12 += weight * (p12 - estimates['p_j2'])
        
        # 机器人1的位置更新 (公式13第一行)
        p1_new = self._project_to_region(
            p1 - self.sigma * (
                (p1 - xi1) + y1 + (p1 - self.anchor_b - p12) + z1 + consensus_p1
            ), LF_vertices
        )
        
        # 机器人1对机器人2的估计更新 (公式13第二行)
        p12_new = self._project_to_region(
            p12 - self.sigma * (
                -y1 - (p1 - p12 - self.anchor_b) + w12 + consensus_p12
            ), LF_vertices
        )
        
        # 更新对偶变量 (公式13第三、四、五行)
        y1_new = y1 + self.eta*(p1_new - p12_new - self.anchor_b)
        z1_new = z1 + self.eta*consensus_p1
        w12_new = w12 + self.eta*consensus_p12
        
        # 保存更新后的值
        self.positions[i] = p1_new
        self.estimates[i]['p12'] = p12_new
        self.dual_vars[i]['y1'] = y1_new
        self.dual_vars[i]['z1'] = z1_new
        self.dual_vars[i]['w12'] = w12_new
    
    def _update_robot2(self, initial_positions, LF_vertices, k):
        """
        按照论文公式(14)更新机器人2
        """
        i = 2
        p2 = self.positions[i]
        p21 = self.estimates[i]['p21']
        y2 = self.dual_vars[i]['y2']
        z21 = self.dual_vars[i]['z21']
        w2 = self.dual_vars[i]['w2']
        xi2 = initial_positions[i-1]
        
        # 获取邻居估计值
        neighbor_estimates = self._get_neighbor_estimates(i)
        
        # 计算一致性项
        consensus_p2 = np.zeros(2)   # 对于p2的一致性项
        consensus_p21 = np.zeros(2)  # 对于p21的一致性项
        
        for j, estimates in neighbor_estimates.items():
            weight = -self.L[i-1, j-1]
            
            if estimates['p_j2'] is not None:
                consensus_p2 += weight * (p2 - estimates['p_j2'])
            if estimates['p_j1'] is not None:
                consensus_p21 += weight * (p21 - estimates['p_j1'])
        
        # 机器人2的位置更新 (公式14第一行)
        p2_new = self._project_to_region(
            p2 - self.sigma * (
                (p2 - xi2) + y2 + (p2 + self.anchor_b - p21) + w2 + consensus_p2
            ), LF_vertices
        )
        
        # 机器人2对机器人1的估计更新 (公式14第二行)
        p21_new = self._project_to_region(
            p21 - self.sigma * (
                -y2 - (p2 - p21 + self.anchor_b) + z21 + consensus_p21
            ), LF_vertices
        )
        
        # 更新对偶变量 (公式14第三、四、五行)
        y2_new = y2 +  self.eta*(p2_new - p21_new + self.anchor_b)
        z21_new = z21 + self.eta*consensus_p21
        w2_new = w2 + self.eta*consensus_p2
        
        # 保存更新后的值
        self.positions[i] = p2_new
        self.estimates[i]['p21'] = p21_new
        self.dual_vars[i]['y2'] = y2_new
        self.dual_vars[i]['z21'] = z21_new
        self.dual_vars[i]['w2'] = w2_new
    
    def _update_robot_i(self, i, initial_positions, LF_vertices, k, B_matrices):
        """
        按照论文公式(15)更新机器人i (i=3,...,m)
        """
        p_i = self.positions[i]
        p_i1 = self.estimates[i]['p_i1']
        p_i2 = self.estimates[i]['p_i2']
        
        y_i = self.dual_vars[i]['y_i']
        z_i1 = self.dual_vars[i]['z_i1']
        w_i2 = self.dual_vars[i]['w_i2']
        xi_i = initial_positions[i-1]
        
        B_i = B_matrices[i]
        p_tilde_i = np.hstack([p_i, p_i1, p_i2])
        
        # 获取邻居估计值
        neighbor_estimates = self._get_neighbor_estimates(i)
        
        # 计算一致性项
        consensus_p_i1 = np.zeros(2)  # 对于p_i1的一致性项
        consensus_p_i2 = np.zeros(2)  # 对于p_i2的一致性项
        
        for j, estimates in neighbor_estimates.items():
            weight = -self.L[i-1, j-1]
            
            if estimates['p_j1'] is not None:
                consensus_p_i1 += weight * (p_i1 - estimates['p_j1'])
            if estimates['p_j2'] is not None:
                consensus_p_i2 += weight * (p_i2 - estimates['p_j2'])
        
        # 计算 B_i 相关项
        B_p_tilde = B_i @ p_tilde_i
        y_B_term = y_i + B_p_tilde
        
        # 机器人i的位置更新 (公式15第一行)
        p_i_new = self._project_to_region(
            p_i - self.sigma * (
                (p_i - xi_i) + self.M_i[2].T @ y_B_term
            ), LF_vertices
        )
        
        # 机器人i对机器人1的估计更新 (公式15第二行)
        p_i1_new = self._project_to_region(
            p_i1 - self.sigma * (
                (self.M_i[i].T - self.M_i[2].T) @ y_B_term + z_i1 + consensus_p_i1
            ), LF_vertices
        )
        
        # 机器人i对机器人2的估计更新 (公式15第三行)
        p_i2_new = self._project_to_region(
            p_i2 - self.sigma * (
                -self.M_i[i].T @ y_B_term + w_i2 + consensus_p_i2
            ), LF_vertices
        )
        
        # 更新对偶变量 (公式15第四、五、六行)
        p_tilde_i_new = np.hstack([p_i_new, p_i1_new, p_i2_new])
        y_i_new = y_i + self.eta*(B_i @ p_tilde_i_new)
        z_i1_new = z_i1 + self.eta*consensus_p_i1
        w_i2_new = w_i2 + self.eta*consensus_p_i2
        
        # 保存更新后的值
        self.positions[i] = p_i_new
        self.estimates[i]['p_i1'] = p_i1_new
        self.estimates[i]['p_i2'] = p_i2_new
        self.dual_vars[i]['y_i'] = y_i_new
        self.dual_vars[i]['z_i1'] = z_i1_new
        self.dual_vars[i]['w_i2'] = w_i2_new
    
    def _project_to_region(self, point, LF_vertices, max_iters=50, tol=1e-6):
        
        """
        将点投影到可行区域（凸多边形）
        :param point: 要投影的点
        :param LF_vertices: 可行区域顶点
        :param max_iters: 最大迭代次数
        :param tol: 容差
        :return: 投影后的点
        """
        hull = ConvexHull(LF_vertices, qhull_options='QJ')
        
        # 归一化平面方程 [A, B, C] → [A/||(A,B)||, B/||(A,B)||, C/||(A,B)||]
        norms = np.linalg.norm(hull.equations[:, :-1], axis=1, keepdims=True)
        normalized_equations = hull.equations / norms
        
        projected_point = point.copy()
        
        for _ in range(max_iters):
            #print(projected_point, end=' -> ')
            # 计算到所有平面的距离: Ax + By + C
            distances = (np.dot(normalized_equations[:, :-1], projected_point) + 
                        normalized_equations[:, -1])
            
            violating = distances > tol  # 违反条件的平面
            if not np.any(violating):
                break
            
            # 找到最远违规平面
            farthest_idx = np.argmax(distances[violating])
            viol_eq = normalized_equations[violating][farthest_idx]
            A, B, C = viol_eq
            distance = distances[violating][farthest_idx]
            
            # 投影到该平面: point = point - distance * [A, B]
            projected_point -= distance * np.array([A, B])
        #print(projected_point)

        
        return projected_point
    
    def _initialize_positions_and_estimates(self, initial_positions):
        """
        初始化位置和估计值
        :param initial_positions: 初始位置数组 (m x 2)
        """
        # 初始化真实位置
        for i in range(1, self.m+1):
            self.positions[i] = initial_positions[i-1].copy()
        
        # 初始化估计值 - 使用保守策略
        # 机器人1对机器人2的估计
        self.estimates[1]['p12'] = initial_positions[0].copy()  # 用机器人2的初始位置
        
        # 机器人2对机器人1的估计
        self.estimates[2]['p21'] = initial_positions[1].copy()  # 用机器人1的初始位置
        
        # 其他机器人对机器人1和2的估计
        for i in range(3, self.m + 1):
            self.estimates[i]['p_i1'] = initial_positions[i-1].copy()  # 用自己的位置初始化
            self.estimates[i]['p_i2'] = initial_positions[i-1].copy()  # 用自己的位置初始化
    
    def optimize_distributed(self, initial_positions, LF_vertices, adjacency_matrix, b):
        """
        分布式优化主函数
        :param initial_positions: 初始位置 (m x 2)
        :param LF_vertices: 可行区域顶点
        :param adjacency_matrix: 通信邻接矩阵
        :param b: 锚点约束向量
        :return: 优化后的位置字典
        """
        # 设置通信拓扑和锚点约束
        self.set_communication_topology(adjacency_matrix)
        self.set_anchor_constraint(b)
        
        # 初始化位置和估计值
        self._initialize_positions_and_estimates(initial_positions)
        
        # 构建分布式约束矩阵
        B_matrices = self._build_distributed_constraints()
        
        # 记录收敛历史
        self.convergence_history = []
        
        print("开始分布式优化...")
        
        # 分布式优化主循环
        for k in range(self.max_iter):
            # 保存旧位置用于收敛检查
            positions_old = {i: self.positions[i].copy() for i in self.positions}
            
            # 按照论文顺序更新所有机器人
            self._update_robot1(initial_positions, LF_vertices, k)
            self._update_robot2(initial_positions, LF_vertices, k)
            
            for i in range(3, self.m+1):
                self._update_robot_i(i, initial_positions, LF_vertices, k, B_matrices)
            
            # 检查收敛性
            max_change = 0
            for i in range(1, self.m + 1):
                change = np.linalg.norm(self.positions[i] - positions_old[i])
                max_change = max(max_change, change)
            
            self.convergence_history.append(max_change)
            
            if k and  k % 100 == 0:
                print(f"迭代 {k}: 最大变化 = {max_change:.6f}")
            
            if max_change < self.tol:
                print(f"在第 {k} 次迭代收敛")
                break
        
        if k == self.max_iter - 1:
            print(f"达到最大迭代次数 {self.max_iter}")
        
        return self.positions
    
    def get_formation_error(self):
        """计算队形误差"""
        # 将位置转换为数组
        positions_array = np.array([self.positions[i] for i in range(1, self.m + 1)])
        
        # 计算形状约束误差
        error = 0
        for i in range(3, self.m+1):
            # 检查形状约束 M(p_i - p_1) = M_i(p_2 - p_1)
            left = self.M_i[2] @ (positions_array[i-1] - positions_array[0])
            
            right = self.M_i[i] @ (positions_array[1] - positions_array[0])
            
            error += np.linalg.norm(left - right)
        
        return error
    
    def get_consensus_error(self):
        """计算估计值的一致性误差"""
        consensus_error = 0
        
        # 检查所有机器人对机器人1的估计是否一致
        p1_estimates = []
        for i in range(1, self.m + 1):
            if i == 1:
                p1_estimates.append(self.positions[1])  # 机器人1的真实位置
            elif i == 2:
                p1_estimates.append(self.estimates[2]['p21'])  # 机器人2对机器人1的估计
            else:
                p1_estimates.append(self.estimates[i]['p_i1'])  # 其他机器人对机器人1的估计
        
        # 计算方差作为一致性误差
        p1_array = np.array(p1_estimates)
        consensus_error += np.sum(np.var(p1_array, axis=0))
        
        return consensus_error


def create_ring_topology(m):
    """创建环状通信拓扑"""
    adjacency_matrix = np.zeros((m, m))
    for i in range(m):
        adjacency_matrix[i, (i-1) % m] = 1
        adjacency_matrix[i, (i+1) % m] = 1
    return adjacency_matrix


def create_star_topology(m):
    """创建星形通信拓扑"""
    adjacency_matrix = np.zeros((m, m))
    # 机器人1作为中心
    for i in range(1, m):
        adjacency_matrix[0, i] = 1
        adjacency_matrix[i, 0] = 1
    return adjacency_matrix


def create_line_topology(m):
    """创建线性通信拓扑"""
    adjacency_matrix = np.zeros((m, m))
    for i in range(m-1):
        adjacency_matrix[i, i+1] = 1
        adjacency_matrix[i+1, i] = 1
    return adjacency_matrix


def visualize_communication_topology(adjacency_matrix, positions, title="通信拓扑"):
    """可视化通信拓扑"""
    plt.figure(figsize=(8, 6))
    
    G = nx.from_numpy_array(adjacency_matrix)
    pos_dict = {i: positions[i] for i in range(len(positions))}
    
    nx.draw(G, pos_dict, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold', 
            edge_color='gray', arrows=False)
    
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_results(initial_positions, final_positions, LF_vertices, ref_shape, 
                           convergence_history=None, topology_type="环状"):
    """修复布局问题的可视化函数"""
    
    # 创建图形时指定布局引擎
    fig = plt.figure(figsize=(15, 12))
    
    # 使用 gridspec 进行更精确的布局控制
    gs = fig.add_gridspec(2, 2)
    
    # 转换为数组形式用于绘图
    initial_array = np.array([initial_positions[i] for i in range(len(initial_positions))])
    final_array = np.array([final_positions[i] for i in sorted(final_positions.keys())])
    
    # 1. 初始位置和通信拓扑
    ax1 = fig.add_subplot(gs[0, 0])
    hull = ConvexHull(LF_vertices)
    ax1.fill(LF_vertices[hull.vertices, 0], LF_vertices[hull.vertices, 1], 
             'lightgray', alpha=0.5, label='Feasible Region')
    
    ax1.scatter(initial_array[:, 0], initial_array[:, 1], c='red', 
                marker='*', s=200, label='Initial Positions', edgecolors='black')
    
    for i, pos in enumerate(initial_array):
        ax1.text(pos[0] + 0.2, pos[1] + 0.2, f'R{i+1}', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax1.set_title(f'Initial Positions ({topology_type} Topology)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 最终位置和参考形状
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill(LF_vertices[hull.vertices, 0], LF_vertices[hull.vertices, 1], 
             'lightgray', alpha=0.5, label='Feasible Region')
    
    ax2.scatter(final_array[:, 0], final_array[:, 1], c='blue', 
                marker='s', s=150, label='Optimized Positions', edgecolors='black')
    
    # 绘制参考形状 - 修复：使用正确的参考形状绘制
    if len(ref_shape) == len(final_array):
        hull_ref = ConvexHull(ref_shape)
        for simplex in hull_ref.simplices:
            ax2.plot(ref_shape[simplex, 0], ref_shape[simplex, 1], 
                     'g--', lw=2, alpha=0.7, label='Reference Shape' if simplex[0] == 0 else "")
    
    for i, pos in enumerate(final_array):
        ax2.text(pos[0] + 0.2, pos[1] + 0.2, f'R{i+1}', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax2.set_title('Distributed Optimization Results')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 收敛历史
    if convergence_history is not None:
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.semilogy(convergence_history, 'b-', linewidth=2)
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Max Position Change (log scale)')
        ax3.set_title('Convergence History')
        ax3.grid(True, alpha=0.3)
        
        # 标记收敛点
        if convergence_history:
            conv_idx = len(convergence_history) - 1
            ax3.plot(conv_idx, convergence_history[conv_idx], 'ro', markersize=8)
            ax3.annotate(f'Converged: {convergence_history[conv_idx]:.2e}', 
                        xy=(conv_idx, convergence_history[conv_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 4. 初始vs最终位置对比
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill(LF_vertices[hull.vertices, 0], LF_vertices[hull.vertices, 1], 
             'lightgray', alpha=0.3, label='Feasible Region')
    
    # 绘制初始位置
    ax4.scatter(initial_array[:, 0], initial_array[:, 1], c='red', 
                marker='*', s=150, label='Initial Positions', alpha=0.7)
    
    # 绘制最终位置
    ax4.scatter(final_array[:, 0], final_array[:, 1], c='blue', 
                marker='s', s=100, label='Optimized Positions', alpha=0.7)
    
    
    
    # 绘制参考形状 - 修复：使用正确的参考形状绘制
    if len(ref_shape) == len(final_array):
        hull_ref = ConvexHull(ref_shape)
        for simplex in hull_ref.simplices:
            ax4.plot(ref_shape[simplex, 0], ref_shape[simplex, 1], 
                     'g-', lw=2, alpha=0.7, label='Reference Shape' if simplex[0] == 0 else "")
    
    ax4.set_title('Position Change Trajectories')
    ax4.axis('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 使用 tight_layout 并增加边距
    plt.tight_layout(pad=3.0)
    plt.show()



def visualize_communication_topology_fixed(adjacency_matrix, positions, title="Communication Topology"):
    """修复通信拓扑可视化函数"""
    plt.figure(figsize=(8, 6))
    
    # 创建图对象
    G = nx.Graph()
    n = len(positions)
    G.add_nodes_from(range(n))
    
    # 添加边
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] != 0 and i != j:
                G.add_edge(i, j)
    
    # 创建布局
    if n == 4:
        pos = {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]}
    else:
        pos = nx.spring_layout(G)
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=800, font_size=12, font_weight='bold',
            edge_color='gray', width=2, arrows=False)
    
    plt.title(title, fontsize=14)
    plt.axis('equal')
    
    # 使用 constrained_layout
    plt.tight_layout()  # 对于简单图形，tight_layout 通常没问题
    plt.show()

def test_distributed_optimization():
    """测试分布式优化算法"""
    np.random.seed(42)
    
    print("=" * 60)
    print("分布式编队优化算法测试")
    print("=" * 60)
    
    # 配置参数
    m = 4  # 6个机器人
    ref_shape = np.array([
        [0, 0],   # 机器人1参考位置
        [1, 0],   # 机器人2参考位置
        #[2, 1],   # 机器人3参考位置
        #[1, 2],   # 机器人4参考位置
        [0, 1],   # 机器人5参考位置
        [1, 1]    # 机器人6参考位置
    ])  # 六边形形状
    
    # 初始位置 (随机生成在可行区域内)
    initial_positions = np.array([
        [1, 1],    # 机器人1
        [2, 1],    # 机器人2
        [2.5, 2],  # 机器人3
        [2, 3],    # 机器人4
        #[1, 3],    # 机器人5
        #[1.5, 2]   # 机器人6
    ])
    
    # 可行区域
    LF_vertices = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
    
    # 选择通信拓扑
    topology_choice = "ring"  # 可选: "ring", "star", "line"
    
    if topology_choice == "ring":
        adjacency_matrix = create_ring_topology(m)
        topology_name = "环状"
    elif topology_choice == "star":
        adjacency_matrix = create_star_topology(m)
        topology_name = "星形"
    else:  # line
        adjacency_matrix = create_line_topology(m)
        topology_name = "线性"
    
    # 锚点约束
    b = np.array([0, 2])  # p1 - p2 = [1.5, 0.5]
    
    print(f"机器人数量: {m}")
    print(f"通信拓扑: {topology_name}")
    print(f"锚点约束: p1 - p2 = {b}")
    print(f"参考形状: {ref_shape.shape}")
    print()
    
    # 创建优化器
    optimizer = DistributedFormationOptimizer(
        ref_shape, 
        sigma=0.1,  # 较小的步长确保稳定性
        max_iter=1000,
        tol=1e-3,
        eta=0.1
    )
    
    # 运行分布式优化
    result_positions = optimizer.optimize_distributed(
        initial_positions, LF_vertices, adjacency_matrix, b
    )
    
    #print(optimizer._project_to_region(np.array([3.0, 3.0]), LF_vertices))





    # 输出结果
    print("\n优化结果:")
    dis = 0
    for i in range(1, m + 1):
        initial = initial_positions[i-1]
        final = result_positions[i]
        movement = np.linalg.norm(final - initial)
        print(f"机器人{i}: {initial} → {final} (移动距离: {movement:.3f})")
        dis += movement*movement
    print(f'f(p)={dis/2}')
    
    # 计算性能指标
    formation_error = optimizer.get_formation_error()
    consensus_error = optimizer.get_consensus_error()
    
    print(f"\n性能指标:")
    print(f"队形误差: {formation_error:.6f}")
    print(f"一致性误差: {consensus_error:.6f}")
    print(f"收敛迭代次数: {len(optimizer.convergence_history)}")
    
    # 可视化通信拓扑
    #visualize_communication_topology(
    #    adjacency_matrix, initial_positions, 
    #    f"{topology_name}通信拓扑"
    #)
    
    # 可视化优化结果
    visualize_results(
        initial_positions, result_positions, LF_vertices, ref_shape,
        optimizer.convergence_history, topology_name
    )
    
    return optimizer, result_positions


def compare_topologies():
    """比较不同通信拓扑的性能"""
    print("=" * 60)
    print("不同通信拓扑性能比较")
    print("=" * 60)
    
    # 固定参数
    m = 5
    ref_shape = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    LF_vertices = np.array([[0, 0], [3, 0], [3, 3], [0, 3]])
    b = np.array([1.0, 0.0])
    
    # 不同拓扑
    topologies = {
        "环状": create_ring_topology(m),
        "星形": create_star_topology(m),
        "线性": create_line_topology(m)
    }
    
    results = {}
    
    for name, adj_matrix in topologies.items():
        print(f"\n测试 {name} 拓扑...")
        
        # 初始位置
        initial_positions = np.array([
            [0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [1.0, 1.0]
        ])
        
        optimizer = DistributedFormationOptimizer(
            ref_shape, sigma=0.05, max_iter=500, tol=1e-4
        )
        
        result_positions = optimizer.optimize_distributed(
            initial_positions, LF_vertices, adj_matrix, b
        )
        
        results[name] = {
            'optimizer': optimizer,
            'positions': result_positions,
            'iterations': len(optimizer.convergence_history),
            'final_error': optimizer.convergence_history[-1] if optimizer.convergence_history else float('inf')
        }
        
        print(f"  - 迭代次数: {results[name]['iterations']}")
        print(f"  - 最终误差: {results[name]['final_error']:.6f}")
    
    # 绘制比较图
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        if result['optimizer'].convergence_history:
            plt.semilogy(result['optimizer'].convergence_history, 
                        label=name, linewidth=2)
    
    plt.xlabel('迭代次数')
    plt.ylabel('最大位置变化 (log scale)')
    plt.title('不同通信拓扑的收敛性能比较')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # 运行单个测试
    optimizer, results = test_distributed_optimization()
    
    # 比较不同拓扑（取消注释以运行）
    # topology_results = compare_topologies()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)