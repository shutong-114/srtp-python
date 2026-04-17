import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 忽略计算中的数值溢出警告
warnings.filterwarnings('ignore')

class PenaltyOptimizer:
    """
    PenaltyOptimizer: 基于 script_8 逻辑的罚函数优化器
    - 接口对标: optimal(initial_positions, LF_vertices, b)
    - 返回值: {1: [x1, y1], 2: [x2, y2], ...}
    - 核心逻辑: 队形变换、动量梯度下降、碰撞避免、边界约束
    """

    # ==========================================================
    # 一、初始化 (参数对标 main 文件与 penalty3)
    # ==========================================================
    def __init__(self, shape_icon, sigma=0.01, max_iter=1000, tol=1e-2, eta=0.2, **kwargs):
        # 强制转换为 numpy 数组确保数学运算正确
        self.shape_icon = np.asanyarray(shape_icon, dtype=float)
        self.num_agents = len(self.shape_icon)
        
        # 基础接口变量
        self.sigma = sigma      
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        
        # script_8 特有超参数 (带默认值)
        self.A = kwargs.get('A', 0.9)                  # 碰撞检测范围
        self.d = kwargs.get('d', 0.5)                  # 安全距离
        self.alpha = sigma                             # 对应梯度步长
        self.beta = kwargs.get('beta', 0.9)            # 动量项系数
        self.formation_strength = kwargs.get('formation_strength', 1.0)
        self.barrier_weight = kwargs.get('barrier_weight', 1000.0)
        self.anchor_pair = kwargs.get('anchor_pair', (0, 1))
        self.EPS = kwargs.get('EPS', 1e-8)
        
        # 内部状态
        self.target_formation = None
        self.LF_vertices = None
        self.convergence_history = []

    # ==========================================================
    # 二、内部计算逻辑 (script_8 核心公式)
    # ==========================================================

    def _compute_transformation(self, b):
        """利用锚点向量 b 将 shape_icon 映射到当前目标姿态"""
        i, j = self.anchor_pair
        b_vec = np.asanyarray(b, dtype=float)
        
        ref_vec = self.shape_icon[i] - self.shape_icon[j]
        ref_norm_sq = np.sum(ref_vec**2) + self.EPS
        
        cos_theta = np.dot(ref_vec, b_vec) / ref_norm_sq
        sin_theta = (ref_vec[0] * b_vec[1] - ref_vec[1] * b_vec[0]) / ref_norm_sq
        
        C = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        self.target_formation = self.shape_icon @ C.T

    def _compute_gradient(self, X):
        """核心梯度计算：Formation + Collision + Boundary"""
        N = self.num_agents
        grad = np.zeros_like(X)
        
        # 1. 队形项
        for i in range(N):
            for j in range(N):
                if i != j:
                    actual_rel = X[i] - X[j]
                    target_rel = self.target_formation[i] - self.target_formation[j]
                    grad[i] += self.formation_strength * (actual_rel - target_rel)
        
        # 2. 碰撞项
        for m in range(N):
            for n in range(N):
                if m == n: continue
                diff = X[m] - X[n]
                dist = np.sqrt(np.sum(diff**2) + self.EPS)
                if dist < self.A:
                    term = (dist - self.d) * (dist - self.A)
                    grad_val = (2 * dist - self.A - self.d) * (diff / (dist + self.EPS))
                    grad[m] += (2 / (self.A - self.d)**2) * term * grad_val
        
        # 3. 边界项 (LF 安全区域)
        if self.LF_vertices is not None:
            V = self.LF_vertices
            num_v = len(V)
            for i in range(N):
                for v_idx in range(num_v):
                    p1, p2 = V[v_idx], V[(v_idx + 1) % num_v]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal /= (np.linalg.norm(normal) + self.EPS)
                    
                    dist_to_edge = np.dot(X[i] - p1, normal)
                    if dist_to_edge < 0:
                        grad[i] -= self.barrier_weight * dist_to_edge * normal
                        
        return grad

    # ==========================================================
    # 三、主执行接口 (更名为 optimal，移除 adj_matrix)
    # ==========================================================
    def optimize(self, initial_positions, LF_vertices, b=None):
        """
        执行编队优化
        :param initial_positions: 初始位置 (N, 2)
        :param LF_vertices: 安全区域顶点列表
        :param b: 相对锚点向量 (R1-R2)
        :return: 1-based 索引字典 {1: [x,y], 2: [x,y], ...}
        """
        # 数据转换与鲁棒性处理ize
        self.X = np.asanyarray(initial_positions, dtype=float)
        if LF_vertices is not None:
            self.LF_vertices = np.asanyarray(LF_vertices, dtype=float)
        
        self.convergence_history = []
        
        # 生成目标队形
        if b is not None:
            self._compute_transformation(b)
        else:
            self.target_formation = self.shape_icon
            
        delta_prev = np.zeros_like(self.X)
        
        # 梯度下降迭代
        for k in range(self.max_iter):
            grad = self._compute_gradient(self.X)
            
            # 带动量的更新规则
            delta = -(self.alpha * grad + self.beta * delta_prev)
            self.X += delta
            
            change = np.max(np.linalg.norm(delta, axis=1))
            self.convergence_history.append(change)
            
            if change < self.tol:
                break
            delta_prev = delta
            
        # 返回 1-based 字典
        return {i + 1: self.X[i].copy() for i in range(self.num_agents)}

    # 为了防止旧版调用报错，保留一个别名指向 optimal
    def optimize_distributed(self, initial_positions, LF_vertices, adj_matrix=None, b=None):
        return self.optimal(initial_positions, LF_vertices, b=b)
    
    # ==========================================================
# 测试代码（在文件末尾执行）
# ==========================================================
def test_penalty_optimizer():
    """
    对 PenaltyOptimizer 进行完整的功能测试与可视化。
    测试场景：
        - 4 个智能体，参考队形为边长为2的正方形（原点为中心）
        - 安全区域为 [-3,3]×[-3,3] 的正方形边界
        - 初始位置随机偏移，部分靠近边界以触发碰撞与边界惩罚
        - 锚点向量 b = (-2, 0) 使得目标队形与参考队形完全一致（无旋转缩放）
    可视化内容：
        1. 初始位置、目标队形、优化后位置以及安全区域
        2. 梯度下降的收敛曲线
    """
    print("=" * 60)
    print("PenaltyOptimizer 单元测试与可视化")
    print("=" * 60)

    # ---------- 1. 定义测试数据 ----------
    # 参考队形：正方形，边长2，中心在原点
    shape_icon = np.array([
        [-1., -1.],
        [ 1., -1.],
        [ 1.,  1.],
        [-1.,  1.]
    ])
    num_agents = len(shape_icon)

    # 初始位置：在正方形内部随机扰动，并让一个点故意靠近边界
    np.random.seed(42)  # 可复现
    initial_positions = np.array([[0, 0] for _ in range(num_agents)])
    # 把第三个点移到边界附近（触发边界惩罚）
    #initial_positions[2] = [2.5, 2.5]

    # 安全区域：矩形 [-3,3]×[-3,3]
    LF_vertices = np.array([
        [-3., -3.],
        [ 3., -3.],
        [ 3.,  3.],
        [-3.,  3.]
    ])

    # 锚点向量 b：使参考向量 (agent0 - agent1) 映射到 (-2, 0)，保持队形不变
    b = np.array([-2., 0.])

    # ---------- 2. 创建优化器并执行优化 ----------
    # 使用自定义超参数（可根据需要调整）
    optimizer = PenaltyOptimizer(
        shape_icon=shape_icon,
        sigma=0.02,                # 梯度步长
        max_iter=800,
        tol=1e-3,
        formation_strength=0.8,    # 队形保持权重
        barrier_weight=500.0,      # 边界惩罚权重
        A=1.2,                     # 碰撞检测范围
        d=0.6,                     # 期望安全距离
        beta=0.85,                # 动量系数
        anchor_pair=(0, 1)        # 固定锚点对
    )

    print("\n开始优化...")
    result_dict = optimizer.optimize(
        initial_positions=initial_positions,
        LF_vertices=LF_vertices,
        b=b
    )
    optimized_positions = np.array([result_dict[i+1] for i in range(num_agents)])
    print(f"优化完成，迭代次数: {len(optimizer.convergence_history)}")
    print(f"最终最大位置变化: {optimizer.convergence_history[-1]:.4f}")

    # ---------- 3. 可视化 ----------
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PenaltyOptimizer 测试与可视化", fontsize=14)

    # ---- 左图：位置分布 ----
    ax1.set_aspect('equal')

    # 绘制安全区域（填充浅色边界）
    poly = Polygon(LF_vertices, closed=True, facecolor='lightgray', edgecolor='black', alpha=0.5, label='Safe Area')
    ax1.add_patch(poly)

    # 绘制参考队形（透明，用于形状对比）
    ax1.scatter(shape_icon[:, 0], shape_icon[:, 1], c='blue', marker='s', s=80, alpha=0.3, label='Reference Formation')
    # 用虚线连接参考队形顶点（显示形状）
    closed_icon = np.vstack([shape_icon, shape_icon[0]])
    ax1.plot(closed_icon[:, 0], closed_icon[:, 1], 'b--', alpha=0.3)

    # 绘制目标队形（经过锚点变换后的理想位置）
    target = optimizer.target_formation
    ax1.scatter(target[:, 0], target[:, 1], c='green', marker='^', s=100, label='Target Formation')
    closed_target = np.vstack([target, target[0]])
    ax1.plot(closed_target[:, 0], closed_target[:, 1], 'g--', alpha=0.5)

    # 绘制初始位置
    ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], c='orange', marker='o', s=80, label='Initial Positions')
    # 添加编号
    for i, (x, y) in enumerate(initial_positions):
        ax1.annotate(str(i+1), (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9, color='orange')

    # 绘制优化后位置
    ax1.scatter(optimized_positions[:, 0], optimized_positions[:, 1], c='red', marker='D', s=80, label='Optimized Positions')
    for i, (x, y) in enumerate(optimized_positions):
        ax1.annotate(str(i+1), (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9, color='red')

    # 连接优化后位置形成队形（显示保持形状）
    closed_opt = np.vstack([optimized_positions, optimized_positions[0]])
    ax1.plot(closed_opt[:, 0], closed_opt[:, 1], 'r-', alpha=0.7)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Position Evolution')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ---- 右图：收敛曲线 ----
    history = optimizer.convergence_history
    ax2.plot(range(1, len(history)+1), history, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max Position Change')
    ax2.set_title('Convergence History')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_yscale('log')  # 对数坐标更清晰

    # 标记最终tol线
    ax2.axhline(y=optimizer.tol, color='r', linestyle='--', label=f'tol={optimizer.tol}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # ---------- 4. 简要结果输出 ----------
    print("\n优化后的位置（1-based 索引）:")
    for i in range(num_agents):
        print(f"  Agent {i+1}: [{optimized_positions[i,0]:.3f}, {optimized_positions[i,1]:.3f}]")
    print("=" * 60)


# 修复原类中的一个别名错误（optimize_distributed 错误调用了 self.optimal）
# 直接在类定义后打补丁，避免影响原有逻辑
def _fixed_optimize_distributed(self, initial_positions, LF_vertices, adj_matrix=None, b=None):
    return self.optimize(initial_positions, LF_vertices, b=b)

PenaltyOptimizer.optimize_distributed = _fixed_optimize_distributed

if __name__ == "__main__":
    
    test_penalty_optimizer()
