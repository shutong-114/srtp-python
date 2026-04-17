import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, distance


def objective(X, target_positions, adj_matrix, initial_distances, initial_angles, alpha=1.0, beta=1.0):
    """
    目标函数：最小化
    1. 小车优化后位置与目标点的距离
    2. 小车之间距离变化与初始距离的差值
    3. 小车角度变化与初始角度的差值
    """
    X = X.reshape(-1, 2)
    num_cars = X.shape[0]
    
    # 第一部分：距离目标点的距离之和
    f1 = np.sum([np.linalg.norm(X[i] - target_positions[i]) for i in range(num_cars)])
    #f1 = np.linalg.norm(np.mean([X[i] for i in range(num_cars)], axis=0) - np.mean([target_positions[i] for i in range(num_cars)], axis=0) )
    # 第二部分：维护小车之间的距离
    f2 = 0
    for i in range(num_cars):
        for j in range(i + 1, num_cars):
            
            if adj_matrix[i, j] == 1:
                current_distance = np.linalg.norm(X[i] - X[j])
                f2 += np.abs(current_distance - initial_distances[i, j])
                '''
            current_distance = np.linalg.norm(X[i] - X[j])
            f2 += np.abs(current_distance - initial_distances[i, j])
            '''
    
    # 第三部分：维护角度变化
    f3 = 0
    
    for i in range(num_cars):
        current_angle_sum = 0
        for m in range(num_cars):
            for k in range(m + 1, num_cars):
                if adj_matrix[i, m] == 1 and adj_matrix[i, k] == 1 and m != i and k != i:
                    # 计算向量
                    qim = X[m] - X[i]  # 向量 q^i - q^m
                    qik = X[k] - X[i]  # 向量 q^i - q^k
                    
                    # 计算夹角（arccos）
                    dot_product = np.dot(qim, qik)
                    norm_qim = np.linalg.norm(qim)
                    norm_qik = np.linalg.norm(qik)
                    
                    # 避免除零错误
                    if norm_qim * norm_qik != 0:
                        cos_theta = np.clip(dot_product / (norm_qim * norm_qik), -1.0, 1.0)
                        theta_imk = np.arccos(cos_theta)
                        current_angle_sum += theta_imk
        
        # 计算角度差值
        f3 += np.abs(current_angle_sum - initial_angles[i])
    
    # 总目标函数
    return f1 + alpha * f2 + beta * f3


def constraint_all_cars_in_LF(X, LF_vertices):
    """
    约束：确保所有小车都在自由区域内或边界上
    """
    X = X.reshape(-1, 2)
    hull = ConvexHull(LF_vertices)

    # 计算每个小车是否在自由区域内
    constraints = []
    for i in range(X.shape[0]):
        point = X[i]
        max_val = -np.inf
        for eq in hull.equations:
            a, b, c = eq
            val = a * point[0] + b * point[1] + c
            if val > max_val:
                max_val = val
        constraints.append(-max_val)  # max_val <= 0 表示点在区域内
    
    return np.array(constraints)

def constraint_min_distance_all(X, threshold):
    """
    约束：保证所有小车间的距离均大于阈值 threshold，
    返回每对小车间的 (distance - threshold)，要求均 >= 0
    """
    X = X.reshape(-1, 2)
    constraints = []
    num_cars = X.shape[0]
    for i in range(num_cars):
        for j in range(i + 1, num_cars):
            constraints.append(np.linalg.norm(X[i] - X[j]) - threshold)  
            # 约束表达式：确保距离大于 threshold，即  np.linalg.norm(X[i] - X[j]) >= threshold
    return np.array(constraints)


def compute_initial_distances(initial_positions):
    """
    计算小车之间的初始距离
    """
    num_cars = len(initial_positions)
    initial_distances = np.zeros((num_cars, num_cars))
    for i in range(num_cars):
        for j in range(i + 1, num_cars):
            initial_distances[i, j] = np.linalg.norm(initial_positions[i] - initial_positions[j])
            initial_distances[j, i] = initial_distances[i, j]
    return initial_distances


def compute_initial_angles(initial_positions, adj_matrix):
    """
    计算初始角度：\omega^i
    """
    num_cars = len(initial_positions)
    initial_angles = np.zeros(num_cars)
    
    for i in range(num_cars):
        initial_angle_sum = 0
        for m in range(num_cars):
            for k in range(m + 1, num_cars):
                if adj_matrix[i, m] == 1 and adj_matrix[i, k] == 1 and m != i and k != i:
                    qim = initial_positions[m] - initial_positions[i]
                    qik = initial_positions[k] - initial_positions[i]
                    
                    dot_product = np.dot(qim, qik)
                    norm_qim = np.linalg.norm(qim)
                    norm_qik = np.linalg.norm(qik)
                    
                    # 避免除零错误
                    if norm_qim * norm_qik != 0:
                        cos_theta = np.clip(dot_product / (norm_qim * norm_qik), -1.0, 1.0)
                        theta_imk = np.arccos(cos_theta)
                        initial_angle_sum += theta_imk
        
        initial_angles[i] = initial_angle_sum
    
    return initial_angles


def calculate_target_positions(center_target, initial_positions):
    """
    根据目标中心位置计算每个小车的目标点
    """
    center_initial = np.mean(initial_positions, axis=0)
    target_positions = initial_positions + (center_target - center_initial)
    return target_positions


def solve_optimization(LF_vertices, target_positions, initial_positions, adj_matrix, min_distance_threshold=0.5, alpha=1.0, beta=1.0):
    """
    解决优化问题，找到最优路径点
    """
    X0 = initial_positions.flatten()
    initial_distances = compute_initial_distances(initial_positions)
    initial_angles = compute_initial_angles(initial_positions, adj_matrix)
    
    # 约束：确保所有小车都在自由区域内
    constraints = [
        {'type': 'ineq', 'fun': constraint_all_cars_in_LF, 'args': (LF_vertices,)},
        {'type': 'ineq', 'fun': constraint_min_distance_all, 'args': (min_distance_threshold,)}    # 添加小车间最小距离约束
    ]


    
    options = {'maxiter': 2000, 'ftol': 1e-4, 'disp': True}
    
    result = minimize(
        objective,
        X0,
        args=(target_positions, adj_matrix, initial_distances, initial_angles, alpha, beta),
        constraints=constraints,
        method='SLSQP',
        options=options
    )
    
    if result.success:
        return result.x.reshape(-1, 2)
    else:
        print("优化失败:", result.message)
        return None


def visualize(LF_vertices, initial_positions, optimal_positions, target_position):
    """
    可视化小车的初始位置、最终位置以及目标点
    """
    plt.figure(figsize=(8, 8))
    
    # 绘制自由区域
    hull = ConvexHull(LF_vertices)
    for simplex in hull.simplices:
        plt.plot(LF_vertices[simplex, 0], LF_vertices[simplex, 1], 'k-')
    
    initial_positions = np.array(initial_positions)
    plt.scatter(initial_positions[:, 0], initial_positions[:, 1], color='blue', label='Initial Positions')

    if optimal_positions is not None:
        plt.scatter(optimal_positions[:, 0], optimal_positions[:, 1], color='green', label='Optimal Positions')
        for i in range(4):
            plt.plot(
                [initial_positions[i, 0], optimal_positions[i, 0]],
                [initial_positions[i, 1], optimal_positions[i, 1]],
                'g--'
            )

    plt.scatter(target_position[0], target_position[1], color='red', marker='x', s=100, label='Target')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.title('Vehicle Formation Optimization')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # ============================
    # 🚀 主程序：开始优化与可视化
    # ============================

    # 设置自由区域的边界顶点
    LF_vertices = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])

    # 目标中心位置（自由区域外）
    center_target = np.array([14, 14])

    # 初始小车位置（形成 1x1 的方形）
    initial_positions = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    # 计算初始距离矩阵
    #initial_distances = compute_initial_distances(initial_positions)

    # 设置邻接矩阵（完全图）
    adj_matrix = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

    # 计算初始角度矩阵
    #initial_angles = compute_initial_angles(initial_positions, adj_matrix)

    # 计算每个小车的目标位置
    target_positions = calculate_target_positions(center_target, initial_positions)

    # 运行优化
    optimal_positions = solve_optimization(
        LF_vertices,
        target_positions,
        initial_positions,
        adj_matrix,
        #initial_distances,
        #initial_angles,
        alpha=5,
        beta=10
    )

    # 可视化结果
    visualize(
        LF_vertices,
        initial_positions,
        optimal_positions,
        center_target
    )
