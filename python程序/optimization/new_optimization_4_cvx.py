import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def solve_optimization(LF_vertices, target_positions, initial_positions, adj_matrix, min_distance_threshold=0.5, alpha=1.0, beta = 0):
    num_cars = len(initial_positions)
    X = cp.Variable((num_cars, 2))
    
    # 第一部分：车辆到目标点的距离（采用平方和）
    # 注意这里用的是 cp.sum_squares，目标变为最小化平方误差，尺度会有所不同
    f1 = cp.sum_squares(X - target_positions)
    
    # 第二部分：车辆间距离保持项，计算平方的差异，即 |‖X[i]-X[j]‖² - (d₀)²|
    penalty_terms = []
    for i in range(num_cars):
        for j in range(i + 1, num_cars):
            if adj_matrix[i, j] == 1:
                d0 = np.linalg.norm(initial_positions[i] - initial_positions[j])
                # 当车辆间的平方距离超过初始值时才施加惩罚
                penalty_terms.append(cp.pos(cp.sum_squares(X[i] - X[j]) - d0**2))
    f2 = cp.sum(penalty_terms)

    objective = cp.Minimize(f1 + alpha * f2)
    
    # 约束部分：确保每个车辆在自由区域内（利用凸包边界约束）
    constraints = []
    hull = ConvexHull(LF_vertices)
    for i in range(num_cars):
        for eq in hull.equations:
            a, b, c = eq
            constraints.append(a * X[i, 0] + b * X[i, 1] + c <= 0)
    
    # 约束：车辆间最小距离限制
    for i in range(num_cars):
        for j in range(i + 1, num_cars):
            constraints.append(cp.norm(X[i] - X[j]) >= min_distance_threshold)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return X.value

def visualize(LF_vertices, initial_positions, optimal_positions, target_positions):
    plt.figure(figsize=(8, 8))
    hull = ConvexHull(LF_vertices)
    for simplex in hull.simplices:
        plt.plot(LF_vertices[simplex, 0], LF_vertices[simplex, 1], 'k-')
    
    plt.scatter(initial_positions[:, 0], initial_positions[:, 1], color='blue', label='Initial Positions')
    plt.scatter(target_positions[:, 0], target_positions[:, 1], color='red', marker='x', s=100, label='Target Positions')
    if optimal_positions is not None:
        plt.scatter(optimal_positions[:, 0], optimal_positions[:, 1], color='green', label='Optimal Positions')
        for i in range(len(initial_positions)):
            plt.plot([initial_positions[i, 0], optimal_positions[i, 0]],
                     [initial_positions[i, 1], optimal_positions[i, 1]], 'g--')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.title('Vehicle Formation Optimization')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    LF_vertices = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
    center_target = np.array([14, 14])
    initial_positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    adj_matrix = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]])
    target_positions = initial_positions + (center_target - np.mean(initial_positions, axis=0))
    optimal_positions = solve_optimization(LF_vertices, target_positions, initial_positions, adj_matrix, alpha=5)
    visualize(LF_vertices, initial_positions, optimal_positions, target_positions)
