import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, distance, Delaunay

def objective(center, target_position):
    """目标函数：最小化几何中心到目标点的距离"""
    return np.linalg.norm(center - target_position)

def constraint_center_in_LF(center, LF_vertices):
    """约束：确保几何中心在自由区域内或边界上"""
    hull = ConvexHull(LF_vertices)
    max_val = -np.inf
    for eq in hull.equations:
        a, b, c = eq
        val = a * center[0] + b * center[1] + c
        if val > max_val:
            max_val = val
    return -max_val  # 确保 max_val <= 0

def solve_optimization(LF_vertices, target_position, initial_center):
    constraints = [
        {'type': 'ineq', 'fun': constraint_center_in_LF, 'args': (LF_vertices,)}
    ]
    options = {'maxiter': 1000, 'ftol': 1e-8, 'disp': True}
    result = minimize(objective, initial_center, args=(target_position,), constraints=constraints, method='SLSQP', options=options)
    return result.x if result.success else None

def visualize(LF_vertices, initial_center, optimal_center, target_position):
    plt.figure(figsize=(8, 8))
    hull = ConvexHull(LF_vertices)
    for simplex in hull.simplices:
        plt.plot(LF_vertices[simplex, 0], LF_vertices[simplex, 1], 'k-')
    plt.scatter(initial_center[0], initial_center[1], color='blue', label='Initial Center')
    if optimal_center is not None:
        plt.scatter(optimal_center[0], optimal_center[1], color='green', label='Optimal Center')
        plt.plot([initial_center[0], optimal_center[0]], [initial_center[1], optimal_center[1]], 'g--')
    plt.scatter(target_position[0], target_position[1], color='red', marker='x', s=100, label='Target')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.title('Center Optimization')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 示例输入
    LF_vertices = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
    target_position = np.array([15, 50])
    initial_center = np.array([5, 5])

    # 运行优化与可视化
    optimal_center = solve_optimization(LF_vertices, target_position, initial_center)
    visualize(LF_vertices, initial_center, optimal_center, target_position)
