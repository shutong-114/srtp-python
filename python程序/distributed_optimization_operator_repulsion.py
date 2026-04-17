"""
distributed_optimization_operator_repulsion.py

分布式编队优化器——内嵌斥力版本 (In-loop Repulsion)

原理：
  每次 Jacobi 同步迭代完成写回后，立即对 self.positions 施加一轮对称斥力修正，
  使所有机器人真实位置之间的最小距离 >= min_dist，随后将被推移的点重投影回 LF
  可行区域，确保硬约束（可行域 + 最小间距）在整个优化过程中同步满足。

与后处理版本（srtp_main7.1.py 中 enforce_min_distance）的区别：
  - 后处理版：优化器先不感知间距约束，收敛后一次性外部斥力修正；
    修正结果不再经过优化器验证，可能偏离可行域。
  - 内嵌版（本文件）：斥力成为每次迭代的一部分，优化梯度可在后续迭代中
    感知并适应斥力带来的位移；修正后立即重投影到可行域，约束始终满足。
"""
import copy as _copy

import numpy as np

from distributed_optimization_operator import DistributedFormationOptimizer


class DistributedFormationOptimizerWithRepulsion(DistributedFormationOptimizer):
    """
    继承自 DistributedFormationOptimizer，在每次优化迭代写回后嵌入斥力修正。

    新增参数
    --------
    min_dist : float
        机器人之间的最小允许距离，默认 0.5。
    repulsion_inner_iters : int
        每次调用斥力修正时的最大内层迭代次数，默认 20。
    repulsion_every : int
        每隔多少次外层优化迭代执行一次斥力修正，默认 1（每次都执行）。
    min_iter : int
        最小迭代次数（避免过快收敛），默认 50。
    formation_tol : float
        队形误差收敛阈值，默认 1e-3。
    consensus_tol : float
        一致性误差收敛阈值，默认 1e-3。
    stable_steps : int
        连续满足所有收敛条件的步数，默认 5。
    """

    def __init__(
        self,
        shape_icon,
        sigma=0.1,
        max_iter=1000,
        tol=1e-2,
        eta=0.2,
        min_dist=0.5,
        repulsion_inner_iters=20,
        repulsion_every=1,
        min_iter=50,
        formation_tol=1e-3,
        consensus_tol=1e-3,
        stable_steps=5,
    ):
        super().__init__(shape_icon, sigma, max_iter, tol, eta)
        self.min_dist = min_dist
        self.repulsion_inner_iters = repulsion_inner_iters
        self.repulsion_every = repulsion_every
        self.min_iter = min_iter
        self.formation_tol = formation_tol
        self.consensus_tol = consensus_tol
        self.stable_steps = stable_steps
        self.formation_error_history = []
        self.consensus_error_history = []

    # ------------------------------------------------------------------
    # 核心：斥力修正 + LF 重投影
    # ------------------------------------------------------------------

    def _apply_repulsion_and_project(self, LF_vertices):
        """
        对 self.positions 执行对称斥力修正，然后将每个被移动的点重投影到
        LF 可行区域。

        斥力规则（与 enforce_min_distance 相同的对称推移逻辑）：
          correction = (min_dist - dist) / 2 * (diff / dist)
          positions[i] += correction
          positions[j] -= correction

        重投影确保斥力推移不会把点推出可行域。
        """
        ids = list(range(1, self.m + 1))
        moved_ids = set()

        for _ in range(self.repulsion_inner_iters):
            any_moved = False
            for a in range(len(ids)):
                for b_idx in range(a + 1, len(ids)):
                    i, j = ids[a], ids[b_idx]
                    diff = self.positions[i] - self.positions[j]
                    dist = np.linalg.norm(diff)
                    if dist < self.min_dist:
                        if dist < 1e-9:
                            diff = np.random.randn(2)
                            diff /= np.linalg.norm(diff)
                        else:
                            diff /= dist
                        correction = (self.min_dist - dist) / 2.0 * diff
                        self.positions[i] += correction
                        self.positions[j] -= correction
                        moved_ids.add(i)
                        moved_ids.add(j)
                        any_moved = True
            if not any_moved:
                break

        # 将被斥力推移的点重投影回 LF 可行域
        for i in moved_ids:
            self.positions[i] = self._project_to_region(
                self.positions[i], LF_vertices
            )

    # ------------------------------------------------------------------
    # 重写 optimize_distributed：在 Jacobi 写回后嵌入斥力
    # ------------------------------------------------------------------

    def optimize_distributed(self, initial_positions, LF_vertices, adjacency_matrix, b):
        """
        Jacobi 同步迭代 + 内嵌斥力版本。

        每次迭代流程：
          1. Snapshot（深拷贝当前 positions / estimates / dual_vars / L）
          2. 用 snapshot 并行计算所有机器人的更新量（不立即写回）
          3. 一次性写回所有更新量
          4. 记录一次纯优化的位移量 max_change（用于收敛判断）
          5. （每 repulsion_every 次）对 self.positions 施加斥力修正并重投影
          6. 收敛检测：在满足最小迭代次数的前提下，同时满足
             max_change / formation_error / consensus_error 阈值并连续稳定 stable_steps 次

        注意：斥力只修改 self.positions（真实位置），self.estimates 通过后续
        迭代的一致性项自动向正确值收敛，无需强制同步。
        """
        self.set_communication_topology(adjacency_matrix)
        self.set_anchor_constraint(b)
        self._initialize_positions_and_estimates(initial_positions)
        B_matrices = self._build_distributed_constraints()
        self.convergence_history = []

        stable_count = 0
        for k in range(self.max_iter):
            # 1. Snapshot
            positions_snap = {i: self.positions[i].copy() for i in self.positions}
            estimates_snap = _copy.deepcopy(self.estimates)
            dual_snap = _copy.deepcopy(self.dual_vars)
            L_snap = self.L.copy()

            # 2. 计算更新量（基于 snapshot，不写回）
            updates_new = {}
            updates_new[1] = self._compute_update_robot1_from_snap(
                initial_positions, LF_vertices, k,
                positions_snap, estimates_snap, dual_snap, L_snap,
            )
            updates_new[2] = self._compute_update_robot2_from_snap(
                initial_positions, LF_vertices, k,
                positions_snap, estimates_snap, dual_snap, L_snap,
            )
            for i in range(3, self.m + 1):
                updates_new[i] = self._compute_update_robot_i_from_snap(
                    i, initial_positions, LF_vertices, k,
                    positions_snap, estimates_snap, dual_snap, L_snap, B_matrices,
                )

            # 3. 一次性写回
            for i, upd in updates_new.items():
                if i == 1:
                    self.positions[1] = upd['p1']
                    self.estimates[1]['p12'] = upd['p12']
                    self.dual_vars[1]['y1'] = upd['y1']
                    self.dual_vars[1]['z1'] = upd['z1']
                    self.dual_vars[1]['w12'] = upd['w12']
                elif i == 2:
                    self.positions[2] = upd['p2']
                    self.estimates[2]['p21'] = upd['p21']
                    self.dual_vars[2]['y2'] = upd['y2']
                    self.dual_vars[2]['z21'] = upd['z21']
                    self.dual_vars[2]['w2'] = upd['w2']
                else:
                    self.positions[i] = upd['p_i']
                    self.estimates[i]['p_i1'] = upd['p_i1']
                    self.estimates[i]['p_i2'] = upd['p_i2']
                    self.dual_vars[i]['y_i'] = upd['y_i']
                    self.dual_vars[i]['z_i1'] = upd['z_i1']
                    self.dual_vars[i]['w_i2'] = upd['w_i2']

            # 4. 记录纯优化位移量（未施加斥力）
            max_change = 0.0
            for i in range(1, self.m + 1):
                change = np.linalg.norm(self.positions[i] - positions_snap[i])
                if change > max_change:
                    max_change = change
            self.convergence_history.append(max_change)

            # 5. 内嵌斥力修正 + LF 重投影
            if (k + 1) % self.repulsion_every == 0:
                self._apply_repulsion_and_project(LF_vertices)

            # 6. 收敛检测（多指标 + 最小迭代次数 + 稳定步数）
            formation_error = self.get_formation_error()
            consensus_error = self.get_consensus_error()
            self.formation_error_history.append(formation_error)
            self.consensus_error_history.append(consensus_error)

            meets_criteria = (
                (k + 1) >= self.min_iter
                and max_change < self.tol
                and formation_error < self.formation_tol
                and consensus_error < self.consensus_tol
            )
            if meets_criteria:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= self.stable_steps:
                print(
                    f"[RepulsionOptimizer] 在第 {k} 次迭代收敛 "
                    f"(max_change={max_change:.3e}, "
                    f"formation_error={formation_error:.3e}, "
                    f"consensus_error={consensus_error:.3e})"
                )
                break

        return self.positions
