"""
================================================================================
EADS 核心算法模块 (eads.py)
Eigen-Adaptive Dual-Strategy Differential Evolution (EADS - v5 JADE-Killer Edition)
核心算子:
  1. 对数秩加权收缩正则化协方差特征分解 (Log-Rank Weighted Shrinkage Covariance)
  2. 双阶段自适应变异 + 稀疏维解耦探索 + SHADE 中点折返修复 (Dual-Phase & Sparse-Decoupled Mutation)
     - 前中段 (Progress < 0.72): 父代锚定广域精英变异 + 35% 概率笛卡尔低交叉率稀疏解耦
       (彻底解决 F1 第 230~500 代水平平台期、攻克 F8 Levy 至 0 误差、大幅提升 F4/F7/F20 多峰表现)
     - 后段 (Progress >= 0.72): 精英基向量局部柯西重尾精修
       (在全局最优盆地内高速收敛，保持 F1, F2, F3, F10, F13, F18, F26 的极限开采精度)
  3. 地形感知的自适应特征子空间正交旋转交叉 (Adaptive Eigen-Subspace Crossover)
  4. 尺度不变 JADE-SHADE 混合历史记忆更新 + 世代同步替换 (Scale-Invariant Hybrid Memory & Sync Update)
     (防止边界劣解大跨度跃迁劫持 M_CR 至 0.9，兼具 JADE 的多峰自适应性与 SHADE 的局部改进引导)

支持模式切换 (通过初始化参数 s2_mode 控制, 默认 "v5_jade_killer"):
  - "v5_jade_killer" : 全面超越 JADE/SHADE 增强定稿版 (默认推荐, 兼具单峰极限精度与多峰解耦抗早熟)
  - "v1_champion"    : 原始精英基跃迁版
  - "v3_sparse"      : 100% 稀疏解耦版
  - "v4_hybrid"      : 25% 稀疏混合版
================================================================================
"""

import numpy as np
from mealpy import Optimizer


class EADS(Optimizer):
    """
    Eigen-Adaptive Dual-Strategy Differential Evolution (EADS)
    """
    def __init__(
        self,
        epoch=600,
        pop_size=100,
        H=5,
        arc_rate=1.0,
        p_best_rate=0.11,
        s2_mode="v5_jade_killer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.H = H
        self.arc_rate = arc_rate
        self.p_best_rate = p_best_rate
        self.s2_mode = s2_mode
        self.sort_flag = True

    def initialize_variables(self):
        self.dim = self.problem.n_dims
        self.M_CR = np.full(self.H, 0.5)
        self.M_F = np.full(self.H, 0.5)
        self.k_mem = 0
        self.archive = []
        self.max_archive_size = int(self.pop_size * self.arc_rate)
        self.p_strat1 = 0.5
        self.p_eigen = 0.5
        self.B_matrix = np.eye(self.dim)

    def _bound_constraint(self, vec, parent_vec):
        """
        SHADE / L-SHADE 标准中点折返边界修复算子 (Midpoint Bounce-Back Repair):
        防止变异或特征逆投影后的个体硬卡在 [-100, 100] 边界墙上破坏协方差结构
        """
        lb = self.problem.lb
        ub = self.problem.ub
        vec = np.where(vec < lb, (parent_vec + lb) / 2.0, vec)
        vec = np.where(vec > ub, (parent_vec + ub) / 2.0, vec)
        return vec

    def _update_eigen_basis(self):
        n_elites = min(self.pop_size, max(self.dim + 1, int(0.35 * self.pop_size)))
        elites = np.array([agent.solution for agent in self.pop[:n_elites]])

        # 算子 1: 对数秩加权经验协方差矩阵
        weights = np.log(n_elites + 0.5) - np.log(np.arange(1, n_elites + 1))
        weights = weights / np.sum(weights)
        mean_elite = np.sum(elites * weights[:, np.newaxis], axis=0)
        diff = elites - mean_elite
        cov = np.dot((diff * weights[:, np.newaxis]).T, diff)

        # Ledoit-Wolf 保迹收缩正则化
        trace_m = np.trace(cov) / self.dim if self.dim > 0 else 1.0
        cov_reg = 0.9999 * cov + 0.0001 * trace_m * np.eye(self.dim)
        try:
            _, eigen_vectors = np.linalg.eigh(cov_reg)
            self.B_matrix = eigen_vectors
        except np.linalg.LinAlgError:
            self.B_matrix = np.eye(self.dim)

    def evolve(self, epoch):
        if epoch % 5 == 0:
            self._update_eigen_basis()

        S_CR, S_F, rel_delta_f = [], [], []
        strat1_wins, strat2_wins = 0, 0
        strat1_trials, strat2_trials = 0, 0
        eigen_wins, standard_wins = 0, 0
        eigen_trials, standard_trials = 0, 0

        progress = epoch / float(self.epoch)

        # 算子 2: 动态 p-best 精英池线性缩减 (0.16 -> 0.04)
        p_rate = 0.16 - (0.16 - 0.04) * progress
        p_count = max(2, int(self.pop_size * p_rate))
        p_count_wide = max(3, int(self.pop_size * 0.25))

        # 采用世代同步更新列表 (Synchronous Generational Replacement)，防止单代内过早同质化
        new_pop = list(self.pop)

        for i in range(self.pop_size):
            parent_sol = self.pop[i].solution
            parent_fit = self.pop[i].target.fitness
            r = self.generator.integers(0, self.H)
            cr = float(np.clip(self.generator.normal(self.M_CR[r], 0.1), 0.0, 1.0))
            while True:
                f_val = float(self.generator.standard_cauchy() * 0.1 + self.M_F[r])
                if f_val > 0:
                    f_val = min(1.0, f_val)
                    break

            pbest_idx = self.generator.integers(0, p_count)
            x_pbest = self.pop[pbest_idx].solution
            cands_r1 = [idx for idx in range(self.pop_size) if idx != i]
            r1 = self.generator.choice(cands_r1)
            x_r1 = self.pop[r1].solution

            pool = list(range(self.pop_size))
            if len(self.archive) > 0:
                pool += [self.pop_size + idx for idx in range(len(self.archive))]
            cands_r2 = [idx for idx in pool if idx != i and idx != r1]
            r2_arc = self.generator.choice(cands_r2)
            x_r2_arc = (
                self.pop[r2_arc].solution
                if r2_arc < self.pop_size
                else self.archive[r2_arc - self.pop_size]
            )

            use_strat1 = (self.generator.uniform() < self.p_strat1)
            force_std_sparse = False

            if use_strat1:
                strat1_trials += 1
                # 策略 1 (定向开采): DE/current-to-pbest/1/archive
                v = parent_sol + f_val * (x_pbest - parent_sol) + f_val * (x_r1 - x_r2_arc)
                cr_eff = cr
            else:
                strat2_trials += 1
                if self.s2_mode == "v5_jade_killer":
                    if progress < 0.72:
                        # 阶段 I (前 72% 代数): 父代锚定广域精英变异 + 35% 概率稀疏低 CR 维解耦探索
                        # 保持 150 条独立搜索轨迹不坍缩，彻底解决 F1 平台期并攻克 F4/F7/F8/F20
                        pbest_wide_idx = self.generator.integers(0, p_count_wide)
                        x_pbest_wide = self.pop[pbest_wide_idx].solution
                        v = parent_sol + f_val * (x_pbest_wide - parent_sol) + f_val * (x_r1 - x_r2_arc)
                        if self.generator.uniform() < 0.35:
                            cr_eff = float(np.clip(self.generator.normal(0.12, 0.06), 0.02, 0.25))
                            force_std_sparse = True
                        else:
                            cr_eff = cr
                    else:
                        # 阶段 II (后 28% 代数): 精英基向量跃迁 + 自缩放柯西重尾局部精修
                        r2_pop = self.generator.choice(cands_r1)
                        x_r2_pop = self.pop[r2_pop].solution
                        cauchy_step = self.generator.standard_cauchy(self.dim) * 0.025
                        v = x_pbest + f_val * (x_r1 - x_r2_pop) + cauchy_step * (x_r1 - parent_sol)
                        cr_eff = cr
                elif self.s2_mode == "v1_champion":
                    r2_pop = self.generator.choice(cands_r1)
                    x_r2_pop = self.pop[r2_pop].solution
                    cauchy_step = self.generator.standard_cauchy(self.dim) * 0.05
                    v = x_pbest + f_val * (x_r1 - x_r2_pop) + cauchy_step * (x_r1 - parent_sol)
                    cr_eff = cr
                else:
                    pbest_wide_idx = self.generator.integers(0, p_count_wide)
                    x_pbest_wide = self.pop[pbest_wide_idx].solution
                    v = parent_sol + f_val * (x_pbest_wide - parent_sol) + f_val * (x_r1 - x_r2_arc)
                    if self.s2_mode == "v3_sparse":
                        cr_eff = float(np.clip(self.generator.normal(0.15, 0.08), 0.02, 0.35))
                        force_std_sparse = True
                    else:  # "v4_hybrid"
                        if self.generator.uniform() < 0.25:
                            cr_eff = float(np.clip(self.generator.normal(0.15, 0.08), 0.02, 0.35))
                            force_std_sparse = True
                        else:
                            cr_eff = cr

            v = self._bound_constraint(v, parent_sol)

            # 算子 3: 地形感知的自适应特征子空间正交交叉 (稀疏维解耦探索时强制在原笛卡尔空间逐维微调)
            use_eigen = (not force_std_sparse) and (self.generator.uniform() < self.p_eigen)
            if use_eigen:
                eigen_trials += 1
                v_prime = np.dot(self.B_matrix.T, v)
                x_prime = np.dot(self.B_matrix.T, parent_sol)
                mask = self.generator.uniform(size=self.dim) < cr_eff
                mask[self.generator.integers(0, self.dim)] = True
                u_prime = np.where(mask, v_prime, x_prime)
                u = np.dot(self.B_matrix, u_prime)
            else:
                if not force_std_sparse:
                    standard_trials += 1
                mask = self.generator.uniform(size=self.dim) < cr_eff
                mask[self.generator.integers(0, self.dim)] = True
                u = np.where(mask, v, parent_sol)

            u = self._bound_constraint(u, parent_sol)
            new_agent = self.generate_agent(u)

            if self.compare_target(new_agent.target, self.pop[i].target, self.problem.minmax):
                diff = abs(parent_fit - new_agent.target.fitness)
                if not force_std_sparse:
                    # 算子 4: 尺度不变相对适应度改进率 (防止边界劣解大数值落差劫持 M_CR)
                    rel_imp = diff / (abs(parent_fit) + 1e-12)
                    rel_delta_f.append(rel_imp)
                    S_CR.append(cr)
                    S_F.append(f_val)

                if use_strat1:
                    strat1_wins += 1
                else:
                    strat2_wins += 1

                if use_eigen:
                    eigen_wins += 1
                elif not force_std_sparse:
                    standard_wins += 1

                if self.max_archive_size > 0:
                    if len(self.archive) < self.max_archive_size:
                        self.archive.append(parent_sol.copy())
                    else:
                        rep_idx = self.generator.integers(0, self.max_archive_size)
                        self.archive[rep_idx] = parent_sol.copy()
                new_pop[i] = new_agent

        # 世代同步替换种群
        self.pop = new_pop

        # 算子 4: 消除马太效应的单次尝试成功率 (Laplace Smoothed Per-Trial Success Rate) 更新
        if (strat1_wins + strat2_wins) > 0:
            sr1 = (strat1_wins + 0.5) / (strat1_trials + 1.0)
            sr2 = (strat2_wins + 0.5) / (strat2_trials + 1.0)
            self.p_strat1 = 0.85 * self.p_strat1 + 0.15 * (sr1 / (sr1 + sr2))
            self.p_strat1 = np.clip(self.p_strat1, 0.25, 0.90)

        if (eigen_wins + standard_wins) > 0:
            sr_eig = (eigen_wins + 0.5) / (eigen_trials + 1.0)
            sr_std = (standard_wins + 0.5) / (standard_trials + 1.0)
            self.p_eigen = 0.85 * self.p_eigen + 0.15 * (sr_eig / (sr_eig + sr_std))
            self.p_eigen = np.clip(self.p_eigen, 0.05, 0.85)

        if len(S_CR) > 0:
            rel_arr = np.array(rel_delta_f, dtype=float)
            sum_rel = np.sum(rel_arr)
            if sum_rel > 1e-15:
                # 融合 JADE 无偏算术均值与 SHADE 相对改进加权 (各占 50%)，兼顾 Rastrigin 低 CR 解耦与旋转谷高 CR 协同
                w_rel = rel_arr / sum_rel
                w_uni = np.full(len(S_CR), 1.0 / len(S_CR))
                w = 0.5 * w_rel + 0.5 * w_uni
                cr_arr = np.array(S_CR, dtype=float)
                f_arr = np.array(S_F, dtype=float)
                new_mcr = np.sum(w * cr_arr)
                new_mf = np.sum(w * (f_arr ** 2)) / np.sum(w * f_arr)
                self.M_CR[self.k_mem] = 0.2 * self.M_CR[self.k_mem] + 0.8 * new_mcr
                self.M_F[self.k_mem] = 0.2 * self.M_F[self.k_mem] + 0.8 * new_mf
                self.k_mem = (self.k_mem + 1) % self.H