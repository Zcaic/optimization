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
        s2_mode="v1_champion",
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

        S_CR, S_F, delta_f = [], [], []
        strat1_wins, strat2_wins = 0, 0
        strat1_trials, strat2_trials = 0, 0
        eigen_wins, standard_wins = 0, 0
        eigen_trials, standard_trials = 0, 0

        # 算子 2: 动态 p-best 精英池线性缩减 (0.20 -> 0.04)
        p_rate = 0.20 - (0.20 - 0.04) * (epoch / self.epoch)
        p_count = max(2, int(self.pop_size * p_rate))
        p_count_wide = max(3, int(self.pop_size * 0.25))

        for i in range(self.pop_size):
            parent_sol = self.pop[i].solution
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

            use_strat1 = (self.generator.uniform() < self.p_strat1)
            is_sparse = False
            if use_strat1:
                strat1_trials += 1
                # 策略 1 (开采): current-to-pbest/1 with archive
                pool = list(range(self.pop_size))
                if len(self.archive) > 0:
                    pool += [self.pop_size + idx for idx in range(len(self.archive))]
                cands_r2 = [idx for idx in pool if idx != i and idx != r1]
                r2 = self.generator.choice(cands_r2)
                x_r2 = self.pop[r2].solution if r2 < self.pop_size else self.archive[r2 - self.pop_size]
                v = parent_sol + f_val * (x_pbest - parent_sol) + f_val * (x_r1 - x_r2)
                cr_eff = cr
            else:
                if self.s2_mode == "v1_champion":
                    strat2_trials += 1
                    # 论文定稿冠军版策略 2 (勘探): 精英基向量跃迁 + 自缩放柯西重尾扰动 (50D Friedman = 1.50)
                    r2 = self.generator.choice(cands_r1)
                    x_r2 = self.pop[r2].solution
                    cauchy_step = self.generator.standard_cauchy(self.dim) * 0.05
                    v = x_pbest + f_val * (x_r1 - x_r2) + cauchy_step * (x_r1 - parent_sol)
                    cr_eff = cr
                else:
                    pool = list(range(self.pop_size))
                    if len(self.archive) > 0:
                        pool += [self.pop_size + idx for idx in range(len(self.archive))]
                    cands_r2 = [idx for idx in pool if idx != i and idx != r1]
                    r2_arc = self.generator.choice(cands_r2)
                    x_r2_arc = self.pop[r2_arc].solution if r2_arc < self.pop_size else self.archive[r2_arc - self.pop_size]
                    pbest_wide_idx = self.generator.integers(0, p_count_wide)
                    x_pbest_wide = self.pop[pbest_wide_idx].solution
                    v = parent_sol + f_val * (x_pbest_wide - parent_sol) + f_val * (x_r1 - x_r2_arc)
                    if self.s2_mode == "v3_sparse":
                        strat2_trials += 1
                        cr_eff = float(np.clip(self.generator.normal(0.15, 0.08), 0.02, 0.35))
                    else:  # "v4_hybrid"
                        if self.generator.uniform() < 0.25:
                            cr_eff = float(np.clip(self.generator.normal(0.15, 0.08), 0.02, 0.35))
                            is_sparse = True
                        else:
                            strat2_trials += 1
                            cr_eff = cr

            v = self._bound_constraint(v, parent_sol)

            # 算子 3: 地形感知的自适应特征子空间正交交叉
            use_eigen = (self.generator.uniform() < self.p_eigen)
            if use_eigen:
                eigen_trials += 1
                v_prime = np.dot(self.B_matrix.T, v)
                x_prime = np.dot(self.B_matrix.T, parent_sol)
                mask = self.generator.uniform(size=self.dim) < cr_eff
                mask[self.generator.integers(0, self.dim)] = True
                u_prime = np.where(mask, v_prime, x_prime)
                u = np.dot(self.B_matrix, u_prime)
            else:
                standard_trials += 1
                mask = self.generator.uniform(size=self.dim) < cr_eff
                mask[self.generator.integers(0, self.dim)] = True
                u = np.where(mask, v, parent_sol)

            u = self._bound_constraint(u, parent_sol)
            new_agent = self.generate_agent(u)

            if self.compare_target(new_agent.target, self.pop[i].target, self.problem.minmax):
                diff = abs(self.pop[i].target.fitness - new_agent.target.fitness)
                delta_f.append(diff)
                S_CR.append(cr)
                S_F.append(f_val)
                if use_strat1:
                    strat1_wins += 1
                elif not is_sparse:
                    strat2_wins += 1
                if use_eigen:
                    eigen_wins += 1
                else:
                    standard_wins += 1

                if self.max_archive_size > 0:
                    if len(self.archive) < self.max_archive_size:
                        self.archive.append(parent_sol.copy())
                    else:
                        rep_idx = self.generator.integers(0, self.max_archive_size)
                        self.archive[rep_idx] = parent_sol.copy()
                self.pop[i] = new_agent

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
            self.p_eigen = np.clip(self.p_eigen, 0.05, 0.95)

        if len(S_CR) > 0:
            sum_df = np.sum(delta_f)
            if sum_df > 1e-12:
                w = np.array(delta_f) / sum_df
                self.M_CR[self.k_mem] = np.sum(w * np.array(S_CR))
                self.M_F[self.k_mem] = np.sum(w * (np.array(S_F) ** 2)) / np.sum(w * np.array(S_F))
                self.k_mem = (self.k_mem + 1) % self.H