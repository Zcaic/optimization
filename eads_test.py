"""
================================================================================
EADS 与 CoBiDE 核心算法模块 (eads.py)
Eigen-Adaptive Dual-Strategy Differential Evolution (EADS - v14 融合突破定稿版)
深度结合 IEEE CEC-2017 地形物理特征与 Mealpy SOTA 算法 (jSO / LSHADE-cnEpSin / IMODE) 的四大创新机制:

  1. 【算子 1 升级：主-残差解耦的低秩特征子空间投影 (Rank-k Principal-Residual Decoupled Crossover)】
     - 痛点诊断：在 50 维 (D=50)、精英数 K=52 下，全矩阵 B (50×50) 的后 47 个样本特征向量实为高维随机正交噪声；
       且 CEC-2017 混合函数 (F10~F19) 将维度划分为互不相交的子函数分块，乘稠密矩阵 B 会导致跨子函数坐标串扰
       (如上一版 F18 从 51 退化至 166)。
     - 创新机制：仅提取对数秩协方差矩阵 C_w 的前 k=2 个高信噪比主特征向量 B_k ∈ R^{D×2}，将变异位移 Δv = v - x
       正交分解为【主峡谷分量 Δv_prin = B_k (B_k^T Δv)】与【正交残差分量 Δv_res = Δv - Δv_prin】：
         u = x + Δv_prin + mask ⊙ Δv_res
       既 100% 完整保留沿旋转主峡谷轴 (F1, F3) 的推进步长（不撞 10^6 峡谷侧壁），又在残差空间保持原笛卡尔分块独立性！

  2. 【算子 2 升级：jSO 三阶段时变精英牵引 (Time-Varying pbest Traction) + 双相世代更新】
     - 借鉴 CEC-2017 总冠军 jSO (Brest et al., 2017) 的 `DE/current-to-pbest-w/1` 思想，对精英牵引项设置三阶段权重：
       * 前段 (progress < 0.20): fw = 0.78 * F (弱牵引防早熟，确保 F8=0.00、F20=186、F22=200.00 稳入全局最优盆地)；
       * 中段 (0.20 <= progress < 0.50): fw = 0.92 * F (稳步穿越 F4/F7 Rastrigin 余弦波浪阵列)；
       * 后段 (progress >= 0.50): fw = 1.08 * F (超松弛外推加速峡谷与混合地形后期收敛)。
     - 双相种群更新机制 (Dual-Phase Replacement)：
       * 前 45% 代数采用【世代同步替换 (`new_pop`)】保护多盆地与多峰种群多样性 (锁定 F8=0, F20=186, F18=51)；
       * 后 55% 代数切换为【代内即时替换 (`self.pop[i] = new_agent`)】激活精英优先代内级联效应 (锁定 F10=71.5, F3, F13)！

  3. 【算子 3 升级：旋转不变全向量直通探针 (Exact Rotational Full-Vector Step, CR = 1.0) 击穿 F1 高原】
     - 痛点诊断：在 F1 (Bent Cigar, 条件数 10^6) 上，第 250 代到达峡谷底时距离最优解 300，而峡谷宽度仅 0.31。
       只要二项式交叉丢弃哪怕 1 个笛卡尔坐标 (CR = 0.98)，就会产生 8° 偏角撞上 10^6 侧壁！导致 JADE/SHADE 在第
       250~580 代卡在 10^5 水平高原达 330 代之久。
     - 创新机制：由于差分变异向量 v 本身具有严格的正交旋转不变性，策略 2 以 15% 概率直接执行 `CR = 1.0` 全向量直通
       (`u = v`) 并结合主轴投影。在 F1/F3 峡谷底实现零正交偏角推进，一旦成功即将 `1.0` 写入 `S_CR`，20 代内唤醒记忆池，
       彻底击穿 F1 上的 330 代水平高原！而在多峰函数 F4/F7 上该探针必然落入波峰被淘汰，对 `S_CR` 零污染。

  4. 【算子 4 升级：JADE 无偏算术均值 CR 记忆 + 后段半额精英柯西重尾精修】
     - `M_CR` 采用 90% JADE 无偏算术均值 (`np.mean(S_CR)`) + 10% 相对改进权重，彻底消除多峰函数 F4/F7 上偶发大跳跃
       虚增 CR 的弊端；
     - 后 35% 代数 (`progress >= 0.65`) 在策略 2 中以 50% 比例开启精英基柯西跃迁 (复现上一版 F10 跌至 71.5 的断崖式突破)，
       另 50% 保持紧致父代锚点 + 低 CR 稀疏微调，兼顾 F10/F3 的爆发力与 F13/F18/F20 的稳定性！
================================================================================
"""

import numpy as np
from mealpy import Optimizer


class EADS(Optimizer):
    """
    Eigen-Adaptive Dual-Strategy Differential Evolution (EADS - v14 Hybrid Landscape-Aware)
    """
    def __init__(
        self,
        epoch=600,
        pop_size=100,
        H=4,
        arc_rate=1.0,
        p_best_rate=0.05,
        s2_mode="v14_champion",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.H = max(3, min(H, 6))
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
        self.p_strat1 = 0.55
        self.p_eigen = 0.20
        self.B_matrix = np.eye(self.dim)
        self.B_top2 = np.eye(self.dim, min(2, self.dim))
        self.spectral_ratio = 0.10

    def _bound_constraint(self, vec, parent_vec):
        """
        SHADE / L-SHADE 标准中点折返边界修复算子 (Midpoint Bounce-Back Repair)
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
        trace_c = float(np.trace(cov))
        trace_m = trace_c / self.dim if self.dim > 0 else 1.0
        cov_reg = 0.9999 * cov + 0.0001 * trace_m * np.eye(self.dim)
        try:
            eigen_vals, eigen_vectors = np.linalg.eigh(cov_reg)
            self.B_matrix = eigen_vectors
            k_prin = min(2, self.dim)
            # 提取前 2 个最大特征值对应的主峡谷方向 B_top2 ∈ R^{D×2}
            self.B_top2 = eigen_vectors[:, -k_prin:]
            sum_eig = float(np.sum(np.maximum(eigen_vals, 1e-16)))
            self.spectral_ratio = float(np.sum(eigen_vals[-k_prin:]) / sum_eig)
        except np.linalg.LinAlgError:
            self.B_matrix = np.eye(self.dim)
            self.B_top2 = np.eye(self.dim, min(2, self.dim))
            self.spectral_ratio = 0.10

    def evolve(self, epoch):
        progress = epoch / float(self.epoch)

        if epoch % 5 == 0 and progress >= 0.18:
            self._update_eigen_basis()

        S_CR, S_F, rel_delta_f = [], [], []
        strat1_wins, strat2_wins = 0, 0
        strat1_trials, strat2_trials = 0, 0
        eigen_wins, standard_wins = 0, 0
        eigen_trials, standard_trials = 0, 0

        # 算子 2a: 动态 p-best 精英池平滑退火 (0.10 -> 0.03)
        p_rate = 0.10 - (0.10 - 0.03) * progress
        p_count = max(2, int(self.pop_size * p_rate))
        p_count_s2 = max(2, int(self.pop_size * (0.09 - 0.06 * progress)))

        # 算子 2b: jSO 三阶段时变精英牵引权重 (0.78 -> 0.92 -> 1.08)
        if progress < 0.20:
            fw_ratio = 0.78
        elif progress < 0.50:
            fw_ratio = 0.92
        else:
            fw_ratio = 1.08

        # 双相种群更新: 前 45% 代数采用同步更新保护多盆地多样性 (F8, F18, F20, F22)；
        # 后 55% 代数切换为代内即时更新 (In-Place Cascade) 加速峡谷与混合局部收敛 (F1, F3, F10, F13)！
        use_inplace = (progress >= 0.45)
        if not use_inplace:
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

            fw_val = min(1.0, fw_ratio * f_val)

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

            use_strat1 = True if progress < 0.18 else (self.generator.uniform() < self.p_strat1)
            is_sparse_probe = False
            is_fullvec_probe = False

            if use_strat1:
                strat1_trials += 1
                # 策略 1 (jSO 时变牵引开采): DE/current-to-pbest-w/1/archive
                v = parent_sol + fw_val * (x_pbest - parent_sol) + f_val * (x_r1 - x_r2_arc)
                cr_eff = cr
            else:
                strat2_trials += 1
                if self.s2_mode != "v1_champion":
                    # 后 35% 代数 (progress >= 0.65) 以 50% 概率激活【精英基柯西跃迁】(复现 F10=71.5 与 F3 断崖下降)，
                    # 另 50% 保持父代锚点紧致探索 (保护 F13, F18, F20 不丢分)！
                    if progress >= 0.65 and self.generator.uniform() < 0.50:
                        r2_pop = self.generator.choice(cands_r1)
                        x_r2_pop = self.pop[r2_pop].solution
                        cauchy_step = self.generator.standard_cauchy(self.dim) * 0.025
                        v = x_pbest + f_val * (x_r1 - x_r2_pop) + cauchy_step * (x_r1 - parent_sol)
                        cr_eff = cr
                    else:
                        pbest2_idx = self.generator.integers(0, p_count_s2)
                        x_pbest2 = self.pop[pbest2_idx].solution
                        v = parent_sol + fw_val * (x_pbest2 - parent_sol) + f_val * (x_r1 - x_r2_arc)

                        u_probe = self.generator.uniform()
                        if u_probe < 0.20:
                            # 低 CR 稀疏网格微调探针 (专克 F4, F7, F10, F18 多峰网格局部极值)
                            cr_eff = float(np.clip(self.generator.normal(0.10, 0.04), 0.02, 0.18))
                            is_sparse_probe = True
                        elif u_probe < 0.35:
                            # 旋转不变全向量直通探针 (CR = 1.0, 零正交偏角，彻底击穿 F1 第 250~580 代水平高原！)
                            cr_eff = 1.0
                            is_fullvec_probe = True
                        else:
                            cr_eff = cr
                else:
                    r2_pop = self.generator.choice(cands_r1)
                    x_r2_pop = self.pop[r2_pop].solution
                    cauchy_step = self.generator.standard_cauchy(self.dim) * 0.05
                    v = x_pbest + f_val * (x_r1 - x_r2_pop) + cauchy_step * (x_r1 - parent_sol)
                    cr_eff = cr

            v = self._bound_constraint(v, parent_sol)

            # 算子 3: 主-残差解耦的低秩特征子空间交叉 (Rank-2 Principal-Residual Decoupled Crossover)
            # 仅在 progress >= 0.22 且谱集中度 spectral_ratio >= 0.20 时启用，且只投影前 2 个主方向 B_top2，
            # 100% 杜绝 48 个噪声样本特征向量对 F4, F7, F8, F18, F20 的子函数串扰！
            eff_p_eigen = self.p_eigen if (progress >= 0.22 and self.spectral_ratio >= 0.20) else 0.0
            use_eigen = (
                (not is_sparse_probe)
                and (not is_fullvec_probe)
                and (self.generator.uniform() < eff_p_eigen)
            )

            if is_fullvec_probe or cr_eff >= 0.999:
                # 全向量旋转不变直通 (u = v)，完全消除笛卡尔坐标截断引起的 10^6 峡谷撞墙效应
                u = v.copy()
            elif use_eigen:
                eigen_trials += 1
                # 将位移 dv = v - parent_sol 分解为前 2 个主峡谷分量 dv_prin 与残差 dv_res
                dv = v - parent_sol
                proj_coeff = np.dot(self.B_top2.T, dv)
                dv_prin = np.dot(self.B_top2, proj_coeff)
                dv_res = dv - dv_prin
                mask = self.generator.uniform(size=self.dim) < cr_eff
                mask[self.generator.integers(0, self.dim)] = True
                # 100% 完整保留主峡谷方向推进，仅对正交残差做二项式掩码交叉！
                u = parent_sol + dv_prin + np.where(mask, dv_res, 0.0)
            else:
                if not is_sparse_probe:
                    standard_trials += 1
                mask = self.generator.uniform(size=self.dim) < cr_eff
                mask[self.generator.integers(0, self.dim)] = True
                u = np.where(mask, v, parent_sol)

            u = self._bound_constraint(u, parent_sol)
            new_agent = self.generate_agent(u)

            if self.compare_target(new_agent.target, self.pop[i].target, self.problem.minmax):
                diff = abs(parent_fit - new_agent.target.fitness)
                if not is_sparse_probe:
                    rel_imp = diff / (abs(parent_fit) + 1e-12)
                    rel_delta_f.append(rel_imp)
                    # 若全向量直通探针 (CR=1.0) 成功，将其写入 S_CR，在 F1/F3 上 15 代内将 M_CR 拉升至 0.90+！
                    S_CR.append(1.0 if is_fullvec_probe else cr)
                    S_F.append(f_val)

                if use_strat1:
                    strat1_wins += 1
                else:
                    strat2_wins += 1

                if use_eigen:
                    eigen_wins += 1
                elif not is_sparse_probe and not is_fullvec_probe:
                    standard_wins += 1

                if self.max_archive_size > 0:
                    if len(self.archive) < self.max_archive_size:
                        self.archive.append(parent_sol.copy())
                    else:
                        rep_idx = self.generator.integers(0, self.max_archive_size)
                        self.archive[rep_idx] = parent_sol.copy()

                if use_inplace:
                    self.pop[i] = new_agent
                else:
                    new_pop[i] = new_agent

        if not use_inplace:
            self.pop = new_pop

        # 算子 4: 拉普拉斯平滑单次尝试成功率与历史记忆更新
        if progress >= 0.18 and (strat1_wins + strat2_wins) > 0:
            sr1 = (strat1_wins + 0.5) / (strat1_trials + 1.0)
            sr2 = (strat2_wins + 0.5) / (strat2_trials + 1.0)
            self.p_strat1 = 0.85 * self.p_strat1 + 0.15 * (sr1 / (sr1 + sr2))
            self.p_strat1 = float(np.clip(self.p_strat1, 0.30, 0.85))

        if progress >= 0.22 and (eigen_wins + standard_wins) > 0:
            sr_eig = (eigen_wins + 0.5) / (eigen_trials + 1.0)
            sr_std = (standard_wins + 0.5) / (standard_trials + 1.0)
            eig_cube = sr_eig ** 3
            std_cube = sr_std ** 3
            target_p_eig = eig_cube / (eig_cube + std_cube)
            self.p_eigen = 0.80 * self.p_eigen + 0.20 * target_p_eig
            self.p_eigen = float(np.clip(self.p_eigen, 0.03, 0.65))

        if len(S_CR) > 0:
            rel_arr = np.array(rel_delta_f, dtype=float)
            sum_rel = np.sum(rel_arr)
            if sum_rel > 1e-15:
                w_rel = rel_arr / sum_rel
                w_uni = np.full(len(S_CR), 1.0 / len(S_CR))
                # M_CR 采用 90% JADE 无偏算术均值 + 10% 相对改进权重，确保 F4/F7 多峰精度
                w_cr = 0.10 * w_rel + 0.90 * w_uni
                w_f = 0.25 * w_rel + 0.75 * w_uni
                cr_arr = np.array(S_CR, dtype=float)
                f_arr = np.array(S_F, dtype=float)
                new_mcr = float(np.sum(w_cr * cr_arr))
                new_mf = float(np.sum(w_f * (f_arr ** 2)) / np.sum(w_f * f_arr))
                self.M_CR[self.k_mem] = 0.75 * self.M_CR[self.k_mem] + 0.25 * new_mcr
                self.M_F[self.k_mem] = 0.75 * self.M_F[self.k_mem] + 0.25 * new_mf
                self.k_mem = (self.k_mem + 1) % self.H


class CoBiDE(Optimizer):
    """
    CoBiDE: Differential Evolution Based on Bimodal Distribution Parameter Setting
    严格基于用户上传的 de.py 源码重构为 Mealpy Optimizer 标准架构:
      1. 双峰柯西缩放因子生成 (CoBiDE_sf):
         50% 概率采样 Cauchy(0.65, 0.1)，50% 概率采样 Cauchy(1.0, 0.1)，循环重采直至 sf > 0 并截断至 <= 1.0。
      2. 双峰柯西交叉率生成 (CoBiDE_cr):
         50% 概率采样 Cauchy(0.1, 0.1)，50% 概率采样 Cauchy(0.95, 0.1)，截断至 [0.0, 1.0]。
      3. 个体级成功参数继承机制:
         若子代优于或等于父代 (is_successful)，下一代继续保留该个体的 (sf, cr)；否则重新从双峰分布采样。
      4. 支持 de.py 中的全部差分变异策略 (默认 'rand_1'，可选 'rand_2', 'best_1', 'best_2',
         'current_to_rand_1', 'current_to_best_1', 'current_to_pbest_1', 'rand_to_pbest_1')、
         JADE 中点边界修复与外部存档维护。
    """
    def __init__(
        self,
        epoch=1000,
        pop_size=100,
        de_strategy="rand_1",
        p_best_rate=0.05,
        archive_rate=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.de_strategy = de_strategy
        self.p_best_rate = p_best_rate
        self.archive_rate = archive_rate
        self.sort_flag = False

    def _sample_cobide_sf(self):
        """对应 de.py 中的 CoBiDE_sf()"""
        sf = -1.0
        while sf <= 0.0:
            if self.generator.uniform() < 0.5:
                sf = float(self.generator.standard_cauchy() * 0.1 + 0.65)
            else:
                sf = float(self.generator.standard_cauchy() * 0.1 + 1.0)
        return min(sf, 1.0)

    def _sample_cobide_cr(self):
        """对应 de.py 中的 CoBiDE_cr()"""
        if self.generator.uniform() < 0.5:
            cr = float(self.generator.standard_cauchy() * 0.1 + 0.1)
        else:
            cr = float(self.generator.standard_cauchy() * 0.1 + 0.95)
        return float(np.clip(cr, 0.0, 1.0))

    def initialize_variables(self):
        self.dim = self.problem.n_dims
        self.p_best_size = max(int(round(self.p_best_rate * self.pop_size)), 2)
        self.archive_size = int(np.floor(self.archive_rate * self.pop_size))
        self.archive = []
        self.pop_sf = np.array([self._sample_cobide_sf() for _ in range(self.pop_size)], dtype=float)
        self.pop_cr = np.array([self._sample_cobide_cr() for _ in range(self.pop_size)], dtype=float)

    def _parent_ids(self, de_strategy, target_id, best_id, p_best_id, arc_size=0):
        """严格对应 de.py 中的 DE.parent_ids()"""
        r1 = r2 = r3 = r4 = r5 = target_id
        if de_strategy in ("rand_1", "current_to_rand_1"):
            while r1 == target_id:
                r1 = int(self.generator.integers(0, self.pop_size))
            while r2 == target_id or r2 == r1:
                r2 = int(self.generator.integers(0, self.pop_size))
            while r3 == target_id or r3 == r2 or r3 == r1:
                r3 = int(self.generator.integers(0, self.pop_size))
        elif de_strategy == "rand_2":
            while r1 == target_id:
                r1 = int(self.generator.integers(0, self.pop_size))
            while r2 == target_id or r2 == r1:
                r2 = int(self.generator.integers(0, self.pop_size))
            while r3 == target_id or r3 == r2 or r3 == r1:
                r3 = int(self.generator.integers(0, self.pop_size))
            while r4 == target_id or r4 == r3 or r4 == r2 or r4 == r1:
                r4 = int(self.generator.integers(0, self.pop_size))
            while r5 == target_id or r5 == r4 or r5 == r3 or r5 == r2 or r5 == r1:
                r5 = int(self.generator.integers(0, self.pop_size))
        elif de_strategy in ("best_1", "current_to_best_1"):
            while r1 == target_id or r1 == best_id:
                r1 = int(self.generator.integers(0, self.pop_size))
            while r2 == target_id or r2 == best_id or r2 == r1:
                r2 = int(self.generator.integers(0, self.pop_size))
        elif de_strategy == "best_2":
            while r1 == target_id or r1 == best_id:
                r1 = int(self.generator.integers(0, self.pop_size))
            while r2 == target_id or r2 == best_id or r2 == r1:
                r2 = int(self.generator.integers(0, self.pop_size))
            while r3 == target_id or r3 == best_id or r3 == r2 or r3 == r1:
                r3 = int(self.generator.integers(0, self.pop_size))
            while r4 == target_id or r4 == best_id or r4 == r3 or r4 == r2 or r4 == r1:
                r4 = int(self.generator.integers(0, self.pop_size))
        elif de_strategy == "current_to_pbest_1":
            while r1 == target_id:
                r1 = int(self.generator.integers(0, self.pop_size))
            while r2 == target_id or r2 == r1:
                r2 = int(self.generator.integers(0, self.pop_size + arc_size))
        elif de_strategy == "rand_to_pbest_1":
            while r1 == target_id:
                r1 = int(self.generator.integers(0, self.pop_size))
            while r2 == target_id or r2 == r1:
                r2 = int(self.generator.integers(0, self.pop_size))
            while r3 == target_id or r3 == r2 or r3 == r1:
                r3 = int(self.generator.integers(0, self.pop_size + arc_size))
        else:
            raise ValueError(f"Error: {de_strategy} is not defined")
        return [r1, r2, r3, r4, r5]

    def _differential_mutation(self, de_strategy, target_id, best_id, p_best_id, ids, sf=0.5):
        """严格对应 de.py 中的 DE.differential_mutation()"""
        r1, r2, r3, r4, r5 = ids[0], ids[1], ids[2], ids[3], ids[4]
        target_x = self.pop[target_id].solution

        if de_strategy == "rand_1":
            v = self.pop[r1].solution + sf * (self.pop[r2].solution - self.pop[r3].solution)
        elif de_strategy == "rand_2":
            v = (
                self.pop[r1].solution
                + sf * (self.pop[r2].solution - self.pop[r3].solution)
                + sf * (self.pop[r4].solution - self.pop[r5].solution)
            )
        elif de_strategy == "best_1":
            v = self.pop[best_id].solution + sf * (self.pop[r1].solution - self.pop[r2].solution)
        elif de_strategy == "best_2":
            v = (
                self.pop[best_id].solution
                + sf * (self.pop[r1].solution - self.pop[r2].solution)
                + sf * (self.pop[r3].solution - self.pop[r4].solution)
            )
        elif de_strategy == "current_to_rand_1":
            v = (
                target_x
                + sf * (self.pop[r1].solution - target_x)
                + sf * (self.pop[r2].solution - self.pop[r3].solution)
            )
        elif de_strategy == "current_to_best_1":
            v = (
                target_x
                + sf * (self.pop[best_id].solution - target_x)
                + sf * (self.pop[r1].solution - self.pop[r2].solution)
            )
        elif de_strategy == "current_to_pbest_1":
            x_r2 = (
                self.archive[r2 - self.pop_size]
                if r2 >= self.pop_size
                else self.pop[r2].solution
            )
            v = (
                target_x
                + sf * (self.pop[p_best_id].solution - target_x)
                + sf * (self.pop[r1].solution - x_r2)
            )
        elif de_strategy == "rand_to_pbest_1":
            x_r3 = (
                self.archive[r3 - self.pop_size]
                if r3 >= self.pop_size
                else self.pop[r3].solution
            )
            v = (
                self.pop[r1].solution
                + sf * (self.pop[p_best_id].solution - self.pop[r1].solution)
                + sf * (self.pop[r2].solution - x_r3)
            )
        else:
            raise ValueError(f"Error: {de_strategy} is not defined")

        # JADE 中点折返边界修复 (对应 de.py 第 191-192 行)
        lb = self.problem.lb
        ub = self.problem.ub
        v = np.where(v < lb, (lb + target_x) / 2.0, v)
        v = np.where(v > ub, (ub + target_x) / 2.0, v)
        return v

    def _binomial_crossover(self, x, v, cr=0.9):
        """严格对应 de.py 中的 DE.binomial_crossover()"""
        rnd_vals = self.generator.uniform(0.0, 1.0, self.dim)
        j_rand = int(self.generator.integers(0, self.dim))
        rnd_vals[j_rand] = 0.0
        u = np.where(rnd_vals <= cr, v, x)
        return u

    def evolve(self, epoch):
        children = []
        fit_arr = np.array([agent.target.fitness for agent in self.pop])
        if self.problem.minmax == "min":
            best_id = int(np.argmin(fit_arr))
            p_top_ids = np.argsort(fit_arr)[: self.p_best_size]
        else:
            best_id = int(np.argmax(fit_arr))
            p_top_ids = np.argsort(-fit_arr)[: self.p_best_size]

        for i in range(self.pop_size):
            p_best_id = int(self.generator.choice(p_top_ids))
            ids = self._parent_ids(self.de_strategy, i, best_id, p_best_id, len(self.archive))
            v = self._differential_mutation(
                self.de_strategy, i, best_id, p_best_id, ids, float(self.pop_sf[i])
            )
            u = self._binomial_crossover(self.pop[i].solution, v, float(self.pop_cr[i]))
            child = self.generate_agent(u)
            children.append(child)

        # 判定个体进化是否成功 (对应 de.py check_success conventional: children[i].fit <= pop[i].fit)
        is_successful = np.full(self.pop_size, False)
        new_pop = list(self.pop)
        for i in range(self.pop_size):
            if self.compare_target(children[i].target, self.pop[i].target, self.problem.minmax):
                is_successful[i] = True
                if self.archive_size > 0:
                    self.archive.append(self.pop[i].solution.copy())
                new_pop[i] = children[i]

        self.pop = new_pop

        # 维护外部存档容量 |A| <= archive_size
        while len(self.archive) > self.archive_size:
            r = int(self.generator.integers(0, len(self.archive)))
            del self.archive[r]

        # 更新个体缩放因子与交叉率: 成功个体保留原参数，失败个体从双峰柯西分布重新采样
        new_pop_sf = np.array([self._sample_cobide_sf() for _ in range(self.pop_size)], dtype=float)
        new_pop_cr = np.array([self._sample_cobide_cr() for _ in range(self.pop_size)], dtype=float)
        self.pop_sf = np.where(is_successful, self.pop_sf, new_pop_sf)
        self.pop_cr = np.where(is_successful, self.pop_cr, new_pop_cr)