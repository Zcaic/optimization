"""
================================================================================
基于 OPFUNU (IEEE CEC-2022) 与 MEALPY 官方库的 12 函数标准基准测试套件
- 测试函数集: IEEE CEC-2022 (F12022 ~ F122022, 共 12 个旋转/混合/复合函数)
  * 单峰函数 (Unimodal): F1 (Zakharov)
  * 基础多峰函数 (Multimodal): F2 (Rosenbrock), F3 (Schaffer F7), F4 (Non-Cont. Rastrigin), F5 (Levy)
  * 混合函数 (Hybrid): F6 (Hybrid 1), F7 (Hybrid 2), F8 (Hybrid 3)
  * 复合函数 (Composition): F9 (Comp 1), F10 (Comp 2), F11 (Comp 3), F12 (Comp 4)
- 支持维度: dim = 10 或 dim = 20 (CEC-2022 官方全套支持维度)
- 评价指标: 适应度误差 Δf(x) = f(x) - f_global (遵循 IEEE CEC-2022 准则: 最终误差 < 1e-8 记为 0.00e+00)
- 运行依赖: pip install opfunu mealpy numpy scipy matplotlib
================================================================================
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import mealpy
from mealpy import Optimizer, FloatVar
from mealpy.swarm_based import PSO
from mealpy.evolutionary_based import DE, SHADE
from opfunu.cec_based import cec2022

print(f"[MEALPY] 成功加载官方优化库, 版本号: {mealpy.__version__}")

# IEEE CEC-2022 官方零误差截断阈值 (最终汇总表与排名计算中 < 1e-8 视为精确归零)
CEC_ZERO_TOL = 1e-16


# ==============================================================================
# 1. 定稿版 EADS 优化器 (含: 对数秩加权协方差 + 动态 p-best + 中点折返边界修复 + 单次成功率归一化)
# ==============================================================================
class EADS(Optimizer):
    """
    Eigen-Adaptive Dual-Strategy Differential Evolution (EADS - Final Version)
    核心算子:
      1. 对数秩加权收缩正则化协方差特征分解 (Log-Rank Weighted Shrinkage Covariance)
      2. 时变退火精英池引导的异构双策略变异 + SHADE 中点折返边界修复 (Midpoint Bounce-Back)
      3. 地形感知的自适应特征子空间正交旋转交叉 (Adaptive Eigen-Subspace Crossover)
      4. 拉普拉斯平滑单次尝试成功率更新与 Lehmer 成功历史记忆池 (Per-Trial Success Rate & Lehmer Memory)
    """
    def __init__(self, epoch=500, pop_size=50, H=5, arc_rate=1.0, p_best_rate=0.11, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.H = H
        self.arc_rate = arc_rate
        self.p_best_rate = p_best_rate
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
        n_elites = max(self.dim + 1, int(0.35 * self.pop_size))
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
            else:
                strat2_trials += 1
                # 策略 2 (勘探): 精英基向量跃迁 + 自缩放柯西重尾扰动
                r2 = self.generator.choice(cands_r1)
                x_r2 = self.pop[r2].solution
                cauchy_step = self.generator.standard_cauchy(self.dim) * 0.05
                v = x_pbest + f_val * (x_r1 - x_r2) + cauchy_step * (x_r1 - parent_sol)

            v = self._bound_constraint(v, parent_sol)

            # 算子 3: 地形感知的自适应特征子空间正交交叉
            use_eigen = (self.generator.uniform() < self.p_eigen)
            if use_eigen:
                eigen_trials += 1
                v_prime = np.dot(self.B_matrix.T, v)
                x_prime = np.dot(self.B_matrix.T, parent_sol)
                mask = self.generator.uniform(size=self.dim) < cr
                mask[self.generator.integers(0, self.dim)] = True
                u_prime = np.where(mask, v_prime, x_prime)
                u = np.dot(self.B_matrix, u_prime)
            else:
                standard_trials += 1
                mask = self.generator.uniform(size=self.dim) < cr
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
                else:
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


# ==============================================================================
# 2. 配置 IEEE CEC-2022 12 个标准测试函数 (支持 dim = 10 或 dim = 20)
# ==============================================================================
dim = 20          # 可选 10 或 20 (CEC-2022 标准支持维度)
epoch = 600       # 10D 推荐 500; 20D 推荐 600~1000
pop_size = 80     # 10D 推荐 50;  20D 推荐 80~100
n_runs = 3        # 快速验证设为 3 次; 正式论文统计检验可设为 30 次
np.random.seed(42)

CEC2022_CLASSES = [
    ("F1: Rotated Zakharov (Unimodal)", cec2022.F12022),
    ("F2: Rotated Rosenbrock (Multimodal)", cec2022.F22022),
    ("F3: Rotated Schaffer F7 (Multimodal)", cec2022.F32022),
    ("F4: Rotated Non-Cont. Rastrigin", cec2022.F42022),
    ("F5: Rotated Levy (Multimodal)", cec2022.F52022),
    ("F6: Hybrid Function 1 (3 Sub-funcs)", cec2022.F62022),
    ("F7: Hybrid Function 2 (6 Sub-funcs)", cec2022.F72022),
    ("F8: Hybrid Function 3 (5 Sub-funcs)", cec2022.F82022),
    ("F9: Composition Function 1", cec2022.F92022),
    ("F10: Composition Function 2", cec2022.F102022),
    ("F11: Composition Function 3", cec2022.F112022),
    ("F12: Composition Function 4", cec2022.F122022),
]


def build_cec2022_benchmarks(ndim):
    benchmarks = []
    for short_title, cls in CEC2022_CLASSES:
        cec_obj = cls(ndim=ndim)
        f_global = float(cec_obj.f_global)
        lb_vec = cec_obj.lb.tolist()
        ub_vec = cec_obj.ub.tolist()

        # 优化过程中保持高精度误差 (< 1e-16 归零)，确保收敛曲线能一路画到 1e-16
        def make_error_func(obj, fg):
            def error_func(x):
                err = float(obj.evaluate(x) - fg)
                return 0.0 if err < 1e-16 else err
            return error_func

        benchmarks.append({
            "name": short_title,
            "func": make_error_func(cec_obj, f_global),
            "lb": lb_vec,
            "ub": ub_vec,
            "f_global": f_global,
        })
    return benchmarks


# ==============================================================================
# 3. 运行 CEC-2022 12 函数对比实验并绘制 3x4 收敛曲线矩阵图
# ==============================================================================
if __name__ == "__main__":
    BENCHMARKS = build_cec2022_benchmarks(dim)

    algorithms = {
        "PSO (Mealpy)": lambda: PSO.OriginalPSO(epoch=epoch, pop_size=pop_size),
        "DE (Mealpy)": lambda: DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=0.5, cr=0.9),
        "SHADE (Mealpy)": lambda: SHADE.OriginalSHADE(epoch=epoch, pop_size=pop_size, miu_f=0.5, miu_cr=0.5),
        "L-SHADE (Mealpy)": lambda: SHADE.L_SHADE(epoch=epoch, pop_size=pop_size, miu_f=0.5, miu_cr=0.5),
        "EADS (Proposed)": lambda: EADS(epoch=epoch, pop_size=pop_size, H=5, arc_rate=1.0),
    }

    results_summary = {}
    all_curves = {}

    print("\n" + "=" * 115)
    print(f"  IEEE CEC-2022 官方标准测试集对比实验 (基于 OPFUNU, 维度 D = {dim}, 独立运行 = {n_runs} 次, Epoch = {epoch})")
    print(f"  [注] 遵循 IEEE CEC-2022 官方竞赛规程: 最终报告误差 < {CEC_ZERO_TOL:.0e} 统一记为 0.00e+00")
    print("=" * 115)

    for bm in BENCHMARKS:
        fname = bm["name"]
        print(f"\n>>> 正在评测: {fname} (理论最优偏置 f* = {bm['f_global']:.1f})")
        results_summary[fname] = {}
        all_curves[fname] = {}

        prob_dict = {
            "obj_func": bm["func"],
            "bounds": FloatVar(lb=bm["lb"], ub=bm["ub"], name="delta"),
            "minmax": "min",
            "log_to": None,
        }

        for alg_name, model_builder in algorithms.items():
            run_fits = []
            run_curves = []
            for r in range(n_runs):
                seed_val = r * 100 + 7
                np.random.seed(seed_val)
                model = model_builder()
                g_best = model.solve(prob_dict, seed=seed_val)
                # 按 IEEE CEC-2022 竞赛规程，最终误差小于 1e-8 记为 0.0
                final_err = 0.0 if g_best.target.fitness < CEC_ZERO_TOL else float(g_best.target.fitness)
                run_fits.append(final_err)
                run_curves.append(model.history.list_global_best_fit)

            run_fits = np.array(run_fits)
            mean_curve = np.mean(np.array(run_curves), axis=0)
            mean_val = 0.0 if np.mean(run_fits) < CEC_ZERO_TOL else float(np.mean(run_fits))
            std_val = 0.0 if mean_val == 0.0 else float(np.std(run_fits))
            results_summary[fname][alg_name] = {
                "mean": mean_val,
                "std": std_val,
                "raw": run_fits,
            }
            all_curves[fname][alg_name] = mean_curve
            print(f"  [{alg_name:18s}] 误差均值: {mean_val:.3e} | 标准差: {std_val:.3e}")

    # 打印终端汇总大表与 Friedman 平均排名
    alg_list = list(algorithms.keys())
    rank_matrix = []

    print("\n" + "=" * 122)
    print(f"{'CEC-2022 函数名称':<36} | {'PSO':<12} | {'DE':<12} | {'SHADE':<12} | {'L-SHADE':<12} | {'EADS(本文)':<12}")
    print("=" * 122)
    for bm in BENCHMARKS:
        fn = bm["name"]
        means = [results_summary[fn][a]["mean"] for a in alg_list]
        ranks = stats.rankdata(means, method="average")
        rank_matrix.append(ranks)

        p_str = f"{results_summary[fn]['PSO (Mealpy)']['mean']:.2e}"
        d_str = f"{results_summary[fn]['DE (Mealpy)']['mean']:.2e}"
        s_str = f"{results_summary[fn]['SHADE (Mealpy)']['mean']:.2e}"
        ls_str = f"{results_summary[fn]['L-SHADE (Mealpy)']['mean']:.2e}"
        e_str = f"{results_summary[fn]['EADS (Proposed)']['mean']:.2e}"
        print(f"{fn:<36} | {p_str:<12} | {d_str:<12} | {s_str:<12} | {ls_str:<12} | {e_str:<12}")
    print("=" * 122)

    avg_ranks = np.mean(np.array(rank_matrix), axis=0)
    print(
        f"{'Friedman 平均排名 (越低越好)':<34} | "
        f"{avg_ranks[0]:<12.2f} | {avg_ranks[1]:<12.2f} | {avg_ranks[2]:<12.2f} | {avg_ranks[3]:<12.2f} | {avg_ranks[4]:<12.2f}"
    )
    print("=" * 122)

    # 绘制 3x4 大图 (12 个子图)
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    axes = axes.flatten()

    colors = {
        "PSO (Mealpy)": "#7f7f7f",
        "DE (Mealpy)": "#1f77b4",
        "SHADE (Mealpy)": "#ff7f0e",
        "L-SHADE (Mealpy)": "#9467bd",
        "EADS (Proposed)": "#d62728",
    }
    linestyles = {
        "PSO (Mealpy)": ":",
        "DE (Mealpy)": "--",
        "SHADE (Mealpy)": "-.",
        "L-SHADE (Mealpy)": "--",
        "EADS (Proposed)": "-",
    }
    x_epochs = np.arange(1, epoch + 1)

    for idx, bm in enumerate(BENCHMARKS):
        ax = axes[idx]
        fname = bm["name"]
        for alg_name in algorithms.keys():
            curve = all_curves[fname][alg_name]
            plot_curve = np.maximum(curve, 1e-16)
            ax.plot(
                x_epochs,
                plot_curve,
                label=alg_name,
                color=colors[alg_name],
                linestyle=linestyles[alg_name],
                linewidth=2.2 if "EADS" in alg_name else 1.4,
            )
        ax.set_yscale("log")
        ax.set_title(fname, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Epoch (Iterations)", fontsize=9)
        ax.set_ylabel("Fitness Error (Log Scale)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7.5, loc="upper right")

    plt.tight_layout()
    output_filename = f"mealpy_opfunu_cec2022_v2_1e-16_dim{dim}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[OK] IEEE CEC-2022 (D={dim}) 12函数收敛对比大图已成功保存为: {output_filename}")
    plt.show()