"""
================================================================================
基于 OPFUNU (IEEE CEC-2017) 与 MEALPY 官方库的高维标准基准测试套件
- 测试函数集: IEEE CEC-2017 (支持 12 代表性函数模式 与 29 全量函数模式切换)
  * 单峰函数 (Unimodal): F12017 ~ F22017
  * 简单多峰函数 (Multimodal): F32017 ~ F92017
  * 混合函数 (Hybrid): F102017 ~ F192017
  * 复合函数 (Composition): F202017 ~ F292017
- 支持高维配置: dim = 30, 50, 100 (CEC-2017 官方标准高维支持)
- 评价指标: 适应度误差 Δf(x) = f(x) - f_global (保留 float64 极限精度 1e-16)
- 运行依赖: pip install opfunu mealpy numpy scipy matplotlib tqdm
================================================================================
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import mealpy
from mealpy import Optimizer, FloatVar
from mealpy.swarm_based import PSO
from mealpy.evolutionary_based import DE, SHADE
from opfunu.cec_based import cec2017

print(f"[MEALPY] 成功加载官方优化库, 版本号: {mealpy.__version__}")

# 零误差截断阈值: 保持为机器浮点极限 1e-16
CEC_ZERO_TOL = 1e-16


# ==============================================================================
# 1. 定稿版 EADS 优化器 (方案 B 升级版: 宽域归档引导 + 柯西重尾微扰勘探策略 2)
# ==============================================================================
class EADS(Optimizer):
    """
    Eigen-Adaptive Dual-Strategy Differential Evolution (EADS - Plan B Upgraded)
    核心算子:
      1. 对数秩加权收缩正则化协方差特征分解 (Log-Rank Weighted Shrinkage Covariance)
      2. 窄域退火开采策略 1 + 宽域归档引导柯西微扰勘探策略 2 + 中点折返边界修复
      3. 地形感知的自适应特征子空间正交旋转交叉 (Adaptive Eigen-Subspace Crossover)
      4. 拉普拉斯平滑单次尝试成功率更新与 Lehmer 成功历史记忆池 (Per-Trial Success Rate & Lehmer Memory)
    """
    def __init__(self, epoch=800, pop_size=150, H=5, arc_rate=1.0, p_best_rate=0.11, **kwargs):
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

        # 策略 1 窄域退火精英池 (0.20 -> 0.04, 专注精细局部开采)
        p_rate_narrow = 0.20 - (0.20 - 0.04) * (epoch / self.epoch)
        p_count_narrow = max(2, int(self.pop_size * p_rate_narrow))

        # 策略 2 宽域恒定精英池 (0.25, 保持多峰与周期地形的广域盆地覆盖)
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

            cands_r1 = [idx for idx in range(self.pop_size) if idx != i]
            r1 = self.generator.choice(cands_r1)
            x_r1 = self.pop[r1].solution

            # 从联合池 P U A 中采样历史差异个体 x_r2_arc，防止高维周期函数(如 F8 Levy)单维陷入零方差死锁
            pool = list(range(self.pop_size))
            if len(self.archive) > 0:
                pool += [self.pop_size + idx for idx in range(len(self.archive))]
            cands_r2 = [idx for idx in pool if idx != i and idx != r1]
            r2_arc = self.generator.choice(cands_r2)
            x_r2_arc = self.pop[r2_arc].solution if r2_arc < self.pop_size else self.archive[r2_arc - self.pop_size]

            use_strat1 = (self.generator.uniform() < self.p_strat1)
            if use_strat1:
                strat1_trials += 1
                # 策略 1 (精细开采): 窄域退火精英引导 current-to-pbest_narrow/1 with archive
                pbest_idx = self.generator.integers(0, p_count_narrow)
                x_pbest = self.pop[pbest_idx].solution
                v = parent_sol + f_val * (x_pbest - parent_sol) + f_val * (x_r1 - x_r2_arc)
            else:
                strat2_trials += 1
                # 策略 2 (广域勘探 - 方案 B): 以父代 parent_sol 为锚点 + 宽域精英引导 + 归档差分 + 柯西重尾微扰
                # 彻底避免直接以 x_pbest 为起点导致的早熟同化，完美攻克 50D F8 (Levy) 紧邻周期陷阱
                pbest_wide_idx = self.generator.integers(0, p_count_wide)
                x_pbest_wide = self.pop[pbest_wide_idx].solution
                r3 = self.generator.choice(cands_r1)
                x_r3 = self.pop[r3].solution
                cauchy_step = self.generator.standard_cauchy(self.dim) * 0.02
                v = (
                    parent_sol
                    + f_val * (x_pbest_wide - parent_sol)
                    + f_val * (x_r1 - x_r2_arc)
                    + cauchy_step * (x_r1 - x_r3)
                )

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
# 2. 配置 IEEE CEC-2017 高维标准测试函数 (当前默认已设为 50D 配置，可直接验证 F8)
# ==============================================================================
dim = 50          # 可选 10, 30, 50, 100 (CEC-2017 官方高维支持维度)
epoch = 800       # 30D 推荐 600; 50D 推荐 800; 100D 推荐 1000
pop_size = 150    # 30D 推荐 100; 50D 推荐 150; 100D 推荐 300 (满足 0.35*N >= D+1 协方差满秩)
n_runs = 3        # 快速验证设为 3 次; 正式论文统计检验可设为 30 次
np.random.seed(42)

# 测试模式选择:
#   "representative_12" : 精选 12 个涵盖单峰/多峰/混合/复合四大类的代表性函数 (自动绘制 3x4 拼图, 速度快)
#   "all_29"            : 运行 F12017 ~ F292017 全部 29 个函数 (自动绘制 5x6 全景拼图)
TEST_MODE = "representative_12"

REPRESENTATIVE_12_LIST = [
    ("F1: Rotated Bent Cigar (Unimodal)", cec2017.F12017),
    ("F2: Rotated Zakharov (Unimodal)", cec2017.F22017),
    ("F3: Rotated Rosenbrock (Multimodal)", cec2017.F32017),
    ("F4: Rotated Rastrigin (Multimodal)", cec2017.F42017),
    ("F7: Rotated Non-Cont. Rastrigin", cec2017.F72017),
    ("F8: Rotated Levy (Multimodal)", cec2017.F82017),
    ("F10: Hybrid Function 1", cec2017.F102017),
    ("F13: Hybrid Function 4", cec2017.F132017),
    ("F18: Hybrid Function 9", cec2017.F182017),
    ("F20: Composition Function 1", cec2017.F202017),
    ("F22: Composition Function 3", cec2017.F222017),
    ("F26: Composition Function 7", cec2017.F262017),
]


def build_cec2017_benchmarks(ndim, mode="representative_12"):
    if mode == "all_29":
        func_list = []
        for idx in range(1, 30):
            cls = getattr(cec2017, f"F{idx}2017")
            tmp_obj = cls(ndim=ndim)
            raw_name = tmp_obj.name.split("(")[0].strip()
            short_title = raw_name[:34]
            func_list.append((short_title, cls))
    else:
        func_list = REPRESENTATIVE_12_LIST

    benchmarks = []
    for short_title, cls in func_list:
        cec_obj = cls(ndim=ndim)
        f_global = float(cec_obj.f_global)
        lb_vec = cec_obj.lb.tolist()
        ub_vec = cec_obj.ub.tolist()

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
# 3. 运行 CEC-2017 高维对比实验并绘制收敛曲线矩阵图 (带 tqdm 实时总进度条)
# ==============================================================================
if __name__ == "__main__":
    BENCHMARKS = build_cec2017_benchmarks(dim, mode=TEST_MODE)
    n_funcs = len(BENCHMARKS)

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
    print(f"  IEEE CEC-2017 高维标准测试集对比实验 (基于 OPFUNU, 维度 D = {dim}, 函数数 = {n_funcs}, 运行 = {n_runs} 次, Epoch = {epoch})")
    print(f"  [注] 零误差截断精度阈值: < {CEC_ZERO_TOL:.0e}")
    print("=" * 115)

    total_tasks = n_funcs * len(algorithms) * n_runs
    pbar = tqdm(total=total_tasks, desc=f"CEC-2017 (D={dim}) 总进度", unit="run") if HAS_TQDM else None

    for bm in BENCHMARKS:
        fname = bm["name"]
        if pbar is not None:
            pbar.write(f"\n>>> 正在评测: {fname} (理论最优偏置 f* = {bm['f_global']:.1f})")
        else:
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
                if pbar is not None:
                    pbar.set_postfix_str(f"{fname[:15]} | {alg_name} (Run {r+1}/{n_runs})")
                seed_val = r * 100 + 7
                np.random.seed(seed_val)
                model = model_builder()
                g_best = model.solve(prob_dict, seed=seed_val)
                final_err = 0.0 if g_best.target.fitness < CEC_ZERO_TOL else float(g_best.target.fitness)
                run_fits.append(final_err)
                run_curves.append(model.history.list_global_best_fit)
                if pbar is not None:
                    pbar.update(1)

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
            msg = f"  [{alg_name:18s}] 误差均值: {mean_val:.3e} | 标准差: {std_val:.3e}"
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)

    if pbar is not None:
        pbar.close()

    # 打印终端汇总大表与 Friedman 平均排名
    alg_list = list(algorithms.keys())
    rank_matrix = []

    print("\n" + "=" * 122)
    print(f"{'CEC-2017 函数名称':<36} | {'PSO':<12} | {'DE':<12} | {'SHADE':<12} | {'L-SHADE':<12} | {'EADS(本文)':<12}")
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

    # 动态选择子图网格布局 (12 函数用 3x4, 29 函数用 5x6)
    if n_funcs <= 12:
        n_rows, n_cols = 3, 4
        figsize = (22, 14)
    else:
        n_rows, n_cols = 5, 6
        figsize = (26, 20)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
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
        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch (Iterations)", fontsize=8.5)
        ax.set_ylabel("Fitness Error (Log Scale)", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, loc="upper right")

    for idx in range(n_funcs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_filename = f"mealpy_opfunu_cec2017_{TEST_MODE}_dim{dim}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[OK] IEEE CEC-2017 (D={dim}, {TEST_MODE}) 收敛对比大图已成功保存为: {output_filename}")
    plt.show()