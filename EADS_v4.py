"""
================================================================================
基于 MEALPY 3.1.0 官方库的 8 大平移旋转测试函数完整对比套件 (已修复元组解包与键名对齐)
对比阵容:
  1. PSO (Mealpy)         - 经典群智能基准
  2. DE (Mealpy)          - 经典差分进化基准
  3. CMA-ES (Mealpy)      - 协方差矩阵进化策略基准 (Hansen, 2003)
  4. SHADE (Mealpy)       - CEC 2013 冠军级基准
  5. L-SHADE (Mealpy)     - CEC 2014 冠军级基准
  6. EADS (Fixed)         - 本文提出 (恒定满秩种群)
  7. EADS-LPSR (IsoFE)    - 本文提出 (带保秩归档补充的线性种群缩减版, 等算力预算)
================================================================================
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import mealpy
from mealpy import Optimizer, FloatVar
from mealpy.swarm_based import PSO
from mealpy.evolutionary_based import DE, SHADE, ES

print(f"[MEALPY] 成功加载官方优化库, 版本号: {mealpy.__version__}")

# ==============================================================================
# 1. 本文提出的 EADS 优化器 (支持恒定种群与保秩 LPSR 线性缩减模式)
# ==============================================================================
class EADS(Optimizer):
    """
    Eigen-Adaptive Dual-Strategy Differential Evolution (EADS)
    """
    def __init__(self, epoch=500, pop_size=100, H=5, arc_rate=1.4, p_best_rate=0.11,
                 use_lpsr=False, min_pop_size=10, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [4, 10000])
        self.init_pop_size = self.pop_size
        self.H = H
        self.arc_rate = arc_rate
        self.p_best_rate = p_best_rate
        self.use_lpsr = use_lpsr
        self.min_pop_size = max(4, min_pop_size)
        self.sort_flag = True

    def initialize_variables(self):
        self.dim = self.problem.n_dims
        self.pop_size = self.init_pop_size
        self.M_CR = np.full(self.H, 0.5)
        self.M_F = np.full(self.H, 0.5)
        self.k_mem = 0
        self.archive = []
        self.max_archive_size = int(self.pop_size * self.arc_rate)
        self.p_strat1 = 0.5
        self.p_eigen = 0.5
        self.B_matrix = np.eye(self.dim)

    def _update_eigen_basis(self):
        curr_n = len(self.pop)
        n_elites = max(2, int(0.35 * curr_n))
        elites_list = [agent.solution for agent in self.pop[:n_elites]]
        
        # 归档集辅助保秩机制: 当 LPSR 后期个体少于 D+1 时, 从归档集补充样本防止协方差矩阵奇异
        if len(elites_list) < self.dim + 1 and len(self.archive) > 0:
            needed = (self.dim + 1) - len(elites_list)
            take = min(needed, len(self.archive))
            idx_arc = self.generator.choice(len(self.archive), size=take, replace=False)
            for idx in idx_arc:
                elites_list.append(self.archive[idx])
                
        if len(elites_list) < 3:
            self.B_matrix = np.eye(self.dim)
            return

        elites = np.array(elites_list)
        cov = np.cov(elites, rowvar=False)
        trace_m = np.trace(cov) / self.dim if self.dim > 0 else 1.0
        cov_reg = 0.9999 * cov + 0.0001 * trace_m * np.eye(self.dim)
        try:
            _, eigen_vectors = np.linalg.eigh(cov_reg)
            self.B_matrix = eigen_vectors
        except np.linalg.LinAlgError:
            self.B_matrix = np.eye(self.dim)

    def evolve(self, epoch):
        curr_pop_size = len(self.pop)
        self.pop_size = curr_pop_size

        if epoch % 5 == 0:
            self._update_eigen_basis()
            
        S_CR, S_F, delta_f = [], [], []
        strat1_wins, strat2_wins = 0, 0
        eigen_wins, standard_wins = 0, 0
        p_count = max(2, int(curr_pop_size * self.p_best_rate))
        
        for i in range(curr_pop_size):
            r = self.generator.integers(0, self.H)
            cr = float(np.clip(self.generator.normal(self.M_CR[r], 0.1), 0.0, 1.0))
            while True:
                # f_val = float(np.random.standard_cauchy() * 0.1 + self.M_F[r])
                f_val = float(self.generator.standard_cauchy() * 0.1 + self.M_F[r])
                if f_val > 0:
                    f_val = min(1.0, f_val)
                    break
                    
            pbest_idx = self.generator.integers(0, p_count)
            x_pbest = self.pop[pbest_idx].solution
            cands_r1 = [idx for idx in range(curr_pop_size) if idx != i]
            r1 = self.generator.choice(cands_r1)
            x_r1 = self.pop[r1].solution
            
            use_strat1 = (self.generator.uniform() < self.p_strat1)
            if use_strat1:
                pool = list(range(curr_pop_size))
                if len(self.archive) > 0:
                    pool += [curr_pop_size + idx for idx in range(len(self.archive))]
                cands_r2 = [idx for idx in pool if idx != i and idx != r1]
                r2 = self.generator.choice(cands_r2)
                x_r2 = self.pop[r2].solution if r2 < curr_pop_size else self.archive[r2 - curr_pop_size]
                v = self.pop[i].solution + f_val * (x_pbest - self.pop[i].solution) + f_val * (x_r1 - x_r2)
            else:
                r2 = self.generator.choice(cands_r1)
                x_r2 = self.pop[r2].solution
                # cauchy_step = np.random.standard_cauchy(self.dim) * 0.05
                cauchy_step = self.generator.standard_cauchy(self.dim) * 0.05
                v = x_pbest + f_val * (x_r1 - x_r2) + cauchy_step * (x_r1 - self.pop[i].solution)
                
            v = self.correct_solution(v)
            
            use_eigen = (self.generator.uniform() < self.p_eigen)
            if use_eigen:
                v_prime = np.dot(self.B_matrix.T, v)
                x_prime = np.dot(self.B_matrix.T, self.pop[i].solution)
                mask = self.generator.uniform(size=self.dim) < cr
                mask[self.generator.integers(0, self.dim)] = True
                u_prime = np.where(mask, v_prime, x_prime)
                u = np.dot(self.B_matrix, u_prime)
            else:
                mask = self.generator.uniform(size=self.dim) < cr
                mask[self.generator.integers(0, self.dim)] = True
                u = np.where(mask, v, self.pop[i].solution)
                
            u = self.correct_solution(u)
            new_agent = self.generate_agent(u)
            
            if self.compare_target(new_agent.target, self.pop[i].target, self.problem.minmax):
                diff = abs(self.pop[i].target.fitness - new_agent.target.fitness)
                delta_f.append(diff)
                S_CR.append(cr)
                S_F.append(f_val)
                if use_strat1: strat1_wins += 1
                else: strat2_wins += 1
                if use_eigen: eigen_wins += 1
                else: standard_wins += 1
                
                if self.max_archive_size > 0:
                    if len(self.archive) < self.max_archive_size:
                        self.archive.append(self.pop[i].solution.copy())
                    else:
                        rep_idx = self.generator.integers(0, self.max_archive_size)
                        self.archive[rep_idx] = self.pop[i].solution.copy()
                self.pop[i] = new_agent
                
        if (strat1_wins + strat2_wins) > 0:
            self.p_strat1 = 0.85 * self.p_strat1 + 0.15 * (strat1_wins / (strat1_wins + strat2_wins))
            self.p_strat1 = np.clip(self.p_strat1, 0.1, 0.9)
            
        if (eigen_wins + standard_wins) > 0:
            self.p_eigen = 0.85 * self.p_eigen + 0.15 * (eigen_wins / (eigen_wins + standard_wins))
            self.p_eigen = np.clip(self.p_eigen, 0.05, 0.95)
            
        if len(S_CR) > 0:
            sum_df = np.sum(delta_f)
            if sum_df > 1e-12:
                w = np.array(delta_f) / sum_df
                self.M_CR[self.k_mem] = np.sum(w * np.array(S_CR))
                self.M_F[self.k_mem] = np.sum(w * (np.array(S_F)**2)) / np.sum(w * np.array(S_F))
                self.k_mem = (self.k_mem + 1) % self.H

        # [修复处]: Mealpy 3.1.0 的 get_sorted_population 返回 (sorted_pop, list_idx) 元组
        if self.use_lpsr:
            next_pop_size = int(np.round(
                self.init_pop_size - (epoch / self.epoch) * (self.init_pop_size - self.min_pop_size)
            ))
            next_pop_size = max(self.min_pop_size, next_pop_size)
            if next_pop_size < len(self.pop):
                sorted_pop, _ = self.get_sorted_population(self.pop, self.problem.minmax)
                self.pop = sorted_pop[:next_pop_size]
                self.pop_size = next_pop_size
                self.max_archive_size = int(self.pop_size * self.arc_rate)
                if len(self.archive) > self.max_archive_size:
                    idx_keep = self.generator.choice(len(self.archive), size=self.max_archive_size, replace=False)
                    self.archive = [self.archive[idx] for idx in idx_keep]

# ==============================================================================
# 2. 实验超参数与 8 大平移旋转测试函数定义
# ==============================================================================
dim = 10 
epoch = 500
pop_size = 50
n_runs = 3

np.random.seed(42)

o1 = np.random.uniform(-30, 30, dim)
o2 = np.random.uniform(-5, 5, dim)
o3 = np.random.uniform(-2.5, 2.5, dim)
o4 = np.random.uniform(-10, 10, dim)
o5 = np.random.uniform(-2.5, 2.5, dim)
o6 = np.random.uniform(-10, 10, dim)
o7 = np.random.uniform(-150, 150, dim)
o8 = np.random.uniform(-2, 2, dim)

H1 = np.random.randn(dim, dim); Q1, _ = np.linalg.qr(H1)
H2 = np.random.randn(dim, dim); Q2, _ = np.linalg.qr(H2)
H3 = np.random.randn(dim, dim); Q3, _ = np.linalg.qr(H3)

def f1_shifted_sphere(x):
    z = x - o1
    return float(np.sum(z**2))

def f2_shifted_rosenbrock(x):
    z = x - o2 + 1.0
    return float(np.sum(100.0 * (z[1:] - z[:-1]**2.0)**2.0 + (1 - z[:-1])**2.0))

def f3_shifted_rastrigin(x):
    z = x - o3
    return float(10.0 * dim + np.sum(z**2.0 - 10.0 * np.cos(2.0 * np.pi * z)))

def f4_shifted_ackley(x):
    z = x - o4
    d = len(z)
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / d)) - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / d) + 20.0 + np.e)

def f5_shifted_rot_rastrigin(x):
    z = np.dot(x - o5, Q1)
    return float(10.0 * dim + np.sum(z**2.0 - 10.0 * np.cos(2.0 * np.pi * z)))

def f6_shifted_rot_ackley(x):
    z = np.dot(x - o6, Q2)
    d = len(z)
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / d)) - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / d) + 20.0 + np.e)

def f7_shifted_griewank(x):
    z = x - o7
    sum_part = np.sum(z**2) / 4000.0
    prod_part = np.prod(np.cos(z / np.sqrt(np.arange(1, dim + 1))))
    return float(sum_part - prod_part + 1.0)

def f8_shifted_rot_hybrid(x):
    z = np.dot(x - o8, Q3) + 1.0
    rosen = 100.0 * (z[1:] - z[:-1]**2.0)**2.0 + (1 - z[:-1])**2.0
    return float(np.sum(rosen**2 / 4000.0 - np.cos(rosen) + 1.0))

BENCHMARKS = [
    {"name": "F1: Shifted Sphere", "func": f1_shifted_sphere, "lb": -100.0, "ub": 100.0},
    {"name": "F2: Shifted Rosenbrock", "func": f2_shifted_rosenbrock, "lb": -30.0, "ub": 30.0},
    {"name": "F3: Shifted Rastrigin", "func": f3_shifted_rastrigin, "lb": -5.12, "ub": 5.12},
    {"name": "F4: Shifted Ackley", "func": f4_shifted_ackley, "lb": -32.768, "ub": 32.768},
    {"name": "F5: Shifted Rot Rastrigin", "func": f5_shifted_rot_rastrigin, "lb": -5.12, "ub": 5.12},
    {"name": "F6: Shifted Rot Ackley", "func": f6_shifted_rot_ackley, "lb": -32.768, "ub": 32.768},
    {"name": "F7: Shifted Griewank", "func": f7_shifted_griewank, "lb": -600.0, "ub": 600.0},
    {"name": "F8: Shifted Rot Hybrid", "func": f8_shifted_rot_hybrid, "lb": -10.0, "ub": 10.0},
]

# ==============================================================================
# 3. 运行全套对比实验 (含 CMA-ES 与 EADS-LPSR)
# ==============================================================================
if __name__ == "__main__":
    min_pop = 10
    iso_init_pop = 2 * pop_size - min_pop  # 自动保证与恒定 pop_size 的总评估次数 (FEs) 完全相等

    algorithms = {
        "PSO (Mealpy)": lambda: PSO.OriginalPSO(epoch=epoch, pop_size=pop_size),
        "DE (Mealpy)": lambda: DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=0.5, cr=0.9),
        "CMA-ES (Mealpy)": lambda: ES.CMA_ES(epoch=epoch, pop_size=pop_size),
        "SHADE (Mealpy)": lambda: SHADE.OriginalSHADE(epoch=epoch, pop_size=pop_size, miu_f=0.5, miu_cr=0.5),
        "L-SHADE (Mealpy)": lambda: SHADE.OriginalL_SHADE(epoch=epoch, pop_size=pop_size, miu_f=0.5, miu_cr=0.5),
        "EADS (Fixed)": lambda: EADS(epoch=epoch, pop_size=pop_size, H=5, arc_rate=1.4, use_lpsr=False),
        "EADS-LPSR (IsoFE)": lambda: EADS(epoch=epoch, pop_size=iso_init_pop, H=5, arc_rate=1.4, use_lpsr=True, min_pop_size=min_pop),
    }

    results_summary = {}
    all_curves = {}

    print("\n" + "="*135)
    print(f"  8 大平移旋转基准实验: 加入 CMA-ES 与 EADS-LPSR 同台竞技 (D = {dim}, Pop = {pop_size}, Runs = {n_runs}, Epoch = {epoch})")
    print("="*135)

    for bm in BENCHMARKS:
        fname = bm["name"]
        print(f"\n>>> 正在评测: {fname}")
        results_summary[fname] = {}
        all_curves[fname] = {}
        
        prob_dict = {
            "obj_func": bm["func"],
            "bounds": FloatVar(lb=[bm["lb"]]*dim, ub=[bm["ub"]]*dim, name="delta"),
            "minmax": "min",
            "log_to": None,
        }

        for alg_name, model_builder in algorithms.items():
            run_fits = []
            run_curves = []
            for r in range(n_runs):
                model = model_builder()
                g_best = model.solve(prob_dict, seed=r*100+7)
                run_fits.append(g_best.target.fitness)
                run_curves.append(model.history.list_global_best_fit)
                
            run_fits = np.array(run_fits)
            mean_curve = np.mean(np.array(run_curves), axis=0)
            results_summary[fname][alg_name] = {"mean": np.mean(run_fits), "std": np.std(run_fits)}
            all_curves[fname][alg_name] = mean_curve
            print(f"  [{alg_name:20s}] 适应度均值: {np.mean(run_fits):.3e} | 标准差: {np.std(run_fits):.3e}")

    # 打印终端汇总大表
    print("\n" + "="*145)
    header = f"{'函数名称':<26} | {'PSO':<10} | {'DE':<10} | {'CMA-ES':<10} | {'SHADE':<10} | {'L-SHADE':<10} | {'EADS(恒定)':<12} | {'EADS-LPSR':<12}"
    print(header)
    print("="*145)
    for bm in BENCHMARKS:
        fn = bm['name']
        p_s   = f"{results_summary[fn]['PSO (Mealpy)']['mean']:.2e}"
        d_s   = f"{results_summary[fn]['DE (Mealpy)']['mean']:.2e}"
        cma_s = f"{results_summary[fn]['CMA-ES (Mealpy)']['mean']:.2e}"
        sh_s  = f"{results_summary[fn]['SHADE (Mealpy)']['mean']:.2e}"
        ls_s  = f"{results_summary[fn]['L-SHADE (Mealpy)']['mean']:.2e}"
        e1_s  = f"{results_summary[fn]['EADS (Fixed)']['mean']:.2e}"
        e2_s  = f"{results_summary[fn]['EADS-LPSR (IsoFE)']['mean']:.2e}"
        print(f"{fn:<26} | {p_s:<10} | {d_s:<10} | {cma_s:<10} | {sh_s:<10} | {ls_s:<10} | {e1_s:<12} | {e2_s:<12}")
    print("="*145)

    # 绘制 2x4 大图
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    colors = {
        "PSO (Mealpy)": "#7f7f7f",
        "DE (Mealpy)": "#1f77b4",
        "CMA-ES (Mealpy)": "#17becf",
        "SHADE (Mealpy)": "#ff7f0e",
        "L-SHADE (Mealpy)": "#9467bd",
        "EADS (Fixed)": "#d62728",
        "EADS-LPSR (IsoFE)": "#2ca02c",
    }
    linestyles = {
        "PSO (Mealpy)": ":",
        "DE (Mealpy)": "--",
        "CMA-ES (Mealpy)": ":",
        "SHADE (Mealpy)": "-.",
        "L-SHADE (Mealpy)": "--",
        "EADS (Fixed)": "-",
        "EADS-LPSR (IsoFE)": "-.",
    }
    x_epochs = np.arange(1, epoch + 1)

    for idx, bm in enumerate(BENCHMARKS):
        ax = axes[idx]
        fname = bm["name"]
        for alg_name in algorithms.keys():
            curve = all_curves[fname][alg_name]
            plot_curve = np.maximum(curve, 1e-16)
            lw = 2.2 if "EADS" in alg_name else 1.4
            ax.plot(x_epochs, plot_curve, label=alg_name, color=colors[alg_name],
                    linestyle=linestyles[alg_name], linewidth=lw)
        ax.set_yscale('log')
        ax.set_title(fname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch (Iterations)', fontsize=9)
        ax.set_ylabel('Best Fitness (Log Scale)', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    output_filename = "mealpy_cmaes_eads_lpsr_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[OK] 包含 CMA-ES 与 EADS-LPSR 的对比收敛图已成功保存为: {output_filename}")
    plt.show()