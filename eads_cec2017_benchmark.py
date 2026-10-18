"""
================================================================================
基于 OPFUNU (IEEE CEC-2017) 与 MEALPY 官方库的高维标准基准测试套件
- 算法模块导入: from eads import EADS (需与 eads.py 放在同一目录下)
- 测试函数集: IEEE CEC-2017 (支持 12 代表性函数模式 与 29 全量函数模式切换)
  * 单峰函数 (Unimodal): F12017 ~ F22017
  * 简单多峰函数 (Multimodal): F32017 ~ F92017
  * 混合函数 (Hybrid): F102017 ~ F192017
  * 复合函数 (Composition): F202017 ~ F292017
- 支持高维配置: dim = 30, 50, 100 (CEC-2017 官方标准高维支持)
- 评价指标: 适应度误差 Δf(x) = f(x) - f_global (保留 float64 极限精度 1e-16)
- 运行依赖: pip install opfunu mealpy numpy scipy matplotlib
================================================================================
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import mealpy
from mealpy import FloatVar
from mealpy.swarm_based import PSO
from mealpy.evolutionary_based import DE, SHADE
from opfunu.cec_based import cec2017

# 从独立的 eads.py 模块导入 EADS 算法类
from eads import EADS

print(f"[MEALPY] 成功加载官方优化库, 版本号: {mealpy.__version__}")

# 零误差截断阈值: 保持为机器浮点极限 1e-16
CEC_ZERO_TOL = 1e-16


# ==============================================================================
# 1. 配置 IEEE CEC-2017 高维标准测试函数 (支持 dim = 30, 50, 100)
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
# 2. 运行 CEC-2017 高维对比实验并绘制收敛曲线矩阵图
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