"""
================================================================================
基于 OPFUNU (IEEE CEC-2022) 与 MEALPY 官方库的 12 函数标准基准测试套件
- 测试函数集: IEEE CEC-2022 (F12022 ~ F122022, 共 12 个旋转/混合/复合函数)
  * 单峰函数 (Unimodal): F1 (Zakharov)
  * 基础多峰函数 (Multimodal): F2 (Rosenbrock), F3 (Schaffer F7), F4 (Non-Cont. Rastrigin), F5 (Levy)
  * 混合函数 (Hybrid): F6 (Hybrid 1), F7 (Hybrid 2), F8 (Hybrid 3)
  * 复合函数 (Composition): F9 (Comp 1), F10 (Comp 2), F11 (Comp 3), F12 (Comp 4)
- 支持维度: dim = 10 或 dim = 20 (CEC-2022 官方全套支持维度)
- 评价指标: 适应度误差 Δf(x) = f(x) - f_global (理论最优误差为 0.00e+00)
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

from EADS import EADS

# IEEE CEC-2022 官方零误差截断阈值 (最终汇总表与排名计算中 < 1e-8 视为精确归零)
CEC_ZERO_TOL = 1e-16

# ==============================================================================
# 2. 配置 IEEE CEC-2022 12 个标准测试函数 (支持 dim = 10 或 dim = 20)
# ==============================================================================
dim = 20          # 可选 10 或 20 (CEC-2022 标准支持维度)
epoch = 600      # 10D 推荐 500; 20D 推荐 600~1000
pop_size = 80     # 10D 推荐 50;  20D 推荐 80~100
n_runs = 3        # 快速验证设为 3 次; 正式论文统计检验可设为 30 次

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

        # 构造闭包以计算适应度误差 Δf(x) = max(0, f(x) - f_global)
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
                model = model_builder()
                g_best = model.solve(prob_dict, seed=r * 100 + 7)

                run_fits.append(g_best.target.fitness)
                run_curves.append(model.history.list_global_best_fit)

            run_fits = np.array(run_fits)
            mean_curve = np.mean(np.array(run_curves), axis=0)
            results_summary[fname][alg_name] = {
                "mean": float(np.mean(run_fits)),
                "std": float(np.std(run_fits)),
                "raw": run_fits,
            }
            all_curves[fname][alg_name] = mean_curve
            print(f"  [{alg_name:18s}] 误差均值: {np.mean(run_fits):.3e} | 标准差: {np.std(run_fits):.3e}")

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
    output_filename = f"mealpy_opfunu_cec2022_dim{dim}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[OK] IEEE CEC-2022 (D={dim}) 12函数收敛对比大图已成功保存为: {output_filename}")
    plt.show()