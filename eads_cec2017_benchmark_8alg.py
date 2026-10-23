"""
================================================================================
IEEE CEC-2017 国际高维基准测试套件 (SCI 论文 8 大黄金对比算法定稿版)
- 算法阵容 (8 大对比算法 + 本文 EADS, 共 9 个算法):
  1. JADE            (DE.JADE, IEEE TEVC 2009 自适应 DE 鼻祖)
  2. SHADE           (SHADE.OriginalSHADE, CEC-2013 顶级历史记忆池 DE)
  3. L-SHADE         (SHADE.DevL_SHADE / OriginalL_SHADE, CEC-2014 冠军)
  4. LSHADE-cnEpSin  (LSHADEcnEpSin.OriginalLSHADEcnEpSin, CEC-2017 前三甲/协方差特征学习)
  5. IMODE           (IMODE.OriginalIMODE, CEC-2020 冠军/多算子混合)
  6. CMA-ES          (ES.CMA_ES, 二阶协方差矩阵自适应理论标杆)
  7. CL-PSO          (PSO.CL_PSO, IEEE TEVC 2006 综合学习群体智能代表)
  8. GSKA            (GSKA.DevGSKA / OriginalGSKA, CEC-2020 亚军 AGSK 基座/人类认知流派)
  9. EADS (本文)     (EADS, 论文定稿 v1_champion 版)

- 核心统计检验 (SCI 顶刊标准规范):
  * 均值与标准差 (Mean ± Std)
  * Friedman 平均排名 (Friedman Average Rank, 越低越好)
  * Wilcoxon 秩和检验 (显著优于 '+', 统计无显著差异 '≈', 劣于 '-')
  * 3x4 (12代表性函数) 或 5x6 (29全量函数) 300 DPI 论文级半对数收敛对比曲线矩阵

- 运行依赖: pip install opfunu mealpy numpy scipy matplotlib
- 算法导入: from eads import EADS (同目录下 eads.py)
================================================================================
"""

import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import mealpy
from mealpy import FloatVar
from mealpy.evolutionary_based import DE, SHADE, ES
from mealpy.swarm_based import PSO
from mealpy.human_based import GSKA
from mealpy.sota_based import IMODE, LSHADEcnEpSin
from opfunu.cec_based import cec2017

# 从独立的 eads.py 模块导入 EADS 算法类
from eads import EADS

print(f"[MEALPY] 成功加载官方优化库, 版本号: {mealpy.__version__}")

# 零误差截断阈值: 保持为机器浮点极限 1e-16
CEC_ZERO_TOL = 1e-16

# ==============================================================================
# 1. 实验参数配置 (支持维度 dim = 30, 50, 100)
# ==============================================================================
dim = 50          # 可选 30, 50, 100 (CEC-2017 官方支持的高维评测基准)
epoch = 800       # 30D 推荐 600; 50D 推荐 800; 100D 推荐 1000
pop_size = 150    # 30D 推荐 100; 50D 推荐 150; 100D 推荐 300 (保证协方差特征分解样本满秩)
n_runs = 3        # 快速验证可设为 3 次; 正式论文投稿实验推荐设为 30 次
np.random.seed(42)

# 测试模式选择:
#   "representative_12" : 精选 12 个涵盖单峰/多峰/混合/复合四大类的核心代表性函数 (绘制 3x4 拼图, 运算高效)
#   "all_29"            : 运行 F12017 ~ F292017 全部 29 个测试函数 (绘制 5x6 全景大拼图)
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
# 2. 构建 SCI 黄金 8 大对比算法 + EADS 模型工厂
# ==============================================================================
def create_lshade(ep, ps):
    """兼容 Mealpy 不同版本对 L-SHADE 类命名的定义"""
    if hasattr(SHADE, "DevL_SHADE"):
        return SHADE.DevL_SHADE(epoch=ep, pop_size=ps)
    elif hasattr(SHADE, "OriginalL_SHADE"):
        return SHADE.OriginalL_SHADE(epoch=ep, pop_size=ps)
    return SHADE.L_SHADE(epoch=ep, pop_size=ps)


def create_gska(ep, ps):
    """兼容 Mealpy GSKA 模块版本"""
    if hasattr(GSKA, "DevGSKA"):
        return GSKA.DevGSKA(epoch=ep, pop_size=ps)
    return GSKA.OriginalGSKA(epoch=ep, pop_size=ps)


def create_cma_es(ep, ps):
    """兼容 Mealpy CMA-ES 模块"""
    if hasattr(ES, "CMA_ES"):
        return ES.CMA_ES(epoch=ep, pop_size=ps)
    return ES.Simple_CMA_ES(epoch=ep, pop_size=ps)


algorithms = {
    # --- 第一梯队：DE 核心演进谱系 ---
    "JADE":            lambda: DE.JADE(epoch=epoch, pop_size=pop_size),
    "SHADE":           lambda: SHADE.OriginalSHADE(epoch=epoch, pop_size=pop_size),
    "L-SHADE":         lambda: create_lshade(epoch, pop_size),

    # --- 第二梯队：CEC 竞赛级 SOTA 顶尖强敌 ---
    "LSHADE-cnEpSin":  lambda: LSHADEcnEpSin.OriginalLSHADEcnEpSin(epoch=epoch, pop_size=pop_size),
    "IMODE":           lambda: IMODE.OriginalIMODE(epoch=epoch, pop_size=pop_size),

    # --- 第三梯队：二阶协方差矩阵自适应标杆 ---
    "CMA-ES":          lambda: create_cma_es(epoch, pop_size),

    # --- 第四梯队：跨流派代表 (群智能与人类认知) ---
    "CL-PSO":          lambda: PSO.CL_PSO(epoch=epoch, pop_size=pop_size),
    "GSKA":            lambda: create_gska(epoch, pop_size),

    # --- 本文提出算法 (定稿冠军版) ---
    "EADS (本文)":     lambda: EADS(epoch=epoch, pop_size=pop_size, H=5, arc_rate=1.0, s2_mode="v1_champion"),
}


# ==============================================================================
# 3. 运行主基准测试循环并采集误差与收敛曲线
# ==============================================================================
if __name__ == "__main__":
    BENCHMARKS = build_cec2017_benchmarks(dim, mode=TEST_MODE)
    n_funcs = len(BENCHMARKS)
    alg_names = list(algorithms.keys())

    results_summary = {}
    all_curves = {}

    print("\n" + "=" * 126)
    print(f"  IEEE CEC-2017 高维标准测试集对比实验 (OPFUNU, 维度 D = {dim}, 函数数 = {n_funcs}, 运行 = {n_runs} 次, Epoch = {epoch})")
    print(f"  [对比阵容] 8 大顶级对比算法 (JADE, SHADE, L-SHADE, LSHADE-cnEpSin, IMODE, CMA-ES, CL-PSO, GSKA) vs. EADS")
    print(f"  [注] 理论最优值精度截断阈值: < {CEC_ZERO_TOL:.0e}")
    print("=" * 126)

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

                # 处理收敛曲线长度对齐 (针对个别具备内部自适应收缩的算法进行插值或截断)
                curve = list(model.history.list_global_best_fit)
                if len(curve) < epoch:
                    curve = curve + [curve[-1]] * (epoch - len(curve))
                elif len(curve) > epoch:
                    curve = curve[:epoch]
                run_curves.append(curve)

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
            print(f"  [{alg_name:16s}] 误差均值: {mean_val:.3e} | 标准差: {std_val:.3e}")

    # ==============================================================================
    # 4. 统计分析：Friedman 平均排名与 Wilcoxon 秩和显著性检验 (+ / ≈ / -)
    # ==============================================================================
    rank_matrix = []
    # 统计 Wilcoxon 检验对其他 8 个算法的优胜情况: [wins, ties, losses]
    eads_key = "EADS (本文)"
    wilcoxon_stats = {alg: [0, 0, 0] for alg in alg_names if alg != eads_key}

    for bm in BENCHMARKS:
        fn = bm["name"]
        means = [results_summary[fn][a]["mean"] for a in alg_names]
        ranks = stats.rankdata(means, method="average")
        rank_matrix.append(ranks)

        # Wilcoxon 秩和统计
        eads_raw = results_summary[fn][eads_key]["raw"]
        for alg in wilcoxon_stats.keys():
            comp_raw = results_summary[fn][alg]["raw"]
            if np.allclose(eads_raw, comp_raw, atol=1e-15):
                wilcoxon_stats[alg][1] += 1  # ≈
            else:
                try:
                    _, p_val = stats.ranksums(eads_raw, comp_raw)
                    if p_val < 0.05:
                        if np.mean(eads_raw) < np.mean(comp_raw):
                            wilcoxon_stats[alg][0] += 1  # +
                        else:
                            wilcoxon_stats[alg][2] += 1  # -
                    else:
                        wilcoxon_stats[alg][1] += 1  # ≈
                except Exception:
                    # 备用容错逻辑
                    if np.mean(eads_raw) < np.mean(comp_raw):
                        wilcoxon_stats[alg][0] += 1
                    else:
                        wilcoxon_stats[alg][2] += 1

    avg_ranks = np.mean(np.array(rank_matrix), axis=0)

    # ==============================================================================
    # 5. 打印 SCI 顶刊标准多算法大对比表 (Formatted Result Table)
    # ==============================================================================
    col_w = 11
    header = f"{'CEC-2017 函数':<32} | " + " | ".join([f"{a[:col_w]:^{col_w}}" for a in alg_names])
    div_line = "=" * len(header)

    print("\n" + div_line)
    print("  IEEE CEC-2017 高维 (D=50) 8 大黄金对比算法综合实验结果汇总表")
    print(div_line)
    print(header)
    print(div_line)

    for bm in BENCHMARKS:
        fn = bm["name"]
        row_str = f"{fn[:32]:<32} | "
        cell_strs = []
        for a in alg_names:
            m_val = results_summary[fn][a]["mean"]
            cell_strs.append(f"{m_val:^{col_w}.2e}")
        row_str += " | ".join(cell_strs)
        print(row_str)

    print(div_line)
    # Friedman 排名行
    rank_cells = [f"{avg_ranks[i]:^{col_w}.2f}" for i in range(len(alg_names))]
    print(f"{'Friedman 平均排名 (越低越好)':<32} | " + " | ".join(rank_cells))
    print(div_line)

    # Wilcoxon 汇总行
    wilcoxon_cells = []
    for a in alg_names:
        if a == eads_key:
            wilcoxon_cells.append(f"{'--':^{col_w}}")
        else:
            w, t, l = wilcoxon_stats[a]
            wilcoxon_cells.append(f"{f'{w}/{t}/{l}':^{col_w}}")
    print(f"{'Wilcoxon 检验 (+/≈/-)':<32} | " + " | ".join(wilcoxon_cells))
    print(div_line)

    # ==============================================================================
    # 6. 绘制高分辨率论文收敛曲线对比矩阵图 (300 DPI Publication Figure)
    # ==============================================================================
    if n_funcs <= 12:
        n_rows, n_cols = 3, 4
        figsize = (24, 15)
    else:
        n_rows, n_cols = 5, 6
        figsize = (28, 22)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # 论文专属高对比度色板配置
    colors = {
        "JADE":           "#1f77b4",  # 沉稳蓝
        "SHADE":          "#ff7f0e",  # 琥珀橙
        "L-SHADE":        "#2ca02c",  # 森林绿
        "LSHADE-cnEpSin": "#9467bd",  # 典雅紫
        "IMODE":          "#8c564b",  # 砖红褐
        "CMA-ES":         "#e377c2",  # 玫粉红
        "CL-PSO":         "#7f7f7f",  # 中灰色
        "GSKA":           "#bcbd22",  # 橄榄黄
        "EADS (本文)":    "#d62728",  # 鲜艳红 (核心突出)
    }

    linestyles = {
        "JADE":           "--",
        "SHADE":          "-.",
        "L-SHADE":        ":",
        "LSHADE-cnEpSin": "--",
        "IMODE":          "-.",
        "CMA-ES":         ":",
        "CL-PSO":         "-.",
        "GSKA":           "--",
        "EADS (本文)":    "-",   # 实线加粗
    }

    x_epochs = np.arange(1, epoch + 1)

    for idx, bm in enumerate(BENCHMARKS):
        ax = axes[idx]
        fname = bm["name"]
        for alg_name in alg_names:
            curve = all_curves[fname][alg_name]
            plot_curve = np.maximum(curve, 1e-16)
            is_eads = (alg_name == eads_key)
            ax.plot(
                x_epochs,
                plot_curve,
                label=alg_name,
                color=colors[alg_name],
                linestyle=linestyles[alg_name],
                linewidth=2.4 if is_eads else 1.2,
                alpha=1.0 if is_eads else 0.8,
                zorder=10 if is_eads else 3,
            )
        ax.set_yscale("log")
        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Iterations (Epochs)", fontsize=8.5)
        ax.set_ylabel(r"Error $\Delta f$ (Log Scale)", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=6.8, loc="upper right", framealpha=0.85)

    for idx in range(n_funcs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_filename = f"mealpy_cec2017_8benchmarks_dim{dim}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[OK] IEEE CEC-2017 (D={dim}, 8大黄金对比算法) 收敛对比大图已成功保存为: {output_filename}")
    plt.show()