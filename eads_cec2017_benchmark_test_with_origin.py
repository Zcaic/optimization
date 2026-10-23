"""
================================================================================
IEEE CEC-2017 多核并行基准测试套件 (支持【JADE/SHADE/CoBiDE/EADS 四强极速对决】与【9大对比算法全量评测】)
- 核心架构设计:
  1. 多模式一键切换 (TEST_MODE):
     - "fast_jade_shade_12" : 【极速四强对决模式 (默认激活)】评测 JADE, SHADE, CoBiDE, EADS (本文) 4 个算法
                              × 12 个代表性函数，剔除耗时的 CMA-ES/IMODE 等算子，整体速度提升 4~5 倍，
                              专用于快速验证和调优 EADS 相对于 JADE、SHADE 与 CoBiDE 的胜负表现。
     - "fast_jade_shade_29" : 【四强全函数模式】评测 JADE, SHADE, CoBiDE, EADS (本文) × 全部 29 个函数。
     - "representative_12"  : 【标准 10 算法模式】9 大对比算法 (含 CoBiDE) + EADS × 12 个代表性函数。
     - "all_29"             : 【论文定稿全量模式】9 大对比算法 (含 CoBiDE) + EADS × 全部 29 个函数。
  2. 多进程函数级并行 (ProcessPoolExecutor):
     同时在多个 CPU 核心上并行评测 12 个代表性函数 (或 29 个全量函数)，
     单进程内部自动锁定 BLAS/OpenMP 线程数为 1，杜绝多核线程争抢。
  3. 全量单次运行数据持久化落盘 (Per-Function JSON + Origin 专用 CSV):
     每个测试函数评测完毕后，立即将该函数下所有待测算法的完整数据保存至结果目录:
     `./cec2017_results_dim{dim}_{TEST_MODE}/`
     * JSON 文件现已完整保存每次独立运行的收敛曲线矩阵 `run_curves` (N_RUNS × EPOCH)、
       每次运行的详细记录 `runs_detail` (含 run_id, seed, final_error, curve)，以及预计算好的
       上下可信度边界曲线 (`mean_curve`, `std_curve`, `sem_curve`, `ci95_lower_curve`,
       `ci95_upper_curve`, `median_curve`, `q25_curve`, `q75_curve`, `min_curve`, `max_curve`)！
     * 同时在 `./cec2017_results_dim{dim}_{TEST_MODE}/origin_csv/` 子目录下自动生成可直接拖入
       Origin 绘图的宽表 CSV 文件 (包含每次运行曲线列与 Mean/Std/SEM/95%CI/四分位上下界列)。
  4. 严格对齐 IEEE CEC-2017 官方零误差标准与统计检验修正:
     - 遵循 IEEE CEC-2017 官方技术报告 (Awad et al., 2016)，误差 Δf < 1e-8 统一视为 0 (0.00e+00)。
     - 汇总表内置【理论最优 f* (Δf*)】列，直观展示每个函数的官方全局最优偏置值 f* 与零误差基准 Δf*=0。
================================================================================
"""

import os
import re
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 锁定单进程内部底层矩阵运算线程数为 1，防止 12 进程并行时发生 CPU 线程过载争抢
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 配置 Matplotlib 字体回退链与负号显示，兼容 Windows/macOS/Linux 并彻底消除中文字形缺失警告
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

import mealpy
from mealpy import FloatVar
from mealpy.evolutionary_based import DE, SHADE, ES
from mealpy.swarm_based import PSO
from mealpy.human_based import GSKA
from mealpy.sota_based import IMODE, LSHADEcnEpSin
from opfunu.cec_based import cec2017

# 从独立的 eads.py 模块同时导入 EADS 与重构后的 CoBiDE 算法类
from eads import EADS, CoBiDE


# ==============================================================================
# 1. 全局实验参数与测试模式配置
# ==============================================================================
DIM = 50                     # 可选 30, 50, 100 (CEC-2017 官方支持高维配置)
EPOCH = 1000                 # 30D 推荐 600; 50D 推荐 800~1000; 100D 推荐 1000
POP_SIZE = 150               # 30D 推荐 100; 50D 推荐 150; 100D 推荐 300
N_RUNS = 5                   # 快速验证设为 3~5 次; 正式论文统计检验推荐设为 30 次
CEC_ZERO_TOL = 1e-8          # IEEE CEC-2017 官方标准零误差截断阈值 (Δf < 1e-8 记为 0.00e+00)

# ------------------------------------------------------------------------------
# 测试模式切换 (TEST_MODE):
#   - "fast_jade_shade_12" : 极速四强模式 (评测 JADE, SHADE, CoBiDE, EADS × 12个代表性函数, 默认激活)
#   - "fast_jade_shade_29" : 四强全量函数模式 (评测 JADE, SHADE, CoBiDE, EADS × 全部29个函数)
#   - "representative_12"  : 9大对比算法 (含 CoBiDE) + EADS (共10个算法) × 12个代表性函数
#   - "all_29"             : 9大对比算法 (含 CoBiDE) + EADS (共10个算法) × 全部29个函数
# ------------------------------------------------------------------------------
TEST_MODE = "fast_jade_shade_12"

# 并行进程数: 默认同时启动最多 12 个工作进程 (自动适配本机 CPU 核心上限)
MAX_WORKERS = min(12, os.cpu_count() or 4)

# 结果文件保存目录 (根据 TEST_MODE 自动隔离目录，避免不同模式结果互相覆盖)
RESULTS_DIR = f"cec2017_results_dim{DIM}_{TEST_MODE}"

# 断点续跑开关:
#   True  -> 如果目录下已有某个函数的结果文件且包含当前模式的全部算法及完整 run_curves，直接读取跳过
#   False -> 每次运行强制重新评测所有函数并覆盖旧文件
SKIP_EXISTING_FILES = False

EADS_KEY = "EADS (本文)"

# 极速 4 算法阵容 (JADE + SHADE + CoBiDE + 本文 EADS，用于 fast_jade_shade_12 与 fast_jade_shade_29)
FAST_3_ALG_NAMES = [
    "JADE",
    "SHADE",
    "CoBiDE",
    EADS_KEY,
]

# 完整 10 算法阵容 (9 大黄金对比算法 [含 CoBiDE] + 本文 EADS，用于 representative_12 与 all_29)
FULL_9_ALG_NAMES = [
    "JADE",
    "SHADE",
    "CoBiDE",
    "L-SHADE",
    "LSHADE-cnEpSin",
    "IMODE",
    "CMA-ES",
    "CL-PSO",
    "GSKA",
    EADS_KEY,
]

REPRESENTATIVE_12_SPECS = [
    ("F1: Rotated Bent Cigar (Unimodal)", "F12017"),
    ("F2: Rotated Zakharov (Unimodal)", "F22017"),
    ("F3: Rotated Rosenbrock (Multimodal)", "F32017"),
    ("F4: Rotated Rastrigin (Multimodal)", "F42017"),
    ("F7: Rotated Non-Cont. Rastrigin", "F72017"),
    ("F8: Rotated Levy (Multimodal)", "F82017"),
    ("F10: Hybrid Function 1", "F102017"),
    ("F13: Hybrid Function 4", "F132017"),
    ("F18: Hybrid Function 9", "F182017"),
    ("F20: Composition Function 1", "F202017"),
    ("F22: Composition Function 3", "F222017"),
    ("F26: Composition Function 7", "F262017"),
]


def get_active_alg_names(mode):
    """根据测试模式返回当前激活的对比算法列表 (4 种模式均已包含 CoBiDE)"""
    if mode in ("fast_jade_shade_12", "fast_jade_shade_29", "fast_3"):
        return list(FAST_3_ALG_NAMES)
    return list(FULL_9_ALG_NAMES)


def get_benchmark_specs(ndim, mode="fast_jade_shade_12"):
    """获取可跨进程序列化的测试函数规格列表 [(short_title, cls_name), ...]"""
    if mode in ("all_29", "fast_jade_shade_29"):
        specs = []
        for idx in range(1, 30):
            cls_name = f"F{idx}2017"
            cls = getattr(cec2017, cls_name)
            tmp_obj = cls(ndim=ndim)
            raw_name = tmp_obj.name.split("(")[0].strip()
            short_title = raw_name[:34]
            specs.append((short_title, cls_name))
        return specs
    return list(REPRESENTATIVE_12_SPECS)


def make_safe_filename(idx, func_title):
    """生成规范的、按序号排列的单函数结果文件名，例如: 01_F1_Rotated_Bent_Cigar_Unimodal.json"""
    clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", func_title).strip("_")
    return f"{idx:02d}_{clean_name}.json"


def export_origin_csv_for_function(idx, func_title, epoch, n_runs, active_alg_names, func_results, results_dir):
    """
    将单个测试函数下所有算法的【每次独立运行收敛曲线 + 均值/标准差/标准误/95%置信上下界/四分位上下界】
    导出为 Origin 专用宽表 CSV 文件，方便直接拖入 Origin 绘制带阴影误差带的收敛曲线图。
    """
    origin_dir = os.path.join(results_dir, "origin_csv")
    os.makedirs(origin_dir, exist_ok=True)
    clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", func_title).strip("_")
    csv_path = os.path.join(origin_dir, f"{idx:02d}_{clean_name}_origin_curves.csv")

    header = ["Epoch"]
    for alg_name in active_alg_names:
        safe_alg = "EADS" if alg_name == EADS_KEY else alg_name
        header.extend([
            f"{safe_alg}_Mean",
            f"{safe_alg}_Std",
            f"{safe_alg}_SEM",
            f"{safe_alg}_CI95_Lower",
            f"{safe_alg}_CI95_Upper",
            f"{safe_alg}_Median",
            f"{safe_alg}_Q25",
            f"{safe_alg}_Q75",
            f"{safe_alg}_Min",
            f"{safe_alg}_Max",
        ])
        for r in range(n_runs):
            header.append(f"{safe_alg}_Run{r + 1}")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ep in range(epoch):
            row = [ep + 1]
            for alg_name in active_alg_names:
                ad = func_results[alg_name]
                row.extend([
                    ad["mean_curve"][ep],
                    ad["std_curve"][ep],
                    ad["sem_curve"][ep],
                    ad["ci95_lower_curve"][ep],
                    ad["ci95_upper_curve"][ep],
                    ad["median_curve"][ep],
                    ad["q25_curve"][ep],
                    ad["q75_curve"][ep],
                    ad["min_curve"][ep],
                    ad["max_curve"][ep],
                ])
                for r in range(n_runs):
                    row.append(ad["run_curves"][r][ep])
            writer.writerow(row)

    return csv_path


# ==============================================================================
# 2. 模型构建器 (在每个独立工作进程内部实例化待测算法)
# ==============================================================================
def build_optimizer(alg_name, epoch, pop_size):
    """根据算法名称构建对应的 Mealpy / CoBiDE / EADS 优化器实例"""
    if alg_name == "JADE":
        return DE.JADE(epoch=epoch, pop_size=pop_size)
    elif alg_name == "SHADE":
        return SHADE.OriginalSHADE(epoch=epoch, pop_size=pop_size)
    elif alg_name == "CoBiDE":
        return CoBiDE(epoch=epoch, pop_size=pop_size)
    elif alg_name == "L-SHADE":
        if hasattr(SHADE, "DevL_SHADE"):
            return SHADE.DevL_SHADE(epoch=epoch, pop_size=pop_size)
        elif hasattr(SHADE, "OriginalL_SHADE"):
            return SHADE.OriginalL_SHADE(epoch=epoch, pop_size=pop_size)
        return SHADE.L_SHADE(epoch=epoch, pop_size=pop_size)
    elif alg_name == "LSHADE-cnEpSin":
        return LSHADEcnEpSin.OriginalLSHADEcnEpSin(epoch=epoch, pop_size=pop_size)
    elif alg_name == "IMODE":
        return IMODE.OriginalIMODE(epoch=epoch, pop_size=pop_size)
    elif alg_name == "CMA-ES":
        if hasattr(ES, "CMA_ES"):
            return ES.CMA_ES(epoch=epoch, pop_size=pop_size)
        return ES.Simple_CMA_ES(epoch=epoch, pop_size=pop_size)
    elif alg_name == "CL-PSO":
        return PSO.CL_PSO(epoch=epoch, pop_size=pop_size)
    elif alg_name == "GSKA":
        if hasattr(GSKA, "DevGSKA"):
            return GSKA.DevGSKA(epoch=epoch, pop_size=pop_size)
        return GSKA.OriginalGSKA(epoch=epoch, pop_size=pop_size)
    elif alg_name == EADS_KEY:
        return EADS(epoch=epoch, pop_size=pop_size, H=4, arc_rate=1.0, s2_mode="v14_champion")
    else:
        raise ValueError(f"未知的算法名称: {alg_name}")


# ==============================================================================
# 3. 顶层多核并行工作函数 (兼容 Windows spawn 与 Linux/macOS fork)
# ==============================================================================
def evaluate_single_function_worker(task_args):
    """
    单个测试函数的完整评测任务 (在独立 CPU 核心进程中运行):
    评测该函数下的全部激活算法 (active_alg_names)，并将每次独立运行的完整曲线及统计边界
    立即保存为独立的 JSON 文件与 Origin CSV 文件。
    """
    (
        idx,
        func_title,
        cls_name,
        ndim,
        epoch,
        pop_size,
        n_runs,
        results_dir,
        skip_existing,
        active_alg_names,
    ) = task_args

    os.makedirs(results_dir, exist_ok=True)
    file_name = make_safe_filename(idx, func_title)
    file_path = os.path.join(results_dir, file_name)

    # 若开启断点续跑且结果文件已存在、包含所需全部算法且已存有每次运行的 run_curves，直接读取返回
    if skip_existing and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cached_algs = cached_data.get("algorithms", {})
            if (
                all(a in cached_algs and "run_curves" in cached_algs[a] for a in active_alg_names)
                and cached_data.get("n_runs") == n_runs
            ):
                return idx, func_title, file_path, cached_data, 0.0, True
        except Exception:
            pass  # 若旧文件损坏则重新计算

    t_start = time.time()

    # 在当前子进程中实例化 CEC-2017 函数对象
    cec_cls = getattr(cec2017, cls_name)
    cec_obj = cec_cls(ndim=ndim)
    f_global = float(cec_obj.f_global)
    lb_vec = cec_obj.lb.tolist()
    ub_vec = cec_obj.ub.tolist()

    def error_func(x):
        err = float(cec_obj.evaluate(x) - f_global)
        return 0.0 if err < 1e-16 else err

    prob_dict = {
        "obj_func": error_func,
        "bounds": FloatVar(lb=lb_vec, ub=ub_vec, name="delta"),
        "minmax": "min",
        "log_to": None,  # 关闭子进程内部日志，防止多核并发时终端进度条互相覆盖乱码
    }

    func_results = {}

    for alg_name in active_alg_names:
        run_fits = []
        run_seeds = []
        run_curves = []
        runs_detail = []

        for r in range(n_runs):
            seed_val = r * 100 + 7
            run_seeds.append(int(seed_val))
            np.random.seed(seed_val)
            model = build_optimizer(alg_name, epoch, pop_size)
            g_best = model.solve(prob_dict, seed=seed_val)

            # 遵循 IEEE CEC-2017 官方规则: 最终误差 < 1e-8 视为 0.0
            final_err = 0.0 if g_best.target.fitness < CEC_ZERO_TOL else float(g_best.target.fitness)
            run_fits.append(final_err)

            # 对齐单次运行收敛曲线长度至 epoch
            curve = [float(v) for v in model.history.list_global_best_fit]
            if len(curve) == 0:
                curve = [final_err] * epoch
            elif len(curve) < epoch:
                curve = curve + [curve[-1]] * (epoch - len(curve))
            elif len(curve) > epoch:
                curve = curve[:epoch]

            run_curves.append(curve)
            runs_detail.append({
                "run_id": r + 1,
                "seed": int(seed_val),
                "final_error": final_err,
                "curve": curve,
            })

        # 转换为二维矩阵: shape = (n_runs, epoch)
        run_fits_arr = np.array(run_fits, dtype=float)
        curves_mat = np.array(run_curves, dtype=float)

        # 计算每代均值、标准差、标准误 (SEM)、95% 置信区间上下界、中位数与 25%/75% 四分位上下界
        mean_curve_arr = np.mean(curves_mat, axis=0)
        std_curve_arr = np.std(curves_mat, axis=0)
        sem_curve_arr = std_curve_arr / np.sqrt(max(1, n_runs))
        # 对数坐标安全下界: 避免 Mean - Std 或 Mean - 1.96*SEM 出现负数导致 Origin 对数轴报错
        ci95_lower_arr = np.maximum(mean_curve_arr - 1.96 * sem_curve_arr, 1e-16)
        ci95_upper_arr = mean_curve_arr + 1.96 * sem_curve_arr
        median_curve_arr = np.median(curves_mat, axis=0)
        q25_curve_arr = np.percentile(curves_mat, 25, axis=0)
        q75_curve_arr = np.percentile(curves_mat, 75, axis=0)
        min_curve_arr = np.min(curves_mat, axis=0)
        max_curve_arr = np.max(curves_mat, axis=0)

        mean_val = 0.0 if np.mean(run_fits_arr) < CEC_ZERO_TOL else float(np.mean(run_fits_arr))
        std_val = 0.0 if mean_val == 0.0 else float(np.std(run_fits_arr))

        func_results[alg_name] = {
            "mean": mean_val,
            "std": std_val,
            "raw": run_fits_arr.tolist(),
            "run_seeds": run_seeds,
            "run_curves": curves_mat.tolist(),          # 维度 [n_runs, epoch]: 每次运行的完整收敛曲线
            "runs_detail": runs_detail,                 # 含每次运行的 run_id, seed, final_error, curve
            "mean_curve": mean_curve_arr.tolist(),      # 均值曲线
            "std_curve": std_curve_arr.tolist(),        # 标准差曲线 (Mean ± Std 作图用)
            "sem_curve": sem_curve_arr.tolist(),        # 标准误曲线 (Mean ± SEM 作图用)
            "ci95_lower_curve": ci95_lower_arr.tolist(),# 95% 置信区间下界 (已做 >=1e-16 截断适配对数坐标)
            "ci95_upper_curve": ci95_upper_arr.tolist(),# 95% 置信区间上界
            "median_curve": median_curve_arr.tolist(),  # 中位数曲线
            "q25_curve": q25_curve_arr.tolist(),        # 25% 下四分位曲线 (IQR 下界，天然正数适配对数坐标)
            "q75_curve": q75_curve_arr.tolist(),        # 75% 上四分位曲线 (IQR 上界)
            "min_curve": min_curve_arr.tolist(),        # 历次运行最小值包络线
            "max_curve": max_curve_arr.tolist(),        # 历次运行最大值包络线
        }

    elapsed = time.time() - t_start

    # 同步导出方便 Origin 直接拖拽作图的宽表 CSV 文件
    origin_csv_path = export_origin_csv_for_function(
        idx, func_title, epoch, n_runs, active_alg_names, func_results, results_dir
    )

    # 组装该测试函数的完整数据包并写入独立 JSON 文件
    output_payload = {
        "index": idx,
        "func_title": func_title,
        "cls_name": cls_name,
        "f_global": f_global,
        "dim": ndim,
        "epoch": epoch,
        "pop_size": pop_size,
        "n_runs": n_runs,
        "elapsed_seconds": round(elapsed, 2),
        "origin_csv_file": os.path.basename(origin_csv_path),
        "algorithms": func_results,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    return idx, func_title, file_path, output_payload, elapsed, False


# ==============================================================================
# 4. 从结果目录加载全部测试文件并进行统计检验与绘图
# ==============================================================================
def load_all_results_from_dir(results_dir, expected_specs, active_alg_names):
    """
    扫描并加载结果目录下的全部测试函数结果文件，校验完整性后返回汇总字典、曲线字典与理论最优值字典
    """
    results_summary = {}
    all_curves = {}
    f_globals = {}
    loaded_files = []

    for idx, (func_title, cls_name) in enumerate(expected_specs, start=1):
        fname = make_safe_filename(idx, func_title)
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"未找到函数 [{func_title}] 的结果文件: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        loaded_files.append(fpath)
        results_summary[func_title] = {}
        all_curves[func_title] = {}

        # 读取该函数在 CEC-2017 中的官方理论全局最优值 f*
        if "f_global" in payload:
            f_globals[func_title] = float(payload["f_global"])
        else:
            cec_cls = getattr(cec2017, cls_name)
            f_globals[func_title] = float(cec_cls(ndim=DIM).f_global)

        for alg_name in active_alg_names:
            alg_data = payload["algorithms"][alg_name]
            raw_arr = np.array(alg_data["raw"], dtype=float)
            raw_arr = np.where(raw_arr < CEC_ZERO_TOL, 0.0, raw_arr)
            mean_val = 0.0 if np.mean(raw_arr) < CEC_ZERO_TOL else float(np.mean(raw_arr))
            std_val = 0.0 if mean_val == 0.0 else float(np.std(raw_arr))

            mean_curve = np.array(alg_data["mean_curve"], dtype=float)
            # 兼容读取单次运行曲线矩阵与上下边界曲线
            run_curves = (
                np.array(alg_data["run_curves"], dtype=float)
                if "run_curves" in alg_data
                else np.expand_dims(mean_curve, axis=0)
            )
            sem_curve = (
                np.array(alg_data["sem_curve"], dtype=float)
                if "sem_curve" in alg_data
                else np.std(run_curves, axis=0) / np.sqrt(max(1, len(run_curves)))
            )

            results_summary[func_title][alg_name] = {
                "mean": mean_val,
                "std": std_val,
                "raw": raw_arr,
                "run_curves": run_curves,
                "sem_curve": sem_curve,
            }
            all_curves[func_title][alg_name] = mean_curve

    return results_summary, all_curves, f_globals, loaded_files


def analyze_and_plot(
    expected_specs,
    results_summary,
    all_curves,
    f_globals,
    results_dir,
    ndim,
    epoch,
    mode,
    active_alg_names,
):
    """
    基于加载的结果文件，计算 Friedman 排名、均值胜平负、Wilcoxon 检验并绘制带可信度阴影带的 300 DPI 收敛曲线图
    """
    n_funcs = len(expected_specs)
    rank_matrix = []
    wilcoxon_stats = {alg: [0, 0, 0] for alg in active_alg_names if alg != EADS_KEY}
    mean_wtl_stats = {alg: [0, 0, 0] for alg in active_alg_names if alg != EADS_KEY}

    for func_title, _ in expected_specs:
        # 对有效数字保留 5 位后计算排名，仅当两算法误差差距 < 0.001% (如同时收敛至 0.00 或 200.000) 时判为并列
        means = [
            0.0 if results_summary[func_title][a]["mean"] < CEC_ZERO_TOL
            else float(f"{results_summary[func_title][a]['mean']:.5e}")
            for a in active_alg_names
        ]
        ranks = stats.rankdata(means, method="average")
        rank_matrix.append(ranks)

        eads_raw = results_summary[func_title][EADS_KEY]["raw"]
        eads_mean = results_summary[func_title][EADS_KEY]["mean"]
        n_samples = len(eads_raw)

        # 小样本校正: 当 N_RUNS <= 5 时采用 α_eff = 0.10，N_RUNS >= 10 时严格采用 α = 0.05
        alpha_thresh = 0.10 if n_samples <= 5 else 0.05

        for alg in wilcoxon_stats.keys():
            comp_raw = results_summary[func_title][alg]["raw"]
            comp_mean = results_summary[func_title][alg]["mean"]

            # 1. 均值直接胜平负统计 (相对容差 rtol=1e-2)
            if np.isclose(eads_mean, comp_mean, rtol=1e-2, atol=CEC_ZERO_TOL):
                mean_wtl_stats[alg][1] += 1
            elif eads_mean < comp_mean:
                mean_wtl_stats[alg][0] += 1
            else:
                mean_wtl_stats[alg][2] += 1

            # 2. Wilcoxon 秩和检验 (+/≈/-)
            if np.allclose(eads_raw, comp_raw, rtol=1e-4, atol=CEC_ZERO_TOL):
                wilcoxon_stats[alg][1] += 1  # ≈
            else:
                try:
                    _, p_val = stats.ranksums(eads_raw, comp_raw)
                    if p_val < alpha_thresh:
                        if eads_mean < comp_mean:
                            wilcoxon_stats[alg][0] += 1  # + (EADS 显著更优)
                        else:
                            wilcoxon_stats[alg][2] += 1  # - (EADS 显著更差)
                    else:
                        wilcoxon_stats[alg][1] += 1  # ≈ (统计无显著差异)
                except Exception:
                    if eads_mean < comp_mean:
                        wilcoxon_stats[alg][0] += 1
                    else:
                        wilcoxon_stats[alg][2] += 1

    avg_ranks = np.mean(np.array(rank_matrix), axis=0)

    # 针对 4 算法极速模式使用宽列宽，同时显示 Mean ± Std；10 算法模式显示紧凑 Mean
    is_fast_mode = (len(active_alg_names) <= 4)
    col_w = 22 if is_fast_mode else 11
    opt_col_w = 16
    opt_col_title = "理论最优 f*(Δf*)"

    header = (
        f"{'CEC-2017 函数':<34} | "
        f"{opt_col_title:^{opt_col_w}} | "
        + " | ".join([f"{a[:col_w]:^{col_w}}" for a in active_alg_names])
    )
    div_line = "=" * len(header)

    suite_title = (
        "JADE vs SHADE vs CoBiDE vs EADS 四强极速对决汇总表"
        if is_fast_mode
        else "9 大对比算法 (含 CoBiDE) 综合实验结果汇总表"
    )

    print("\n" + div_line)
    print(f"  IEEE CEC-2017 高维 (D={ndim}) {suite_title} (零误差阈值 < {CEC_ZERO_TOL:.0e})")
    print(f"  [说明] 表中算法列数值为适应度误差 Δf = f(x) - f* (理论最优误差 Δf* = 0.00e+00，越接近 0 越优)")
    print(div_line)
    print(header)
    print(div_line)

    table_rows_for_file = []
    for func_title, _ in expected_specs:
        fg_val = f_globals.get(func_title, 0.0)
        opt_str = f"{fg_val:.0f} (0.00)".center(opt_col_w)
        row_str = f"{func_title[:34]:<34} | {opt_str} | "
        if is_fast_mode:
            cell_strs = [
                f"{results_summary[func_title][a]['mean']:.2e} ± {results_summary[func_title][a]['std']:.1e}".center(col_w)
                for a in active_alg_names
            ]
        else:
            cell_strs = [f"{results_summary[func_title][a]['mean']:^{col_w}.2e}" for a in active_alg_names]
        row_str += " | ".join(cell_strs)
        print(row_str)
        table_rows_for_file.append(row_str)

    print(div_line)
    rank_cells = [f"{avg_ranks[i]:^{col_w}.2f}" for i in range(len(active_alg_names))]
    rank_line = (
        f"{'Friedman 平均排名 (越低越好)':<34} | "
        f"{'--':^{opt_col_w}} | "
        + " | ".join(rank_cells)
    )
    print(rank_line)
    print(div_line)

    wtl_cells = []
    wilcoxon_cells = []
    for a in active_alg_names:
        if a == EADS_KEY:
            wtl_cells.append(f"{'-- (基准)':^{col_w}}" if is_fast_mode else f"{'--':^{col_w}}")
            wilcoxon_cells.append(f"{'-- (基准)':^{col_w}}" if is_fast_mode else f"{'--':^{col_w}}")
        else:
            w1, t1, l1 = mean_wtl_stats[a]
            wtl_cells.append(f"{f'{w1} 胜 / {t1} 平 / {l1} 负':^{col_w}}" if is_fast_mode else f"{f'{w1}/{t1}/{l1}':^{col_w}}")
            w2, t2, l2 = wilcoxon_stats[a]
            wilcoxon_cells.append(f"{f'{w2} / {t2} / {l2}':^{col_w}}" if is_fast_mode else f"{f'{w2}/{t2}/{l2}':^{col_w}}")

    wtl_line = (
        f"{'均值直接胜平负 (+/≈/-)':<34} | "
        f"{'--':^{opt_col_w}} | "
        + " | ".join(wtl_cells)
    )
    wilcoxon_line = (
        f"{'Wilcoxon 检验 (+/≈/-)':<34} | "
        f"{'--':^{opt_col_w}} | "
        + " | ".join(wilcoxon_cells)
    )
    print(wtl_line)
    print(wilcoxon_line)
    print(div_line)

    # 同时将汇总统计表保存至结果目录下的 txt 文件方便随时查阅
    summary_txt_path = os.path.join(results_dir, f"CEC2017_D{ndim}_{mode}_Summary_Table.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(div_line + "\n")
        f.write(header + "\n")
        f.write(div_line + "\n")
        for r_str in table_rows_for_file:
            f.write(r_str + "\n")
        f.write(div_line + "\n")
        f.write(rank_line + "\n")
        f.write(div_line + "\n")
        f.write(wtl_line + "\n")
        f.write(wilcoxon_line + "\n")
        f.write(div_line + "\n")
    print(f"[OK] 汇总统计表格已同步保存至: {summary_txt_path}")

    # 绘制 3x4 (或 5x6) 300 DPI 半对数收敛曲线对比大图 (含 ±SEM 阴影可信度边界带)
    if n_funcs <= 12:
        n_rows, n_cols = 3, 4
        figsize = (22, 14) if is_fast_mode else (24, 15)
    else:
        n_rows, n_cols = 5, 6
        figsize = (28, 22)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    colors = {
        "JADE":           "#1f77b4",  # 沉稳蓝
        "SHADE":          "#ff7f0e",  # 琥珀橙
        "CoBiDE":         "#17becf",  # 青色 (新增 CoBiDE)
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
        "CoBiDE":         ":",
        "L-SHADE":        ":",
        "LSHADE-cnEpSin": "--",
        "IMODE":          "-.",
        "CMA-ES":         ":",
        "CL-PSO":         "-.",
        "GSKA":           "--",
        "EADS (本文)":    "-",   # 实线加粗
    }

    x_epochs = np.arange(1, epoch + 1)

    for idx, (func_title, _) in enumerate(expected_specs):
        ax = axes[idx]
        for alg_name in active_alg_names:
            curve = all_curves[func_title][alg_name]
            sem_curve = results_summary[func_title][alg_name]["sem_curve"]
            plot_curve = np.maximum(curve, 1e-16)
            lower_band = np.maximum(curve - sem_curve, 1e-16)
            upper_band = np.maximum(curve + sem_curve, 1e-16)

            is_eads = (alg_name == EADS_KEY)
            plot_label = "EADS (Proposed)" if is_eads else alg_name
            c_hex = colors.get(alg_name, "#333333")

            ax.plot(
                x_epochs,
                plot_curve,
                label=plot_label,
                color=c_hex,
                linestyle=linestyles.get(alg_name, "--"),
                linewidth=2.5 if is_eads else (1.8 if is_fast_mode else 1.2),
                alpha=1.0 if is_eads else 0.85,
                zorder=10 if is_eads else 3,
            )
            # 在 4 算法模式下自动叠加半透明 ±SEM 可信度误差阴影带 (10 算法模式下仅对 EADS 绘制以免画面拥挤)
            if is_fast_mode or is_eads:
                ax.fill_between(
                    x_epochs,
                    lower_band,
                    upper_band,
                    color=c_hex,
                    alpha=0.15 if is_eads else 0.09,
                    zorder=9 if is_eads else 2,
                )

        ax.set_yscale("log")
        ax.set_title(func_title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Iterations (Epochs)", fontsize=8.5)
        ax.set_ylabel(r"Error $\Delta f$ (Log Scale)", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8.5 if is_fast_mode else 6.8, loc="upper right", framealpha=0.85)

    for idx in range(n_funcs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    fig_tag = "fast4_jade_shade_cobide" if is_fast_mode else "9benchmarks_with_cobide"
    output_fig_path = os.path.join(results_dir, f"mealpy_cec2017_{fig_tag}_dim{ndim}.png")
    plt.savefig(output_fig_path, dpi=300)
    root_fig_path = f"mealpy_cec2017_{fig_tag}_dim{ndim}.png"
    plt.savefig(root_fig_path, dpi=300)
    print(f"[OK] IEEE CEC-2017 (D={ndim}) 收敛对比大图已保存至: {output_fig_path} 与 {root_fig_path}")
    plt.show()


# ==============================================================================
# 5. 主程序入口: 多进程并发调度 -> 生成独立结果文件 -> 汇总统计与绘图
# ==============================================================================
if __name__ == "__main__":
    print(f"[MEALPY] 成功加载官方优化库, 版本号: {mealpy.__version__}")
    active_alg_names = get_active_alg_names(TEST_MODE)
    specs = get_benchmark_specs(DIM, mode=TEST_MODE)
    n_funcs = len(specs)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 120)
    print(f"  IEEE CEC-2017 多核并行基准测试启动 | 当前模式: [{TEST_MODE}]")
    print(f"  [实验规模] 维度 D = {DIM} | 函数总数 = {n_funcs} | 算法数量 = {len(active_alg_names)} | 独立运行 = {N_RUNS} 次 | Epoch = {EPOCH}")
    print(f"  [并行配置] 同时启动 {MAX_WORKERS} 个 CPU 核心工作进程 | 单函数结果保存目录: ./{RESULTS_DIR}/")
    print(f"  [数据落盘] JSON 内含每次运行的完整曲线矩阵 (run_curves) + 自动导出 Origin 专用 CSV 至 ./{RESULTS_DIR}/origin_csv/")
    print(f"  [测试阵容] {', '.join(active_alg_names)}")
    print("=" * 120)

    # 构造每个函数的独立任务参数列表 (显式传入 active_alg_names 保证跨平台多进程一致性)
    tasks = [
        (
            idx,
            func_title,
            cls_name,
            DIM,
            EPOCH,
            POP_SIZE,
            n_runs := N_RUNS,
            RESULTS_DIR,
            SKIP_EXISTING_FILES,
            active_alg_names,
        )
        for idx, (func_title, cls_name) in enumerate(specs, start=1)
    ]

    total_start_time = time.time()
    completed_count = 0

    # 启动多进程池并发评测所有函数
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(evaluate_single_function_worker, t): t for t in tasks
        }

        for future in as_completed(future_to_task):
            t_info = future_to_task[future]
            f_idx, f_title = t_info[0], t_info[1]
            try:
                idx, func_title, file_path, payload, elapsed, from_cache = future.result()
                completed_count += 1
                fg_val = payload.get("f_global", f_idx * 100.0)
                eads_mean = payload["algorithms"][EADS_KEY]["mean"]
                jade_mean = payload["algorithms"]["JADE"]["mean"]
                shade_mean = payload["algorithms"]["SHADE"]["mean"]
                cobide_mean = payload["algorithms"]["CoBiDE"]["mean"]
                status_tag = "读取缓存" if from_cache else f"耗时 {elapsed:.1f}s"

                print(
                    f"  [{completed_count:02d}/{n_funcs:02d} 已落盘] {func_title:<35} | "
                    f"f*={fg_val:.0f} | JADE: {jade_mean:.2e} | SHADE: {shade_mean:.2e} | "
                    f"CoBiDE: {cobide_mean:.2e} | EADS: {eads_mean:.2e} "
                    f"({status_tag}) -> {os.path.basename(file_path)}"
                )
            except Exception as exc:
                print(f"  [ERROR] 函数 #{f_idx} ({f_title}) 评测发生异常: {exc}")
                raise exc

    total_elapsed = time.time() - total_start_time
    saved_json_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")])
    print("\n" + "-" * 120)
    print(f"  [全部测试完成] 总耗时: {total_elapsed:.1f}s | 目录 ./{RESULTS_DIR}/ 下已成功生成 {len(saved_json_files)} 个 JSON 文件及对应 Origin CSV 文件:")
    for jf in saved_json_files:
        print(f"    - {os.path.join(RESULTS_DIR, jf)}")
    print("-" * 120)

    # 待所有测试函数结果文件落盘后，统一从目录读取全部文件进行统计检验与绘图
    results_summary, all_curves, f_globals, _ = load_all_results_from_dir(
        RESULTS_DIR, specs, active_alg_names
    )
    analyze_and_plot(
        specs,
        results_summary,
        all_curves,
        f_globals,
        RESULTS_DIR,
        DIM,
        EPOCH,
        TEST_MODE,
        active_alg_names,
    )