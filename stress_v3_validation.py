# -*- coding: utf-8 -*-
"""
Round 4 · 独立验证（防止在同一批数据上做版本选择）
新种子：算法 seed = 5000+run，初始化 seed = 9000+run（与 Round 3 的 3000+/888+ 不重叠）
候选：OLDE-v2（Round 3 版）、OLDE-v3（两项全开）、v3-only-floor（仅 Bandit 下限衰减）
函数：Rosenbrock / Schwefel / Rot-Rastrigin（50D, 150k FEs）
"""
import json, time
import numpy as np
from scipy import stats
from OLDE import OLDE, rosenbrock, schwefel
from stress_test import make_rotated

DIM, MAX_FES, RUNS = 50, 150000, 60
BENCH = {
    "Rosenbrock(50D)":      (rosenbrock, -30.0, 30.0),
    "Schwefel(50D)":        (schwefel, -500.0, 500.0),
    "Rot-Rastrigin(50D)":   (make_rotated(__import__("OLDE").rastrigin, DIM, 2024, -5.12, 5.12), -5.12, 5.12),
}
ALGOS = {
    "OLDE-v2":       lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s,
                                              div_feedback=False, floor_decay=False),
    "OLDE-v3":       lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s),
    "v3-only-floor": lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s,
                                               div_feedback=False, floor_decay=True),
}

if __name__ == "__main__":
    results = {fn: {an: [] for an in ALGOS} for fn in BENCH}
    t0 = time.time()
    for fi, (fname, (func, lb, ub)) in enumerate(BENCH.items()):
        for run in range(RUNS):
            X0 = np.random.default_rng(9000 + run).uniform(lb, ub, (100, DIM))
            for aname, maker in ALGOS.items():
                _, bf = maker(func, lb, ub, 5000 + run).optimize(init_pop=X0)
                results[fname][aname].append(float(bf))
        print(f"[{fi+1}/3] {fname} 完成 ({time.time()-t0:.0f}s)", flush=True)
    json.dump(results, open("/home/user/stress_v3_validation.json", "w"))

    print("\n函数(50D, 独立种子, 60 runs)            " + "".join(f"{a:>26}" for a in ALGOS))
    for fname in BENCH:
        row = f"{fname:<38}"
        for an in ALGOS:
            v = np.array(results[fname][an])
            row += f"{v.mean():>13.3e}±{v.std():>8.1e}  "
            row += f"[收敛<1e-6: {int(np.sum(v<1e-6))}/{RUNS}]" if fname.startswith("Schwefel") else ""
        print(row)
    print("\n配对 Wilcoxon (单侧, 行 < 列) 及逐 run 胜场:")
    pairs = [("OLDE-v3", "OLDE-v2"), ("v3-only-floor", "OLDE-v2"), ("v3-only-floor", "OLDE-v3")]
    for fname in BENCH:
        r = results[fname]
        for a, b in pairs:
            va, vb = np.array(r[a]), np.array(r[b])
            try:
                p = stats.wilcoxon(va, vb, alternative="less").pvalue
            except ValueError:
                p = float("nan")
            print(f"{fname:<22} {a:>14} vs {b:<10} p={p:.2e}  胜 {int(np.sum(va<vb))}/{RUNS}  平 {int(np.sum(va==vb))}")
