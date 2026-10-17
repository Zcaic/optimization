# -*- coding: utf-8 -*-
"""Round 4 回归检查：Round 2 的 30D 六函数协议下 v2 vs v3（seed=1000+run, 初始化 777+run）"""
import numpy as np
from scipy import stats
from OLDE import OLDE, BENCH
DIM, MAX_FES, RUNS = 30, 60000, 20
res = {fn: {"v2": [], "v3": []} for fn in BENCH}
for fname, (func, lb, ub) in BENCH.items():
    for run in range(RUNS):
        X0 = np.random.default_rng(777 + run).uniform(lb, ub, (100, DIM))
        s = 1000 + run
        res[fname]["v2"].append(float(OLDE(func, DIM, lb, ub, MAX_FES, seed=s, div_feedback=False, floor_decay=False).optimize(init_pop=X0)[1]))
        res[fname]["v3"].append(float(OLDE(func, DIM, lb, ub, MAX_FES, seed=s).optimize(init_pop=X0)[1]))
print(f"{'函数(30D,60k FEs)':<24}{'v2 均值±std':>24}{'v3 均值±std':>24}{'p(v3<v2)':>12}{'v3胜':>8}")
for fname in BENCH:
    a, b = np.array(res[fname]["v3"]), np.array(res[fname]["v2"])
    try: p = stats.wilcoxon(a, b, alternative="less").pvalue
    except ValueError: p = float("nan")
    print(f"{fname:<24}{b.mean():>14.3e}±{b.std():>8.1e}{a.mean():>14.3e}±{a.std():>8.1e}{p:>12.2e}{int(np.sum(a<b)):>6}/20")
