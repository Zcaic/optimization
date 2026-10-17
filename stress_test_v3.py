# -*- coding: utf-8 -*-
"""
Round 4 · 消缺改进判决实验（OLDE v3）
改进内容：
  (a) 呼吸幅度接实时多样性反馈（div_feedback）
  (b) Bandit 下限随进程衰减 0.15 -> 0.05（floor_decay）
协议：与 Round 3 完全一致（50维 · 150,000 FEs · 20 runs · seed=3000+run · 初始化 seed=888+run）
对照：
  SHADE-lite   : 呼吸/逃逸/Bandit 全关
  OLDE-v2      : Round 3 版本（两项消缺关闭）——应逐位复现 Round 3 数值，作为复现性检查
  OLDE-v3      : 两项消缺全开（本轮主角）
  v3-only-div  : 仅开 (a)，用于归因
  v3-only-floor: 仅开 (b)，用于归因
"""
import json
import numpy as np
from scipy import stats
from OLDE import OLDE, rosenbrock, schwefel, rastrigin
from stress_test import make_rotated, weierstrass  # 复用 Round 3 的测试函数定义

DIM, MAX_FES, RUNS = 50, 150000, 20

BENCH = {
    "Rosenbrock(病态/50D)":        (rosenbrock, -30.0, 30.0),
    "Schwefel(欺骗/50D)":          (schwefel, -500.0, 500.0),
    "Rot-Rastrigin(旋转多峰/50D)": (make_rotated(rastrigin, DIM, 2024, -5.12, 5.12), -5.12, 5.12),
    "Rot-Schwefel(旋转欺骗/50D)":  (make_rotated(schwefel, DIM, 2025, -500.0, 500.0), -500.0, 500.0),
    "Weierstrass(全崎岖/50D)":     (weierstrass, -0.5, 0.5),
}

OFF_V2 = dict(div_feedback=False, floor_decay=False)
ALGOS = {
    "SHADE-lite": lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, breathing=False,
                                            escape=False, bandit=False, seed=s),
    "OLDE-v2":    lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s, **OFF_V2),
    "OLDE-v3":    lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s),
    "v3-only-div":   lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s,
                                               div_feedback=True, floor_decay=False),
    "v3-only-floor": lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s,
                                               div_feedback=False, floor_decay=True),
}

if __name__ == "__main__":
    import time
    results = {fn: {an: [] for an in ALGOS} for fn in BENCH}
    t0 = time.time()
    for fi, (fname, (func, lb, ub)) in enumerate(BENCH.items()):
        for run in range(RUNS):
            seed = 3000 + run
            X0 = np.random.default_rng(888 + run).uniform(lb, ub, (100, DIM))
            for aname, maker in ALGOS.items():
                _, bf = maker(func, lb, ub, seed).optimize(init_pop=X0)
                results[fname][aname].append(float(bf))
        print(f"[{fi+1}/5] {fname} 完成  ({time.time()-t0:.0f}s)", flush=True)

    with open("/home/user/stress_v3_results.json", "w") as fp:
        json.dump(results, fp)

    print("\n" + "=" * 130)
    print(f"{'函数(50维,150k FEs)':<30}" + "".join(f"{a:>22}" for a in ALGOS))
    print("-" * 130)
    for fname in BENCH:
        row = f"{fname:<30}"
        for an in ALGOS:
            v = np.array(results[fname][an])
            row += f"{v.mean():>13.3e}±{v.std():>7.1e}"
        print(row)
    print("=" * 130)

    print("\nWilcoxon p (单侧, 备择: 行算法 < 对照算法)，逐 run 胜场 (行算法更小的次数/20)：")
    pairs = [("OLDE-v3", "SHADE-lite"), ("OLDE-v3", "OLDE-v2"),
             ("v3-only-div", "OLDE-v2"), ("v3-only-floor", "OLDE-v2")]
    print(f"{'函数':<30}" + "".join(f"{a+' vs '+b:>34}" for a, b in pairs))
    for fname in BENCH:
        r = results[fname]
        row = f"{fname:<30}"
        for a, b in pairs:
            va, vb = np.array(r[a]), np.array(r[b])
            try:
                p = stats.wilcoxon(va, vb, alternative="less").pvalue
            except ValueError:
                p = float("nan")
            wins = int(np.sum(va < vb))
            row += f"{'p=%.2e (%d/20)' % (p, wins):>34}"
        print(row)
