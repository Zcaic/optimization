# -*- coding: utf-8 -*-
"""
Round 3 压力测试（Agent C 主持的判决性实验）
假设：OLDE 的多样性恢复机制（呼吸/逃逸/Bandit）只在"会发生早熟"的条件下产生净收益。
条件：50维 · 150,000 FEs · 旋转(非可分)/欺骗/全地形崎岖函数。
对照：SHADE-lite（= OLDE 关三增量），同初始终、同预算、同 seed。
"""
import numpy as np
from OLDE import OLDE, rosenbrock, schwefel, rastrigin
from scipy import stats

DIM, MAX_FES, RUNS = 50, 150000, 20


def make_rotated(base, dim, seed, lb, ub):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(G)
    o = rng.uniform(lb * 0.25, ub * 0.25, dim)
    return lambda X: base((X - o) @ Q.T)


def weierstrass(X, a=0.5, b=3, kmax=20):
    k = np.arange(kmax + 1)
    ak, bk = a ** k, b ** k
    t = X[..., None] + 0.5
    s = np.sum(ak * np.cos(2 * np.pi * bk * t), axis=(1, 2))
    return s - X.shape[1] * np.sum(ak * np.cos(np.pi * bk))


BENCH = {
    "Rosenbrock(病态/50D)":        (rosenbrock, -30.0, 30.0),
    "Schwefel(欺骗/50D)":          (schwefel, -500.0, 500.0),
    "Rot-Rastrigin(旋转多峰/50D)": (make_rotated(rastrigin, DIM, 2024, -5.12, 5.12), -5.12, 5.12),
    "Rot-Schwefel(旋转欺骗/50D)":  (make_rotated(schwefel, DIM, 2025, -500.0, 500.0), -500.0, 500.0),
    "Weierstrass(全崎岖/50D)":     (weierstrass, -0.5, 0.5),
}

ALGOS = {
    "SHADE-lite": lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES,
                                            breathing=False, escape=False, bandit=False, seed=s),
    "OLDE":       lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s),
}

if __name__ == "__main__":
    results = {fn: {an: [] for an in ALGOS} for fn in BENCH}
    for fi, (fname, (func, lb, ub)) in enumerate(BENCH.items()):
        for run in range(RUNS):
            seed = 3000 + run
            X0 = np.random.default_rng(888 + run).uniform(lb, ub, (100, DIM))
            for aname, maker in ALGOS.items():
                _, bf = maker(func, lb, ub, seed).optimize(init_pop=X0)
                results[fname][aname].append(bf)
        print(f"[{fi+1}/5] {fname} 完成", flush=True)

    print("\n" + "=" * 84)
    print(f"{'函数(50维,150k FEs)':<28}{'SHADE-lite 均值±std':>26}{'OLDE 均值±std':>26}")
    print("-" * 84)
    for fname in BENCH:
        row = f"{fname:<28}"
        for an in ALGOS:
            v = np.array(results[fname][an])
            row += f"{v.mean():>16.3e}±{v.std():>8.1e}"
        print(row)
    print("=" * 84)
    print("\nWilcoxon p值 (OLDE < SHADE-lite 为更优; 标注[劣]=方向相反):")
    for fname in BENCH:
        r = results[fname]
        p = stats.wilcoxon(r["OLDE"], r["SHADE-lite"], alternative='less').pvalue
        tag = "更优" if (p < 0.05 and np.mean(r["OLDE"]) < np.mean(r["SHADE-lite"])) else ("[劣]" if np.mean(r["OLDE"]) > np.mean(r["SHADE-lite"]) else "平")
        print(f"{fname:<28}p={p:.3e}  {tag}")
    print("\n逐 run 获胜次数 (20次中 OLDE 更小的次数):")
    for fname in BENCH:
        r = results[fname]
        wins = sum(1 for a, b in zip(r["OLDE"], r["SHADE-lite"]) if a < b)
        print(f"{fname:<28}{wins}/20")
