# -*- coding: utf-8 -*-
"""
AHSDO v2 —— 自适应混合群体差分优化器 (Adaptive Hybrid Swarm-Differential Optimizer)
================================================================================
将 DE 的「current-to-pbest/1 差分变异 + SHADE 成功历史自适应」与
PSO 的「速度引导 + 社会/认知学习」统一到同一搜索流，并叠加：
  - 反向学习 (OBL) 初始化 + 停滞重启     -> 维持多样性、逃逸局部最优
  - Lévy 飞行扰动                          -> 打破早熟
  - 线性种群缩减 (LPSR)                  -> 提升末段精度

依赖：仅需要 numpy（已用标准库 math/numpy 实现 levy，无需 scipy）。

运行：
    python ahsdo.py            # 默认在 6 个基准函数上演示
    python ahsdo.py --D 30     # 指定维度
    python ahsdo.py --nfes 300000
"""

import argparse
import math
import time
import numpy as np


# ----------------------------------------------------------------------------- #
# 工具函数
# ----------------------------------------------------------------------------- #
def _levy(n: int, beta: float = 1.5) -> np.ndarray:
    """生成 n 维独立 Lévy 步长（Mantegna 近似）。"""
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0.0, sigma, n)
    v = np.random.normal(0.0, 1.0, n)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step


def _obl_sample(lb: np.ndarray, ub: np.ndarray, X: np.ndarray,
                f_X: np.ndarray, func) -> tuple:
    """对给定候选 X 生成反向解并用竞争选择保留 NP 个最优。"""
    X_opp = lb + ub - X
    f_opp = np.array([func(ind) for ind in X_opp])
    cand = np.vstack([X, X_opp])
    f = np.concatenate([f_X, f_opp])
    idx = np.argsort(f)[: X.shape[0]]
    return cand[idx], f[idx]


def _bounce_back(u: np.ndarray, x_i: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """越界反射修正（L-SHADE 风格）。"""
    for j in range(len(u)):
        if u[j] < lb[j]:
            u[j] = lb[j] + np.random.rand() * (x_i[j] - lb[j])
        elif u[j] > ub[j]:
            u[j] = ub[j] - np.random.rand() * (ub[j] - x_i[j])
    return u


# ----------------------------------------------------------------------------- #
# 核心优化器
# ----------------------------------------------------------------------------- #
class AHSDO:
    """自适应混合群体差分优化器 v2。"""

    def __init__(self, func, lb, ub, dim, max_nfe,
                 NP0=None, Hm=6, p_de=0.5, p_l=0.05,
                 c1=1.5, c2=1.5, seed=None):
        self.seed = seed
        self.func = func
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = dim
        self.max_nfe = int(max_nfe)

        # —— 种群规模：初始 NP0，末段缩减到 NPend（L-SHADE 式）——
        self.NP0 = int(NP0) if NP0 else (100 if dim <= 30 else 4 * dim)
        self.NPend = max(4, int(0.1 * self.NP0))

        # —— 成功历史记忆（SHADE 风格）——
        self.Hm = Hm
        self.HF = np.full(Hm, 0.5)
        self.HCR = np.full(Hm, 0.9)
        self.kF = 0
        self.kCR = 0

        # —— 控制参数 ——
        self.p_de = p_de        # 选用 DE 变异的概率
        self.p_l = p_l          # Lévy 扰动概率
        self.c1 = c1
        self.c2 = c2
        self.p = 0.11           # pbest 排名选择比例（L-SHADE）

        # —— 停滞重启阈值（代数）——
        self.N_stag = max(5, int(0.2 * self.max_nfe / self.NP0))

        # 统计
        self.nfe = 0
        self.history = []       # 记录每代最优适应度，便于画图

    # ----------------------------- 主循环 ----------------------------- #
    def optimize(self, verbose=False):
        if self.seed is not None:
            np.random.seed(self.seed)

        dim = self.dim
        lb, ub = self.lb, self.ub

        # —— 1. OBL 初始化 ——
        X = lb + np.random.rand(self.NP0, dim) * (ub - lb)
        fX = np.array([self.func(ind) for ind in X])
        X, fX = _obl_sample(lb, ub, X, fX, self.func)
        self.nfe += 2 * self.NP0

        V = np.zeros_like(X)                       # PSO 速度
        Pbest = X.copy()
        fPbest = fX.copy()
        gbest_idx = np.argmin(fX)
        gbest = X[gbest_idx].copy()
        fgbest = fX[gbest_idx]

        no_improve = 0
        SF, SCR = [], []                           # 成功历史缓冲

        while self.nfe < self.max_nfe:
            # —— 2. 线性种群缩减 LPSR ——
            frac = self.nfe / self.max_nfe
            NP_cur = int(round(self.NP0 - (self.NP0 - self.NPend) * frac))
            NP_cur = max(self.NPend, NP_cur)
            if X.shape[0] > NP_cur:
                order = np.argsort(fX)[:NP_cur]
                X, V, Pbest, fPbest, fX = (X[order], V[order], Pbest[order],
                                           fPbest[order], fX[order])

            # —— 惯性权重线性衰减 ——
            w = 0.9 - 0.5 * frac

            # pbest 排名（用于 current-to-pbest/1 选择）
            pbest_order = np.argsort(fPbest)
            p_thresh = max(2, int(self.p * NP_cur))

            improved_this_gen = False

            for i in range(NP_cur):
                x_i = X[i]

                # —— 3. 采样自适应参数 F, CR ——
                ri = np.random.randint(self.Hm)
                F = self.HF[ri] + 0.1 * np.random.randn()
                F = min(max(F, 0.0), 1.0)
                CR = self.HCR[ri] + 0.1 * np.random.randn()
                CR = min(max(CR, 0.0), 1.0)

                # —— 4. 选择算子：DE 变异 或 PSO 速度引导 ——
                if np.random.rand() < self.p_de:
                    # current-to-pbest/1
                    p_idx = pbest_order[np.random.randint(p_thresh)]
                    r1, r2 = i, i
                    while r1 == i:
                        r1 = np.random.randint(NP_cur)
                    while r2 == i or r2 == r1:
                        r2 = np.random.randint(NP_cur)
                    v = (x_i
                         + F * (Pbest[p_idx] - x_i)
                         + F * (X[r1] - X[r2]))
                else:
                    # PSO 速度引导（位置空间）
                    r1, r2 = np.random.rand(dim), np.random.rand(dim)
                    v = (w * x_i
                         + self.c1 * r1 * (Pbest[i] - x_i)
                         + self.c2 * r2 * (gbest - x_i))

                # —— 5. 二项交叉 ——
                u = x_i.copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # —— 6. Lévy 逃逸扰动 ——
                if np.random.rand() < self.p_l:
                    u = u + _levy(dim) * (ub - lb)

                # —— 越界修正 ——
                u = _bounce_back(u, x_i, lb, ub)

                # —— 7. 选择 + 成功记录 ——
                fu = self.func(u)
                self.nfe += 1

                if fu <= fX[i]:
                    X[i] = u
                    fX[i] = fu
                    SF.append(F)
                    SCR.append(CR)
                    if fu < fPbest[i]:
                        Pbest[i] = u.copy()
                        fPbest[i] = fu
                    if fu < fgbest:
                        gbest = u.copy()
                        fgbest = fu
                        improved_this_gen = True

            self.history.append(fgbest)

            # —— 8. 回灌历史记忆（成功历史自适应）——
            if len(SF) > 0:
                self.HF[self.kF] = np.mean(SF) + 0.1 * np.std(SF)
                self.HCR[self.kCR] = np.mean(SCR) + 0.1 * np.std(SCR)
                self.HF[self.kF] = min(max(self.HF[self.kF], 0.0), 1.0)
                self.HCR[self.kCR] = min(max(self.HCR[self.kCR], 0.0), 1.0)
                self.kF = (self.kF + 1) % self.Hm
                self.kCR = (self.kCR + 1) % self.Hm
                SF, SCR = [], []

            # —— 9. 停滞重启（最差 20% OBL 重生成）——
            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.N_stag:
                worst = np.argsort(fX)[int(0.8 * NP_cur):]
                Xw, fXw = _obl_sample(lb, ub, X[worst], fX[worst], self.func)
                X[worst], fX[worst] = Xw, fXw
                self.nfe += 2 * len(worst)
                # 同步 Pbest/V 中对应项
                Pbest[worst] = Xw.copy()
                fPbest[worst] = fXw.copy()
                no_improve = 0

            if verbose and self.nfe % (10 * self.NP0) < self.NP0:
                print(f"  NFE={self.nfe:>7d}  best={fgbest:.4e}")

        return gbest, fgbest


# ----------------------------------------------------------------------------- #
# 标准基准函数（单峰 / 多峰 / 组合型）
# ----------------------------------------------------------------------------- #
def F_sphere(x):
    return float(np.sum(x ** 2))

def F_schwefel222(x):
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))

def F_rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))

def F_rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

def F_ackley(x):
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

def F_griewank(x):
    n = len(x)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1)))) + 1)

BENCHMARKS = {
    "Sphere":      (F_sphere,      -100, 100),
    "Schwefel2.22":(F_schwefel222, -100, 100),
    "Rosenbrock":  (F_rosenbrock,  -30, 30),
    "Rastrigin":   (F_rastrigin,   -5.12, 5.12),
    "Ackley":      (F_ackley,      -32, 32),
    "Griewank":    (F_griewank,    -600, 600),
}


# ----------------------------------------------------------------------------- #
# 演示
# ----------------------------------------------------------------------------- #
def run_demo(D=30, nfes=None, runs=3, seed=2024, verbose=False):
    print(f"AHSDO v2 演示  |  维度 D={D}  |  每函数独立运行 {runs} 次\n")
    print(f"{'函数':<14}{'最优均值':>16}{'标准差':>14}{'耗时(s)':>10}")
    print("-" * 56)
    for name, (func, lo, hi) in BENCHMARKS.items():
        lb = np.full(D, lo)
        ub = np.full(D, hi)
        max_nfe = nfes if nfes else 10000 * D
        best_vals, t0 = [], time.time()
        for r in range(runs):
            opt = AHSDO(func, lb, ub, D, max_nfe, seed=seed + r)
            _, fbest = opt.optimize(verbose=verbose)
            best_vals.append(fbest)
        mean = np.mean(best_vals)
        std = np.std(best_vals)
        print(f"{name:<14}{mean:>16.4e}{std:>14.4e}{time.time()-t0:>10.1f}")
    print("-" * 56)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AHSDO v2 优化器演示")
    ap.add_argument("--D", type=int, default=30, help="问题维度")
    ap.add_argument("--nfes", type=int, default=None, help="最大函数评价次数")
    ap.add_argument("--runs", type=int, default=3, help="独立运行次数")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run_demo(D=args.D, nfes=args.nfes, runs=args.runs,
             seed=args.seed, verbose=args.verbose)
