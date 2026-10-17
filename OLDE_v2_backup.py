# -*- coding: utf-8 -*-
"""
OLDE: Oscillation-breathing Lifecycle Differential Evolution
（振荡呼吸-生命周期自适应差分进化算法）

三方Agent共识设计（基于 mealpy 208 算法规律总结 Blueprint v1.0）：
  [核心] SHADE 骨架: current-to-pbest/1 + 成功历史记忆自适应(F,CR) + 败者存档 + LPSR 种群缩减
  [增量1] 呼吸振荡扩散系数 o(prog) —— 只作用于扩散项，防单调收缩早熟（规律2,4）
  [增量2] 生命周期 Lévy 精英逃逸 —— 停滞 life_max 代触发，贪婪兜底（规律6,8,3）
  [增量3] 双策略信用 Bandit —— current-to-pbest/1 与 rand/1 按改进量竞争（规律7,10）
对外仅需两个参数: n_init, max_fes。关闭三个增量(breathing/escape/bandit=False)即退化为 SHADE-lite 基线。
"""
import numpy as np

LEVY_BETA = 1.5
SIGMA_U = 0.6966  # Mantegna 方法中 beta=1.5 对应的 sigma_u


class OLDE:
    def __init__(self, func, dim, lb, ub, max_fes=60000, n_init=100, n_min=4,
                 H=6, p=0.11, life_max=20, eta=0.3, osc_freq=3.0,
                 bandit_floor=0.15, breathing=True, escape=True, bandit=True, seed=42,
                 div_feedback=True, rho_star=0.2, kappa=1.0,
                 floor_decay=True, bandit_floor_end=0.05):
        self.func = func
        self.dim = dim
        self.lb = np.asarray(lb, dtype=float) * np.ones(dim)
        self.ub = np.asarray(ub, dtype=float) * np.ones(dim)
        self.max_fes = max_fes
        self.n_init, self.n_min = n_init, n_min
        self.H, self.p = H, p
        self.life_max, self.eta, self.osc_freq = life_max, eta, osc_freq
        self.bandit_floor = bandit_floor
        self.breathing, self.escape, self.bandit = breathing, escape, bandit
        # v3 消缺参数 ----------------------------------------------------
        # div_feedback: 呼吸幅度由种群实时多样性比 r 调制（v2 固定幅度）
        # rho_star: 目标多样性比（r < rho_star 视为收敛，加强呼吸）
        # kappa: 调制灵敏度；g 被夹在 [0.5, 2.0]
        # floor_decay: Bandit 下限随 prog 线性衰减 bandit_floor -> bandit_floor_end
        self.div_feedback, self.rho_star, self.kappa = div_feedback, rho_star, kappa
        self.floor_decay, self.bandit_floor_end = floor_decay, bandit_floor_end
        self.rng = np.random.default_rng(seed)

    # ---------- 内部工具 ----------
    def _cauchy(self, loc, scale, size):
        return loc + scale * np.tan(np.pi * (self.rng.random(size) - 0.5))

    def _levy(self, shape):
        u = self.rng.normal(0, SIGMA_U, shape)
        v = self.rng.normal(0, 1, shape)
        return u / (np.abs(v) ** (1.0 / LEVY_BETA) + 1e-12)

    @staticmethod
    def _w_lehmer(x, w):
        w = w / (w.sum() + 1e-30)
        return (w * x ** 2).sum() / ((w * x).sum() + 1e-30)

    @staticmethod
    def _w_mean(x, w):
        w = w / (w.sum() + 1e-30)
        return (w * x).sum()

    # ---------- 主流程 ----------
    def optimize(self, init_pop=None):
        rng, dim = self.rng, self.dim
        N = self.n_init
        X = rng.uniform(self.lb, self.ub, (N, dim)) if init_pop is None else init_pop.copy()
        FX = self.func(X)
        FEs = N
        bi = np.argmin(FX)
        gbest_x, gbest_f = X[bi].copy(), float(FX[bi])

        M_F = np.full(self.H, 0.5)
        M_CR = np.full(self.H, 0.5)
        k_mem = 0
        A = np.empty((0, dim))          # 败者外部存档
        life = np.zeros(N, dtype=int)   # 生命周期停滞计数
        credit = np.ones(2)             # Bandit 信用: [S1=current-to-pbest/1, S2=rand/1]

        while FEs < self.max_fes:
            prog = FEs / self.max_fes
            phase = 1.0 + self.eta * np.sin(2 * np.pi * self.osc_freq * prog)
            o = 1.0 + self.eta * (1 - prog) * np.sin(2 * np.pi * self.osc_freq * prog) \
                if self.breathing else 1.0            # [增量1] 呼吸系数
            # ----- 参数自适应采样（SHADE 成功记忆） -----
            r_slot = rng.integers(0, self.H, N)
            Fv = np.clip(self._cauchy(M_F[r_slot], 0.1, N), 0.1, 1.0)
            CRv = np.clip(rng.normal(M_CR[r_slot], 0.1, N), 0.0, 1.0)
            # ----- 精英池（top-p%） -----
            order = np.argsort(FX)
            elite_cnt = max(2, int(np.ceil(self.p * N)))
            pbest_idx = order[rng.integers(0, elite_cnt, N)]
            # ----- 变异伙伴 -----
            idx = np.arange(N)
            r1 = (idx + rng.integers(1, N, N)) % N                    # r1 != i
            U = np.vstack([X, A]) if len(A) else X                    # r2 来自 P∪A
            M = len(U)
            r2 = (idx + rng.integers(1, M, N)) % M
            # ----- 策略分配（Bandit） -----
            if self.bandit:
                tot = credit.sum()
                pS1 = self.bandit_floor + (1 - 2 * self.bandit_floor) * credit[0] / tot
            else:
                pS1 = 1.0
            is_S1 = rng.random(N) < pS1
            # ----- 生命周期逃逸掩码 -----
            esc = (life >= self.life_max) & np.ones(N, bool) if self.escape else np.zeros(N, bool)
            # ----- 变异 -----
            V = X.copy()
            m1 = is_S1 & ~esc                                          # S1: current-to-pbest/1
            if m1.any():
                V[m1] = (X[m1] + Fv[m1, None] * (X[pbest_idx[m1]] - X[m1])
                         + o * Fv[m1, None] * (X[r1[m1]] - U[r2[m1]]))
            m2 = ~is_S1 & ~esc                                         # S2: rand/1
            if m2.any():
                i2 = idx[m2]
                r3 = (i2 + rng.integers(1, N, i2.size)) % N
                V[m2] = X[r3] + o * Fv[m2, None] * (X[r1[m2]] - U[r2[m2]])
            if esc.any():                                              # [增量2] 精英引导 Lévy 逃逸
                ne = int(esc.sum())
                Ei = order[rng.integers(0, elite_cnt, ne)]
                S_esc = (1 - prog) * phase
                V[esc] = X[Ei] + S_esc * self._levy((ne, dim)) * (X[Ei] - X[esc])
            # ----- 二项交叉 -----
            mask = rng.random((N, dim)) < CRv[:, None]
            mask[idx, rng.integers(0, dim, N)] = True
            T = np.clip(np.where(mask, V, X), self.lb, self.ub)
            FT = self.func(T)
            FEs += N
            # ----- 贪婪选择 + 记忆更新 -----
            better = FT < FX
            wimp = np.where(better, FX - FT, 0.0)
            if better.any():
                A = np.vstack([A, X[better]]) if len(A) else X[better].copy()
                X[better], FX[better] = T[better], FT[better]
                if self.bandit:
                    credit[0] = 0.5 * credit[0] + wimp[is_S1 & better].sum()
                    credit[1] = 0.5 * credit[1] + wimp[~is_S1 & better].sum()
                M_F[k_mem] = self._w_lehmer(Fv[better], wimp[better])
                M_CR[k_mem] = self._w_mean(CRv[better], wimp[better])
                k_mem = (k_mem + 1) % self.H
            life[better] = 0
            life[~better] += 1
            life[esc] = 0                                              # 逃逸后重置，防抖
            bi = np.argmin(FX)
            if FX[bi] < gbest_f:
                gbest_x, gbest_f = X[bi].copy(), float(FX[bi])
            # ----- LPSR 线性种群缩减 -----
            prog2 = min(FEs / self.max_fes, 1.0)
            N_tar = int(round(self.n_init - (self.n_init - self.n_min) * prog2))
            if N > N_tar:
                keep = np.argsort(FX)[:N_tar]
                X, FX, life = X[keep], FX[keep], life[keep]
                N = N_tar
            if len(A) > N:
                A = A[rng.choice(len(A), N, replace=False)]
        return gbest_x, gbest_f


# ============================================================
# 基线算法（Agent C 审核要求：标准实现、同预算、同初始化、同seed）
# ============================================================
class DE:
    """Canonical DE/rand/1/bin (Storn & Price)"""
    def __init__(self, func, dim, lb, ub, max_fes=60000, NP=100, F=0.5, CR=0.9, seed=42):
        self.func, self.dim = func, dim
        self.lb = np.asarray(lb, float) * np.ones(dim)
        self.ub = np.asarray(ub, float) * np.ones(dim)
        self.max_fes, self.NP, self.F, self.CR = max_fes, NP, F, CR
        self.rng = np.random.default_rng(seed)

    def optimize(self, init_pop=None):
        rng, dim, NP, F, CR = self.rng, self.dim, self.NP, self.F, self.CR
        X = rng.uniform(self.lb, self.ub, (NP, dim)) if init_pop is None else init_pop.copy()
        FX = self.func(X)
        FEs = NP
        best = (X[np.argmin(FX)].copy(), float(FX.min()))
        idx = np.arange(NP)
        while FEs < self.max_fes:
            a = (idx + rng.integers(1, NP, NP)) % NP                # a != i
            b = (idx + rng.integers(1, NP, NP)) % NP
            c = (idx + rng.integers(1, NP, NP)) % NP
            for _ in range(4):                                       # 拒绝采样: 保证 b 与 i,a 互异
                coll = (b == idx) | (b == a)
                if not coll.any():
                    break
                b[coll] = rng.integers(0, NP, int(coll.sum()))
            for _ in range(4):                                       # 拒绝采样: 保证 c 与 i,a,b 互异
                coll = (c == idx) | (c == a) | (c == b)
                if not coll.any():
                    break
                c[coll] = rng.integers(0, NP, int(coll.sum()))
            V = X[a] + F * (X[b] - X[c])
            mask = rng.random((NP, dim)) < CR
            mask[idx, rng.integers(0, dim, NP)] = True
            T = np.clip(np.where(mask, V, X), self.lb, self.ub)
            FT = self.func(T); FEs += NP
            better = FT < FX
            X[better], FX[better] = T[better], FT[better]
            bi = np.argmin(FX)
            if FX[bi] < best[1]:
                best = (X[bi].copy(), float(FX[bi]))
        return best


class PSO:
    """Canonical constriction PSO (Clerc-Kennedy), gbest 拓扑"""
    def __init__(self, func, dim, lb, ub, max_fes=60000, NP=60,
                 w=0.7298, c1=1.49618, c2=1.49618, seed=42):
        self.func, self.dim = func, dim
        self.lb = np.asarray(lb, float) * np.ones(dim)
        self.ub = np.asarray(ub, float) * np.ones(dim)
        self.max_fes, self.NP = max_fes, NP
        self.w, self.c1, self.c2 = w, c1, c2
        self.rng = np.random.default_rng(seed)

    def optimize(self, init_pop=None):
        rng, dim, NP = self.rng, self.dim, self.NP
        X = rng.uniform(self.lb, self.ub, (NP, dim)) if init_pop is None else init_pop.copy()
        V = rng.uniform(-0.1, 0.1, (NP, dim)) * (self.ub - self.lb)
        vmax = 0.2 * (self.ub - self.lb)
        FX = self.func(X); FEs = NP
        PX, PF = X.copy(), FX.copy()
        gi = np.argmin(FX); GX, GF = X[gi].copy(), float(FX[gi])
        while FEs < self.max_fes:
            V = self.w * V + self.c1 * rng.random((NP, dim)) * (PX - X) \
                + self.c2 * rng.random((NP, dim)) * (GX - X)
            V = np.clip(V, -vmax, vmax)
            X = np.clip(X + V, self.lb, self.ub)
            FX = self.func(X); FEs += NP
            better = FX < PF
            PX[better], PF[better] = X[better], FX[better]
            gi = np.argmin(PF)
            if PF[gi] < GF:
                GX, GF = PX[gi].copy(), float(PF[gi])
        return GX, GF


class GWO:
    """Canonical Grey Wolf Optimizer (Mirjalili 2014) —— 现代群智能代表"""
    def __init__(self, func, dim, lb, ub, max_fes=60000, NP=60, seed=42):
        self.func, self.dim = func, dim
        self.lb = np.asarray(lb, float) * np.ones(dim)
        self.ub = np.asarray(ub, float) * np.ones(dim)
        self.max_fes, self.NP = max_fes, NP
        self.rng = np.random.default_rng(seed)

    def optimize(self, init_pop=None):
        rng, dim, NP = self.rng, self.dim, self.NP
        X = rng.uniform(self.lb, self.ub, (NP, dim)) if init_pop is None else init_pop.copy()
        FX = self.func(X); FEs = NP
        best_f = float(FX.min())
        while FEs < self.max_fes:
            prog = FEs / self.max_fes
            a = 2 - 2 * prog
            top3 = np.argsort(FX)[:3]
            L = X[top3]
            A_ = a * (2 * rng.random((3, NP, dim)) - 1)
            C_ = 2 * rng.random((3, NP, dim))
            Xk = np.empty((3, NP, dim))
            for k in range(3):
                Dk = np.abs(C_[k] * L[k] - X)
                Xk[k] = L[k] - A_[k] * Dk
            X = np.clip(Xk.mean(axis=0), self.lb, self.ub)
            FX = self.func(X); FEs += NP
            best_f = min(best_f, float(FX.min()))
        return None, best_f


# ============================================================
# 基准测试协议（Agent C 设计并监督公平性）
#   6个经典函数 · 30维 · 20次独立运行 · 同预算 60,000 FEs · 同初始化
# ============================================================
def sphere(x):    return np.sum(x ** 2, 1)
def rosenbrock(x): return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, 1)
def rastrigin(x): return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), 1)
def ackley(x):
    d = x.shape[1]
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, 1) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x), 1) / d) + 20 + np.e
def griewank(x):  return np.sum(x ** 2, 1) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), 1) + 1
def schwefel(x):  return 418.9828872724339 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), 1)

BENCH = {
    "Sphere(F1 单峰)":     (sphere,     -100.0,   100.0),
    "Rosenbrock(F2 病态)": (rosenbrock,  -30.0,    30.0),
    "Rastrigin(F3 多峰)":  (rastrigin,   -5.12,    5.12),
    "Ackley(F4 多峰)":     (ackley,     -32.768,  32.768),
    "Griewank(F5 多峰)":   (griewank,  -600.0,   600.0),
    "Schwefel(F6 欺骗)":   (schwefel,  -500.0,   500.0),
}

DIM, MAX_FES, RUNS = 30, 60000, 20

ALGOS = {
    "PSO":   lambda f, lb, ub, s: PSO(f, DIM, lb, ub, MAX_FES, seed=s),
    "DE":    lambda f, lb, ub, s: DE(f, DIM, lb, ub, MAX_FES, seed=s),
    "GWO":   lambda f, lb, ub, s: GWO(f, DIM, lb, ub, MAX_FES, seed=s),
    "SHADE-lite": lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, breathing=False,
                                            escape=False, bandit=False, seed=s),
    "OLDE":  lambda f, lb, ub, s: OLDE(f, DIM, lb, ub, MAX_FES, seed=s),
}

if __name__ == "__main__":
    import json
    from scipy import stats
    results = {fn: {an: [] for an in ALGOS} for fn in BENCH}
    for fi, (fname, (func, lb, ub)) in enumerate(BENCH.items()):
        for run in range(RUNS):
            seed = 1000 + run
            X0 = np.random.default_rng(777 + run).uniform(lb, ub, (100, DIM))  # 同初始化
            for aname, maker in ALGOS.items():
                algo = maker(func, lb, ub, seed)
                _, bf = algo.optimize(init_pop=X0[:algo.NP] if hasattr(algo, 'NP') else X0)
                results[fname][aname].append(bf)
        print(f"[{fi+1}/6] {fname} 完成", flush=True)

    with open("/home/user/benchmark_results.json", "w") as fp:
        json.dump(results, fp)

    print("\n" + "=" * 96)
    hdr = f"{'函数':<22}" + "".join([f"{a + ' 均值±std':>28}" for a in ALGOS])
    print(hdr); print("-" * 96)
    for fname in BENCH:
        row = f"{fname:<22}"
        for an in ALGOS:
            v = np.array(results[fname][an])
            row += f"{v.mean():>16.3e}±{v.std():>9.1e}  "
        print(row)
    print("=" * 96)
    print("\nWilcoxon符号秩检验 p值 (OLDE vs 各基线, p<0.05 视为显著更优):")
    print(f"{'函数':<22}{'vs PSO':>12}{'vs DE':>12}{'vs GWO':>12}{'vs SHADE-lite':>16}")
    for fname in BENCH:
        r = results[fname]
        row = f"{fname:<22}"
        for base in ["PSO", "DE", "GWO", "SHADE-lite"]:
            try:
                p = stats.wilcoxon(r["OLDE"], r[base], alternative='less').pvalue
                row += f"{p:>12.2e}" if np.mean(r["OLDE"]) <= np.mean(r[base]) else f"{p:>10.2e}(劣)"
            except Exception:
                row += f"{'tied':>12}"
        print(row)
