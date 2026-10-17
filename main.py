import numpy as np
from scipy.linalg import svd


class RMA_ES:
    """RMA-ES: 基于随机矩阵理论（RMT）与 BBP 尖峰相变滤波的高维流形自适应演化策略。

    内部搜索域完全闭包于拓扑超立方体 [0, 1]^d 中。
    """

    def __init__(
        self,
        d,
        obj_func,
        bounds,
        max_fe,
        N=12,
        k_max=10,
        sigma0=0.2,
        alpha_rmt=1.2,
        seed=None,
    ):
        """参数设置符合演化计算与统计物理公理体系。

        :param alpha_rmt: RMT 显著性保护系数 (理论推荐 1.1 ~ 1.5)，越大约保守
        """
        self.d = int(d)
        self.obj_func = obj_func
        b = np.asarray(bounds, dtype=float)
        self.lower, self.upper = b[:, 0], b[:, 1]
        self.width = self.upper - self.lower

        if np.any(self.width <= 0):
            raise ValueError("边界设定非法：必须满足 upper > lower。")

        self.max_fe = int(max_fe)
        self.N = int(N)
        self.mu = self.N // 2
        if self.mu < 2:
            raise ValueError("种群规模 N 至少应设置为 4。")

        self.k_max = int(max(0, min(k_max, self.d - 1)))
        self.rng = np.random.default_rng(seed)

        # 内部拓扑空间 [0, 1]^d 状态变量
        self.sigma = float(sigma0)
        self.sigma_min, self.sigma_max = 1e-12, 1.0
        self.m = self.rng.uniform(0.1, 0.9, self.d)

        # 序数重组对数权重
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mu_w = 1.0 / np.sum(self.weights**2)  # 方差有效选择质量

        # 演化参数设定 (基于 CMA-ES 理论体系)
        self.c_sigma = 4.0 / (self.d + 4.0)
        self.c_c = 2.0 / (self.d + np.sqrt(2.0)) ** 2  # 协方差学习率严格解耦

        # 理论自洽的 RMT 尖峰滤波阈值 (BBP Phase-Transition Threshold)
        # 纯随机采样下单轴投影方差的最大涨落上确界
        noise_variance = self.c_c * (1.0 + np.sqrt(self.mu / self.d)) ** 2
        self.rmt_threshold = float(alpha_rmt * noise_variance)

        # 协方差流形结构状态
        self.k = 0
        self.U = np.zeros((self.d, 0))  # 正交基底 (d x k)
        self.Lambda = np.zeros(0)  # 真实方差谱 (k,)

        self.p_N = np.zeros(self.d)  # 法空间共轭动量

        # 统计记录
        self.fe_evals = 0
        self.X_best_phys = None
        self.F_best = np.inf

    def _to_phys(self, X_norm):
        return self.lower + X_norm * self.width

    def _reflect_bounds(self, X_norm):
        rem = np.mod(X_norm, 2.0)
        return np.clip(np.where(rem > 1.0, 2.0 - rem, rem), 0.0, 1.0)

    def _chi_mean(self, df):
        if df <= 0:
            return 1e-10
        return np.sqrt(df) * (1.0 - 1.0 / (4.0 * df) + 1.0 / (21.0 * df**2))

    def _sample(self):
        """基于精确协方差低秩直和分解的采样: C = I + U(Lambda - I)U^T"""
        Z_iso = self.rng.standard_normal((self.N, self.d))

        if self.k > 0:
            # 严格各向异性拉伸：切向乘以 sqrt(Lambda)，法向保持 1.0
            coef = Z_iso @ self.U
            Z_iso += coef * (np.sqrt(self.Lambda) - 1.0) @ self.U.T

        X_norm = self._reflect_bounds(self.m + self.sigma * Z_iso)
        return X_norm

    def _evaluate(self, X_norm):
        rem_evals = self.max_fe - self.fe_evals
        if rem_evals <= 0:
            return None, None

        X_norm = X_norm[:rem_evals]
        X_phys = self._to_phys(X_norm)

        F = np.empty(len(X_phys))
        for i, x in enumerate(X_phys):
            try:
                f_val = float(self.obj_func(x))
                F[i] = f_val if np.isfinite(f_val) else np.inf
            except Exception:
                F[i] = np.inf

        self.fe_evals += len(X_phys)

        best_idx = int(np.argmin(F))
        if F[best_idx] < self.F_best:
            self.F_best = F[best_idx]
            self.X_best_phys = X_phys[best_idx].copy()

        return X_norm, F

    def _update(self, X_norm, F):
        order = np.argsort(F, kind="stable")
        X_elite = X_norm[order[: self.mu]]

        m_old = self.m
        U_sample = self.U
        k_sample = self.k

        Y = (X_elite - m_old) / self.sigma  # (mu, d)
        y_w = self.weights @ Y  # (d,)

        # 1. 均值平滑位移
        self.m = self._reflect_bounds(m_old + self.sigma * y_w)

        # 2. GO-CSA 正交法向动量控制 (严格投影于生成样本时的切空间正交补)
        z_N = y_w - U_sample @ (U_sample.T @ y_w) if k_sample > 0 else y_w
        if k_sample > 0:
            self.p_N = self.p_N - U_sample @ (U_sample.T @ self.p_N)

        self.p_N = (1.0 - self.c_sigma) * self.p_N + np.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_w
        ) * z_N

        d_free = self.d - k_sample
        ratio = np.linalg.norm(self.p_N) / self._chi_mean(d_free)
        self.sigma = float(
            np.clip(
                self.sigma
                * np.exp(
                    (self.c_sigma / (1.0 + self.c_sigma)) * (ratio - 1.0)
                ),
                self.sigma_min,
                self.sigma_max,
            )
        )

        # 3. 基于增广矩阵 SVD 的 RMT-BBP 流形提取
        if self.k_max > 0:
            # 构建历史协方差的低秩骨干阵 (d x k)
            if self.k > 0:
                B_old = self.U * np.sqrt(
                    np.maximum(0.0, self.Lambda - 1.0)
                )  # 严格提取超额方差
            else:
                B_old = np.zeros((self.d, 0))

            # 当代选择扰动矩阵 (d x mu)
            Y_weighted = (Y.T * np.sqrt(self.c_c * self.weights))

            # 增广矩阵 (d x (k + mu))
            B_aug = np.hstack([np.sqrt(1.0 - self.c_c) * B_old, Y_weighted])

            # 薄 SVD 快速分解
            U_aug, S_aug, _ = svd(B_aug, full_matrices=False)

            # S_aug^2 直接对应于超额方差 (Excess Variance) 的特征值谱
            excess_var_spectrum = S_aug**2

            # 【修复致命死锁点】：基于 BBP 尖峰检验，判定哪些维度真正具有超越随机高斯的物理信号
            valid = excess_var_spectrum > self.rmt_threshold
            k_new = int(np.sum(valid))
            self.k = max(0, min(k_new, self.k_max))

            if self.k > 0:
                self.U = U_aug[:, : self.k]
                # 真实方差 = 背景方差 (1.0) + 显著超额方差，并设置安全阻尼
                self.Lambda = 1.0 + excess_var_spectrum[: self.k]
            else:
                self.U = np.zeros((self.d, 0))
                self.Lambda = np.zeros(0)

    def optimize(self):
        X_norm, F = self._evaluate(self._sample())
        while (
            X_norm is not None
            and len(X_norm) >= self.mu
            and self.fe_evals < self.max_fe
        ):
            self._update(X_norm, F)
            X_norm, F = self._evaluate(self._sample())

        return self.X_best_phys, self.F_best