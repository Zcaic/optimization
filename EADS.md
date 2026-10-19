# IEEE 期刊标准算法伪代码：EADS (`v1_champion` 定稿版)

本文档提供与正文中 **算子 1 \~ 算子 4 数学公式（式 (1) – 式 (21)）** 及 `eads.py`（`s2_mode="v1_champion"`）**100% 严格对齐**的两种格式伪代码：

1. **Markdown 数学排版伪代码**（适合直接在 Markdown 预览器、Notion、Typora 或 Word 初稿中阅读与排版）；

2. **IEEE Transactions (`IEEEtran`) 标准 `algorithm2e` LaTeX 源码**（适合直接复制到 Overleaf / LaTeX 双栏模板中编译）。

## 一、Markdown 格式 IEEE 标准算法伪代码

> **Algorithm 1: Eigen-Adaptive Dual-Strategy Differential Evolution (EADS)**
>
> **Input:**
>
> * 种群规模 $N$ (`pop_size`)，问题维度 $D$ (`dim`)，最大进化代数 $G_{\max}$ (`epoch`)；
>
> * 历史记忆池容量 $H = 5$，外部存档比例 $r_{\text{arc}} = 1.0$（最大容量 $\vert{}\mathcal{A}\vert{}_{\max} = \lfloor r_{\text{arc}} N \rfloor$）；
>
> * 特征基更新周期 $G_e = 5$，收缩正则化系数 $\gamma = 10^{-4}$；
>
> * 精英池退火边界 $p_{\max} = 0.20, p_{\min} = 0.04$，柯西重尾尺度 $\sigma_c = 0.05$，EMA 学习率 $\alpha = 0.15$。
>
> **Output:**
>
> * 全局最优解 $\mathbf{x}_{\text{best}}$ 及其目标适应度值 $f(\mathbf{x}_{\text{best}})$。
>
> **/* 阶段 0：种群与自适应记忆状态初始化 \*/*\*
>
> 1. 在搜索空间 $[\mathbf{lb}, \mathbf{ub}] \subset \mathbb{R}^D$ 内均匀随机初始化种群 $\mathcal{P}^{(0)} = \{\mathbf{x}_1^{(0)}, \dots, \mathbf{x}_N^{(0)}\}$，计算全部个体的目标函数值 $f(\mathbf{x}_i^{(0)})$ 并按升序排列；
>
> 2. 初始化成功历史记忆池：$M_{CR, h} \leftarrow 0.5, \; M_{F, h} \leftarrow 0.5 \;\; (\forall h \in \{1, \dots, H\})$，循环写入指针 $k_{\text{mem}} \leftarrow 1$；
>
> 3. 初始化淘汰父代外部存档 $\mathcal{A} \leftarrow \emptyset$，算子选择概率 $p_{\text{strat1}}^{(0)} \leftarrow 0.5, \; p_{\text{eigen}}^{(0)} \leftarrow 0.5$，正交特征基矩阵 $\mathbf{B} \leftarrow \mathbf{I}_D$；
>
> **/* 主进化循环 \*/*\*
> 4. **for** 进化代数 $g = 0$ **to** $G_{\max} - 1$ **do**
> 5.  将当前种群 $\mathcal{P}^{(g)}$ 按适应度升序排列：$f(\mathbf{x}_{(1)}^{(g)}) \le f(\mathbf{x}_{(2)}^{(g)}) \le \dots \le f(\mathbf{x}_{(N)}^{(g)})$；
>
>  **/\* 算子 1：对数秩加权收缩正则化协方差特征分解 — 对应正文式 (1)–(5) \*/**
> 6.  **if** $g \bmod G_e == 0$ **then**
> 7.   确定满秩截断精英规模：$K \leftarrow \min\big(N, \max(D + 1, \lfloor 0.35 N \rfloor)\big)$，提取前 $K$ 个精英个体 $\{\mathbf{x}_{(1)}^{(g)}, \dots, \mathbf{x}_{(K)}^{(g)}\}$；
> 8.   计算对数秩权重：$w_k \leftarrow \frac{\ln(K + 0.5) - \ln(k)}{\sum_{j=1}^{K} (\ln(K + 0.5) - \ln(j))}, \quad k = 1, \dots, K$；
> 9.   计算精英加权质心 $\boldsymbol{\mu}_{\text{elite}} \leftarrow \sum_{k=1}^{K} w_k \mathbf{x}_{(k)}^{(g)}$ 与加权经验协方差矩阵 $\mathbf{C}_w \leftarrow \sum_{k=1}^{K} w_k (\mathbf{x}_{(k)}^{(g)} - \boldsymbol{\mu}_{\text{elite}})(\mathbf{x}_{(k)}^{(g)} - \boldsymbol{\mu}_{\text{elite}})^T$；
> 10.   执行 Ledoit-Wolf 保迹收缩正则化：$\mathbf{C}_{\text{reg}} \leftarrow (1 - \gamma)\mathbf{C}_w + \gamma \frac{\text{Tr}(\mathbf{C}_w)}{D} \mathbf{I}_D$；
> 11.   对实对称正定矩阵 $\mathbf{C}_{\text{reg}}$ 进行谱分解 $\mathbf{C}_{\text{reg}} = \mathbf{B} \mathbf{\Lambda} \mathbf{B}^T$，更新正交特征基矩阵 $\mathbf{B} \in \mathbb{R}^{D \times D}$；
> 12.  **end if**
>
>  **/\* 初始化当代表现统计容器与动态退火精英池 — 对应正文式 (6)–(8) \*/**
> 13.  清空成功参数记录集：$S_{CR} \leftarrow \emptyset, \; S_F \leftarrow \emptyset, \; \Delta \mathcal{F} \leftarrow \emptyset$；
> 14.  重置尝试与成功计数器：$N_1, N_2, n_1, n_2 \leftarrow 0$；$N_{\text{eig}}, N_{\text{std}}, n_{\text{eig}}, n_{\text{std}} \leftarrow 0$；
> 15.  计算线性退火精英比例：$p^{(g)} \leftarrow p_{\max} - (p_{\max} - p_{\min}) \cdot \frac{g}{G_{\max}}$，精英池容量 $K_p^{(g)} \leftarrow \max\big(2, \lfloor N \cdot p^{(g)} \rfloor\big)$；
>
>  **/\* 遍历种群个体生成变异与交叉试验向量 \*/**
> 16.  **for** $i = 1$ **to** $N$ **do**
> 17.   随机采样记忆槽索引 $r_i \in \{1, \dots, H\}$；
> 18.   生成交叉率 $CR_i \leftarrow \text{clip}\big(\mathcal{N}(M_{CR, r_i}, 0.1^2), 0, 1\big)$；
> 19.   重复采样缩放因子 $F_i \sim \mathcal{C}(M_{F, r_i}, 0.1)$ 直至 $F_i > 0$，并截断 $F_i \leftarrow \min(1.0, F_i)$；
> 20.   从前 $K_p^{(g)}$ 个精英中均匀采样 $\mathbf{x}_{p\text{best}}^{(g)}$，从 $\mathcal{P}^{(g)} \setminus \{\mathbf{x}_i^{(g)}\}$ 中均匀采样 $\mathbf{x}_{r_1}^{(g)}$；
>
>   **/\* 算子 2：异构双策略变异与中点折返修复 — 对应正文式 (9)–(11) \*/**
> 21.   **if** $\text{rand}(0, 1) < p_{\text{strat1}}^{(g)}$ **then**  // **策略** $\mathcal{S}_1$**：定向精细开采 (`DE/current-to-pbest/1/archive`)**
> 22.    $N_1 \leftarrow N_1 + 1$；从联合池 $(\mathcal{P}^{(g)} \cup \mathcal{A}) \setminus \{\mathbf{x}_i^{(g)}, \mathbf{x}_{r_1}^{(g)}\}$ 中均匀采样 $\tilde{\mathbf{x}}_{r_2}^{(g)}$；
> 23.    $\mathbf{v}_i^{(g)} \leftarrow \mathbf{x}_i^{(g)} + F_i \cdot (\mathbf{x}_{p\text{best}}^{(g)} - \mathbf{x}_i^{(g)}) + F_i \cdot (\mathbf{x}_{r_1}^{(g)} - \tilde{\mathbf{x}}_{r_2}^{(g)})$；
> 24.   **else**  // **策略** $\mathcal{S}_2$**：精英基跃迁与自缩放柯西重尾扰动勘探 (`v1_champion`)**
> 25.    $N_2 \leftarrow N_2 + 1$；从 $\mathcal{P}^{(g)} \setminus \{\mathbf{x}_i^{(g)}\}$ 中采样 $\mathbf{x}_{r_2}^{(g)}$；生成 $D$ 维独立柯西向量 $\boldsymbol{\delta}_i \sim \mathcal{C}(0, \sigma_c)^D$；
> 26.    $\mathbf{v}_i^{(g)} \leftarrow \mathbf{x}_{p\text{best}}^{(g)} + F_i \cdot (\mathbf{x}_{r_1}^{(g)} - \mathbf{x}_{r_2}^{(g)}) + \boldsymbol{\delta}_i \odot (\mathbf{x}_{r_1}^{(g)} - \mathbf{x}_i^{(g)})$；
> 27.   **end if**
> 28.   对变异向量 $\mathbf{v}_i^{(g)}$ 执行 SHADE 中点折返边界修复（式 (11)）；
>
>   **/\* 算子 3：地形感知的自适应特征子空间正交旋转交叉 — 对应正文式 (12)–(15) \*/**
> 29.   随机选取强制交叉维度 $j_{\text{rand}} \in \{1, \dots, D\}$；
> 30.   **if** $\text{rand}(0, 1) < p_{\text{eigen}}^{(g)}$ **then**  // **特征子空间正交旋转交叉 (`Eigen-Subspace Crossover`)**
> 31.    $N_{\text{eig}} \leftarrow N_{\text{eig}} + 1$；将父代与变异向量正交投影至特征空间：$\hat{\mathbf{x}}_i^{(g)} \leftarrow \mathbf{B}^T \mathbf{x}_i^{(g)}, \quad \hat{\mathbf{v}}_i^{(g)} \leftarrow \mathbf{B}^T \mathbf{v}_i^{(g)}$；
> 32.    对 $\forall j \in \{1, \dots, D\}$：若 $\text{rand}_j(0,1) < CR_i$ 或 $j = j_{\text{rand}}$，令 $\hat{u}_{i,j}^{(g)} \leftarrow \hat{v}_{i,j}^{(g)}$，否则 $\hat{u}_{i,j}^{(g)} \leftarrow \hat{x}_{i,j}^{(g)}$；
> 33.    正交逆投影回原笛卡尔空间：$\mathbf{u}_i^{(g)} \leftarrow \mathbf{B} \hat{\mathbf{u}}_i^{(g)}$；
> 34.   **else**  // **标准笛卡尔坐标二项式交叉 (`Standard Binomial Crossover`)**
> 35.    $N_{\text{std}} \leftarrow N_{\text{std}} + 1$；
> 36.    对 $\forall j \in \{1, \dots, D\}$：若 $\text{rand}_j(0,1) < CR_i$ 或 $j = j_{\text{rand}}$，令 $u_{i,j}^{(g)} \leftarrow v_{i,j}^{(g)}$，否则 $u_{i,j}^{(g)} \leftarrow x_{i,j}^{(g)}$；
> 37.   **end if**
> 38.   对试验向量 $\mathbf{u}_i^{(g)}$ 执行 SHADE 中点折返边界修复（式 (11)），并评测适应度 $f(\mathbf{u}_i^{(g)})$；
>
>   **/\* 贪婪选择、外部存档维护与成功信息记录 \*/**
> 39.   **if** $f(\mathbf{u}_i^{(g)}) < f(\mathbf{x}_i^{(g)})$ **then**
> 40.    记录适应度改进量与成功参数：$\Delta \mathcal{F} \leftarrow \Delta \mathcal{F} \cup \{f(\mathbf{x}_i^{(g)}) - f(\mathbf{u}_i^{(g)})\}, \; S_{CR} \leftarrow S_{CR} \cup \{CR_i\}, \; S_F \leftarrow S_F \cup \{F_i\}$；
> 41.    累加获胜算子计数：对应变异策略成功数 $n_1 \text{ (或 } n_2) \leftarrow +1$，对应交叉模式成功数 $n_{\text{eig}} \text{ (或 } n_{\text{std}}) \leftarrow +1$；
> 42.    将被淘汰父代 $\mathbf{x}_i^{(g)}$ 存入外部存档 $\mathcal{A}$（若 $\vert{}\mathcal{A}\vert{} == \vert{}\mathcal{A}\vert{}_{\max}$，则随机替换 $\mathcal{A}$ 中一个历史个体）；
> 43.    更新个体：$\mathbf{x}_i^{(g+1)} \leftarrow \mathbf{u}_i^{(g)}$；
> 44.   **else**
> 45.    保留父代：$\mathbf{x}_i^{(g+1)} \leftarrow \mathbf{x}_i^{(g)}$；
> 46.   **end if**
> 47.  **end for**
>
>  **/\* 算子 4：拉普拉斯平滑单次尝试成功率与 Lehmer 历史记忆更新 — 对应正文式 (16)–(21) \*/**
> 48.  **if** $n_1 + n_2 > 0$ **then**
> 49.   计算贝叶斯平滑单次尝试成功率：$\hat{s}_1 \leftarrow \frac{n_1 + 0.5}{N_1 + 1.0}, \quad \hat{s}_2 \leftarrow \frac{n_2 + 0.5}{N_2 + 1.0}$；
> 50.   更新变异策略概率：$p_{\text{strat1}}^{(g+1)} \leftarrow \text{clip}\Big((1 - \alpha) p_{\text{strat1}}^{(g)} + \alpha \frac{\hat{s}_1}{\hat{s}_1 + \hat{s}_2}, \; 0.25, \; 0.90\Big)$；
> 51.  **end if**
> 52.  **if** $n_{\text{eig}} + n_{\text{std}} > 0$ **then**
> 53.   计算贝叶斯平滑单次尝试成功率：$\hat{s}_{\text{eig}} \leftarrow \frac{n_{\text{eig}} + 0.5}{N_{\text{eig}} + 1.0}, \quad \hat{s}_{\text{std}} \leftarrow \frac{n_{\text{std}} + 0.5}{N_{\text{std}} + 1.0}$；
> 54.   更新特征交叉概率：$p_{\text{eigen}}^{(g+1)} \leftarrow \text{clip}\Big((1 - \alpha) p_{\text{eigen}}^{(g)} + \alpha \frac{\hat{s}_{\text{eig}}}{\hat{s}_{\text{eig}} + \hat{s}_{\text{std}}}, \; 0.05, \; 0.95\Big)$；
> 55.  **end if**
> 56.  **if** $S_{CR} \neq \emptyset$ **and** $\sum_{k=1}^{\vert{}S_{CR}\vert{}} \Delta f_k > 10^{-12}$ **then**
> 57.   计算归一化改进权重 $\omega_k \leftarrow \frac{\Delta f_k}{\sum_{j=1}^{\vert{}S_{CR}\vert{}} \Delta f_j}$；
> 58.   更新历史记忆槽：$M_{CR, k_{\text{mem}}} \leftarrow \sum_{k=1}^{\vert{}S_{CR}\vert{}} \omega_k CR_k$，$M_{F, k_{\text{mem}}} \leftarrow \frac{\sum_{k=1}^{\vert{}S_F\vert{}} \omega_k F_k^2}{\sum_{k=1}^{\vert{}S_F\vert{}} \omega_k F_k}$；
> 59.   推进循环指针：$k_{\text{mem}} \leftarrow (k_{\text{mem}} \bmod H) + 1$；
> 60.  **end if**
> 61. **end for**
> 62. **return** 全局最优解 $\mathbf{x}_{\text{best}} = \mathbf{x}_{(1)}^{(G_{\max})}$ 及最优适应度 $f(\mathbf{x}_{\text{best}})$。



## 二、论文数学符号与代码变量对照表

| **伪代码数学符号** | **物理含义** | **eads.py 对应变量 / 方法** | **默认参数 / 取值范围** | 
| $N, D, G_{\max}$ | 种群规模、问题维度、最大进化代数 | `self.pop_size`, `self.dim`, `self.epoch` | 如 50D 下 $N=150, G_{\max}=800$ | 
| $H, k_{\text{mem}}$ | 成功历史记忆池大小、循环更新指针 | `self.H`, `self.k_mem` | $H = 5$, $k_{\text{mem}} \in \{0, \dots, 4\}$ | 
| $M_{CR}, M_F$ | 交叉率与缩放因子历史记忆槽数组 | `self.M_CR`, `self.M_F` | 初始全为 $0.5$ | 
| $\mathcal{A}, r_{\text{arc}}$ | 外部淘汰父代存档及容量系数 | `self.archive`, `self.arc_rate` | $r_{\text{arc}} = 1.0, \vert{}\mathcal{A}\vert{}_{\max} = N$ | 
| $G_e$ | 协方差特征基更新周期 | `epoch % 5 == 0` | $G_e = 5$ | 
| $K$ | 协方差估计精英截断数 | `n_elites` | $\min(N, \max(D+1, \lfloor 0.35N \rfloor))$ | 
| $w_k$ | 对数秩正定权重 | `weights` | $\propto \ln(K+0.5) - \ln(k)$ | 
| $\mathbf{C}_w, \mathbf{C}_{\text{reg}}$ | 经验协方差矩阵与保迹收缩正则化矩阵 | `cov`, `cov_reg` | $\gamma = 10^{-4}$ (`0.9999` 与 `0.0001`) | 
| $\mathbf{B}$ | 特征向量正交旋转基矩阵 | `self.B_matrix` | $\mathbf{B} \in \mathbb{R}^{D \times D}, \mathbf{B}^T\mathbf{B} = \mathbf{I}_D$ | 
| $p^{(g)}, K_p^{(g)}$ | 动态线性退火精英率与精英个体数 | `p_rate`, `p_count` | $0.20 \to 0.04$, $K_p^{(g)} \ge 2$ | 
| $\boldsymbol{\delta}_i, \sigma_c$ | 策略 2 的 $D$ 维柯西重尾扰动向量与尺度 | `cauchy_step` | $\mathcal{C}(0, 0.05)^D$ | 
| $p_{\text{strat1}}^{(g)}, p_{\text{eigen}}^{(g)}$ | 策略 1 变异概率、特征子空间交叉概率 | `self.p_strat1`, `self.p_eigen` | 初始 $0.5$，边界 $[0.25, 0.90]$ 与 $[0.05, 0.95]$ | 
| $N_1, N_2, n_1, n_2$ | 策略 1/2 当代尝试总次数与获胜次数 | `strat1_trials`, `strat2_trials`, `strat1_wins`, `strat2_wins` | 每代清零重新统计 | 
| $\hat{s}_1, \hat{s}_2, \hat{s}_{\text{eig}}, \hat{s}_{\text{std}}$ | 拉普拉斯平滑单次尝试成功率 | `sr1`, `sr2`, `sr_eig`, `sr_std` | $\frac{n + 0.5}{N_{\text{trial}} + 1.0}$ | 


以下按照你最终定稿的 **`v1_champion` 冠军版**（CEC-2022 10D: `2.04`、20D: `1.50`；CEC-2017 30D: `1.75`、50D: `1.50` 全部排名第一），从 **SCI 顶刊（如 *IEEE TEVC*、*Swarm and Evolutionary Computation*）论文撰写视角**，系统阐述 **EADS (*Eigen-Adaptive Dual-Strategy Differential Evolution*)** 的核心设计动机、四大算子的严密数学公式、理论性质证明以及时间复杂度分析。

---

### 一、 EADS 算法的核心设计动机与三大学术创新点（Introduction & Motivation）

在撰写论文引言（Introduction）与方法论概述（Methodology Overview）时，可以将 EADS 的核心特点归纳为针对传统自适应差分进化（如 SHADE、L-SHADE、CoBiDE）三大固有缺陷的靶向突破：

1. **突破旋转耦合地形下的“笛卡尔正交网格盲区”（Rotational Invariance & Valley Alignment）**
传统 DE、SHADE 和 L-SHADE 依赖标准笛卡尔二项式交叉，隐含假设优化变量沿原始坐标轴 $\mathbf{e}_1, \dots, \mathbf{e}_D$ 相互独立。在经过正交矩阵 $\mathbf{M} \in \text{O}(D)$ 旋转的病态狭长山谷（如条件数 $\kappa = 10^6$ 的 Bent Cigar、非线性耦合的 Rosenbrock）上，笛卡尔交叉以 $1 - CR^k$ 的高概率切断共轭下降方向。EADS 通过提取精英群体的**对数秩加权收缩正则化协方差特征基 $\mathbf{B}^{(g)}$**，将交叉操作动态映射至与海森矩阵（Hessian Matrix）主轴对齐的特征子空间。
2. **突破“局部精细开采”与“跨盆地重尾跳出”的单一算子矛盾（Heterogeneous Dual-Strategy Synergy）**
单一的 `DE/current-to-pbest/1` 策略擅长沿单峰斜坡下潜，但在复杂混合函数（Hybrid Functions）与多漏斗复合函数（Composition Functions）中极易陷入局部极值。EADS 构建了**时变退火精英池引导的异构双策略变异算子**：策略 1 利用外部历史归档库（Archive $\mathcal{A}$）实现稳健的局部山谷开采；策略 2 以精英个体 $\mathbf{x}_{p\text{best}}$ 为跳板，耦合**自缩放各向异性柯西重尾扰动（Self-Scaling Cauchy Perturbation）**，兼顾中前期跨盆地长程跃迁与后期 $10^{-16}$ 浮点极限收敛。
3. **破除算子自适应中的“马太效应死锁”（Jeffreys-Prior Smoothed Per-Trial Success Rate）**
传统多算子自适应算法按“成功次数占比” $\frac{n_1}{n_1+n_2}$ 更新选择概率，导致早期偶然获胜的算子因获得更多抽样次数而产生正反馈锁死（Rich-Get-Richer Bias），丧失中后期模式切换能力。EADS 提出基于 **Jeffreys 先验 $\text{Beta}(0.5, 0.5)$ 的拉普拉斯平滑单次尝试成功率更新准则**，实现无偏的在线算子调度。

---

### 二、 EADS 四大核心算子的数学公式与理论推导（Section 3: The Proposed EADS Algorithm）

设优化问题定义在有界搜索空间 $\Omega = \prod_{j=1}^{D} [lb_j, ub_j] \subset \mathbb{R}^D$ 上，种群规模为 $N$，最大迭代代数为 $G_{\max}$。第 $g$ 代种群按适应度升序排列为 $f(\mathbf{x}_{1:N}^{(g)}) \le f(\mathbf{x}_{2:N}^{(g)}) \le \dots \le f(\mathbf{x}_{N:N}^{(g)})$。

#### 算子 1：对数秩加权保迹收缩协方差特征分解算子（Log-Rank Weighted Shrinkage-Regularized Eigenspace Extraction）

为将特征分解的 $\mathcal{O}(D^3)$ 代数开销摊薄至每代 $\frac{1}{G_e}\mathcal{O}(D^3)$，EADS 每隔 $G_e = 5$ 代（即满足 $g \bmod 5 = 0$ 时）更新一次正交特征基矩阵 $\mathbf{B}^{(g)} \in \mathbb{R}^{D \times D}$：

1. **满秩自适应精英截断规模**：
选取前 $K$ 个最优精英个体构建局部地形估计池：

$$K = \min\left(N, \; \max\left(D + 1, \; \lfloor 0.35 N \rfloor\right)\right)$$



**数学原理**：强制要求 $K \ge D + 1$（当 $N > D$ 时），从线性代数层面保证样本中心化矩阵的秩满足 $\text{rank}(\mathbf{X}_{\text{elite}} - \boldsymbol{\mu}) = D$，避免高维空间（如 $D=50, 100$）中样本数小于维度导致的协方差矩阵奇异退化。
2. **信息几何对数秩非线性加权（Log-Rank Utility Weighting）**：
不同排名的精英个体携带不同的地形曲率信息。为抑制次优个体中的离群噪声，赋予第 $k$ 名精英个体对数递减权重：

$$\tilde{w}_k = \ln(K + 0.5) - \ln(k), \qquad w_k = \frac{\tilde{w}_k}{\sum_{j=1}^{K} \tilde{w}_j}, \qquad k = 1, 2, \dots, K$$



由此计算加权精英质心 $\boldsymbol{\mu}_{\text{elite}}^{(g)}$ 与加权经验协方差矩阵 $\mathbf{C}_w^{(g)}$：

$$\boldsymbol{\mu}_{\text{elite}}^{(g)} = \sum_{k=1}^{K} w_k \mathbf{x}_{k:N}^{(g)}, \qquad \mathbf{C}_w^{(g)} = \sum_{k=1}^{K} w_k \left(\mathbf{x}_{k:N}^{(g)} - \boldsymbol{\mu}_{\text{elite}}^{(g)}\right)\left(\mathbf{x}_{k:N}^{(g)} - \boldsymbol{\mu}_{\text{elite}}^{(g)}\right)^T$$


3. **保迹 Ledoit-Wolf 收缩正则化与谱分解（Trace-Preserving Shrinkage & Spectral Decomposition）**：
当进化后期种群沿狭长山谷高度聚集时，$\mathbf{C}_w^{(g)}$ 的最小特征值趋近于机器精度零，易引发数值病态。引入保迹收缩正则化算子：

$$\mathbf{C}_{\text{reg}}^{(g)} = (1 - \gamma)\mathbf{C}_w^{(g)} + \gamma \frac{\text{Tr}\left(\mathbf{C}_w^{(g)}\right)}{D}\mathbf{I}_D, \qquad \gamma = 10^{-4}$$



随后对对称正定矩阵 $\mathbf{C}_{\text{reg}}^{(g)} \succ \mathbf{0}$ 执行实对称特征分解：

$$\mathbf{C}_{\text{reg}}^{(g)} = \mathbf{B}^{(g)} \mathbf{\Lambda}^{(g)} \left(\mathbf{B}^{(g)}\right)^T$$



其中 $\mathbf{B}^{(g)} = [\mathbf{b}_1^{(g)}, \dots, \mathbf{b}_D^{(g)}]$ 为满足 $\left(\mathbf{B}^{(g)}\right)^T \mathbf{B}^{(g)} = \mathbf{I}_D$ 的正交特征向量矩阵。
**二阶泰勒近似对齐原理**：在局部极小值点 $\mathbf{x}^*$ 邻域内，目标函数二阶泰勒展开为 $f(\mathbf{x}) \approx f(\mathbf{x}^*) + \frac{1}{2}(\mathbf{x}-\mathbf{x}^*)^T \mathbf{H} (\mathbf{x}-\mathbf{x}^*)$。在截断选择压力下，精英分布的协方差矩阵渐近收敛于海森矩阵的逆 $\mathbf{C}_{\text{reg}}^{(g)} \propto \mathbf{H}^{-1}$，故特征向量矩阵 $\mathbf{B}^{(g)}$ 的列向量精确对齐黑盒旋转矩阵的主轴方向。

---

#### 算子 2：时变退火精英引导的异构双策略变异与中点折返边界修复算子（Time-Annealing Heterogeneous Dual-Strategy Mutation & Midpoint Bounce-Back Repair）

1. **个体参数采样与时变退火精英池（Time-Annealing Elite Pool）**：
对每个目标个体 $i \in \{1, \dots, N\}$，从成功历史记忆池中随机抽取索引 $r \in \{0, \dots, H-1\}$，按正态分布与柯西分布生成交叉率 $CR_i$ 与缩放因子 $F_i$：

$$CR_i = \text{clip}\left(\mathcal{N}(M_{CR, r}^{(g)}, \, 0.1^2), \; 0, \; 1\right), \qquad F_i = \min\left(1.0, \; \mathcal{C}(M_{F, r}^{(g)}, \, 0.1)\right) \quad (\text{若 } F_i \le 0 \text{ 则重采样})$$



精英池比例 $p^{(g)}$ 随迭代代数线性退火收缩：

$$p^{(g)} = p_{\max} - (p_{\max} - p_{\min}) \cdot \frac{g}{G_{\max}}, \qquad K_p^{(g)} = \max\left(2, \; \lfloor N \cdot p^{(g)} \rfloor\right)$$



其中 $p_{\max} = 0.20$，$p_{\min} = 0.04$。前期大精英池（$20\%$）防止种群过早坍缩至单一局部极值，后期小精英池（$4\%$）集中算力进行高精度局部收敛。
2. **异构双策略变异（Heterogeneous Dual-Strategy Mutation）**：
生成均匀随机数 $u_{\text{strat}} \sim \mathcal{U}(0, 1)$，按自适应概率 $p_{\text{strat1}}^{(g)}$ 动态分配变异策略：
* **策略 1：带外部归档的定向开采策略（Directional Exploitation, 当 $u_{\text{strat}} < p_{\text{strat1}}^{(g)}$ 时）**

$$\mathbf{v}_i^{(g)} = \mathbf{x}_i^{(g)} + F_i \left(\mathbf{x}_{p\text{best}}^{(g)} - \mathbf{x}_i^{(g)}\right) + F_i \left(\mathbf{x}_{r_1}^{(g)} - \tilde{\mathbf{x}}_{r_2}^{(g)}\right)$$



其中 $\mathbf{x}_{p\text{best}}^{(g)}$ 随机取自前 $K_p^{(g)}$ 个精英个体，$\mathbf{x}_{r_1}^{(g)} \in \mathcal{P}^{(g)} \setminus \{\mathbf{x}_i^{(g)}\}$，而 $\tilde{\mathbf{x}}_{r_2}^{(g)} \in (\mathcal{P}^{(g)} \cup \mathcal{A}^{(g)}) \setminus \{\mathbf{x}_i^{(g)}, \mathbf{x}_{r_1}^{(g)}\}$ 从当前种群 $\mathcal{P}^{(g)}$ 与历史淘汰父代归档库 $\mathcal{A}^{(g)}$（最大容量 $\vert{}\mathcal{A}\vert{}_{\max} = N$）的并集中采样，为长狭山谷提供稳定的历史步长方向记忆。
* **策略 2：精英锚定与自缩放柯西重尾勘探策略（Elite-Anchored Self-Scaling Cauchy Exploration, 当 $u_{\text{strat}} \ge p_{\text{strat1}}^{(g)}$ 时）**

$$\mathbf{v}_i^{(g)} = \mathbf{x}_{p\text{best}}^{(g)} + F_i \left(\mathbf{x}_{r_1}^{(g)} - \mathbf{x}_{r_2}^{(g)}\right) + \boldsymbol{\delta}_i \odot \left(\mathbf{x}_{r_1}^{(g)} - \mathbf{x}_i^{(g)}\right), \qquad \boldsymbol{\delta}_i \sim \mathcal{C}(0, \, \sigma_c)^D, \quad \sigma_c = 0.05$$



其中 $\mathbf{x}_{r_1}^{(g)}, \mathbf{x}_{r_2}^{(g)} \in \mathcal{P}^{(g)} \setminus \{\mathbf{x}_i^{(g)}\}$，$\odot$ 表示 Hadamard 逐元素乘积，$\boldsymbol{\delta}_i$ 为 $D$ 维独立标准柯西随机向量乘上尺度因子 $\sigma_c = 0.05$。
**数学原理（重尾逃逸与渐近自缩放性）**：
1. *重尾跨盆地跳出*：柯西分布的概率密度函数 $p(\delta) = \frac{1}{\pi\sigma_c[1+(\delta/\sigma_c)^2]} \sim \mathcal{O}(\delta^{-2})$ 具有多项式重尾特性，偶发的大幅值脉冲能协助个体跨越混合函数（F10/F13/F18）与多峰函数（F4）的高势垒；
2. *渐近零误差自缩放（Self-Scaling Decay）*：扰动项 $\boldsymbol{\delta}_i$ 并非直接加在坐标上，而是与个体差异向量 $(\mathbf{x}_{r_1}^{(g)} - \mathbf{x}_i^{(g)})$ 作 Hadamard 乘积。在勘探初期 $\Vert{}\mathbf{x}_{r_1}^{(g)} - \mathbf{x}_i^{(g)}\Vert{}$ 较大，重尾步长被自然放大；而在收敛末期当种群逼近全局最优解（$\Vert{}\mathbf{x}_{r_1}^{(g)} - \mathbf{x}_i^{(g)}\Vert{} \to 10^{-16}$）时，扰动幅值随种群半径同步自动衰减至零，从根本上避免了固定步长柯西变异在后期破坏高精度收敛的缺陷。




3. **SHADE 中点折返边界修复（Midpoint Bounce-Back Boundary Repair）**：
对变异向量 $\mathbf{v}_i^{(g)}$（以及交叉后的试验向量 $\mathbf{u}_i^{(g)}$）逐维执行以父代 $\mathbf{x}_i^{(g)}$ 为锚点的中点折返修复：

$$v_{i,j}^{(g)} \leftarrow \begin{cases} \dfrac{x_{i,j}^{(g)} + lb_j}{2}, & \text{若 } v_{i,j}^{(g)} < lb_j \\ \dfrac{x_{i,j}^{(g)} + ub_j}{2}, & \text{若 } v_{i,j}^{(g)} > ub_j \\ v_{i,j}^{(g)}, & \text{其它} \end{cases}$$



**数学原理**：若采用硬截断（Hard Clamping）将越界分量直接置为 $lb_j$ 或 $ub_j$，大量个体会堆积在超立方体边界 $\partial\Omega$ 上，导致边界法向方差归零并严重污染算子 1 的协方差矩阵 $\mathbf{C}_w^{(g)}$。中点折返确保修复后的个体严格位于可行域内部 $\text{int}(\Omega)$，保持协方差矩阵的非奇异性。

---

#### 算子 3：地形感知的自适应特征子空间正交旋转交叉算子（Landscape-Aware Adaptive Eigen-Subspace Crossover）

生成均匀随机数 $u_{\text{eig}} \sim \mathcal{U}(0, 1)$，依据特征交叉概率 $p_{\text{eigen}}^{(g)}$ 动态选择交叉坐标系：

* **模式 A：特征子空间正交旋转交叉（当 $u_{\text{eig}} < p_{\text{eigen}}^{(g)}$ 时）**
1. *正向正交投影（Projection to Eigenspace）*：将父代 $\mathbf{x}_i^{(g)}$ 与变异向量 $\mathbf{v}_i^{(g)}$ 投影至解耦特征子空间：

$$\hat{\mathbf{x}}_i^{(g)} = \left(\mathbf{B}^{(g)}\right)^T \mathbf{x}_i^{(g)}, \qquad \hat{\mathbf{v}}_i^{(g)} = \left(\mathbf{B}^{(g)}\right)^T \mathbf{v}_i^{(g)}$$


2. *特征主轴内二项式交叉（Eigenspace Binomial Crossover）*：生成随机掩码对角矩阵 $\mathbf{D}_{\text{mask}} = \text{diag}(m_1, \dots, m_D)$，其中 $m_j = \mathbb{I}(\text{rand}_{i,j} < CR_i \lor j = j_{\text{rand}})$，$j_{\text{rand}} \sim \mathcal{U}\{1, \dots, D\}$：

$$\hat{\mathbf{u}}_i^{(g)} = \hat{\mathbf{x}}_i^{(g)} + \mathbf{D}_{\text{mask}} \left(\hat{\mathbf{v}}_i^{(g)} - \hat{\mathbf{x}}_i^{(g)}\right)$$


3. *逆向正交还原（Inverse Projection to Decision Space）*：

$$\mathbf{u}_i^{(g)} = \mathbf{B}^{(g)} \hat{\mathbf{u}}_i^{(g)} = \mathbf{x}_i^{(g)} + \underbrace{\mathbf{B}^{(g)} \mathbf{D}_{\text{mask}} \left(\mathbf{B}^{(g)}\right)^T}_{\mathbf{P}_{\text{eigen}}} \left(\mathbf{v}_i^{(g)} - \mathbf{x}_i^{(g)}\right)$$




* **模式 B：标准笛卡尔二项式交叉（当 $u_{\text{eig}} \ge p_{\text{eigen}}^{(g)}$ 时）**

$$\mathbf{u}_i^{(g)} = \mathbf{x}_i^{(g)} + \mathbf{D}_{\text{mask}} \left(\mathbf{v}_i^{(g)} - \mathbf{x}_i^{(g)}\right)$$



**正交投影与旋转不变性原理**：由于 $\mathbf{B}^{(g)}$ 是正交矩阵，$\mathbf{P}_{\text{eigen}} = \mathbf{B}^{(g)} \mathbf{D}_{\text{mask}} (\mathbf{B}^{(g)})^T$ 是一个沿特征向量子空间的正交投影算子（满足 $\mathbf{P}_{\text{eigen}}^2 = \mathbf{P}_{\text{eigen}}$ 且 $\mathbf{P}_{\text{eigen}}^T = \mathbf{P}_{\text{eigen}}$）。在不可分旋转函数 $f(\mathbf{M}(\mathbf{x}-\mathbf{o}))$ 上，它在主轴方向上独立重组坐标分量而不破坏山谷共轭方向；而在部分可分或含轴向阶梯特征的函数上，保留的笛卡尔交叉分支则提供跨轴方差交换。

---

#### 算子 4：拉普拉斯平滑单次成功率更新与加权 Lehmer 历史记忆算子（Laplace-Smoothed Per-Trial Success Rate & Weighted Lehmer Memory）

1. **贪婪选择与外部归档更新**：
若 $f(\mathbf{u}_i^{(g)}) < f(\mathbf{x}_i^{(g)})$，记录适应度改进量 $\Delta f_i = \vert{}f(\mathbf{x}_i^{(g)}) - f(\mathbf{u}_i^{(g)})\vert{}$，将成功参数 $CR_i, F_i$ 存入集合 $\mathcal{S}_{CR}, \mathcal{S}_F$，将被替换的父代 $\mathbf{x}_i^{(g)}$ 存入归档库 $\mathcal{A}^{(g)}$（若 $\vert{}\mathcal{A}^{(g)}\vert{} = N$ 则随机替换其中一个历史个体），并更新 $\mathbf{x}_i^{(g+1)} \leftarrow \mathbf{u}_i^{(g)}$。
2. **消除马太效应的拉普拉斯平滑单次成功率更新（Jeffreys-Prior Smoothed Per-Trial Success Rate）**：
设第 $g$ 代中策略 1 与策略 2 的实际调用次数分别为 $N_1, N_2$（满足 $N_1 + N_2 = N$），严格改进父代的成功次数分别为 $n_1, n_2$；同理设特征交叉与标准笛卡尔交叉的调用次数与成功次数分别为 $N_{\text{eig}}, N_{\text{std}}$ 与 $n_{\text{eig}}, n_{\text{std}}$。
* **为什么不能用成功次数比 $\frac{n_1}{n_1+n_2}$？**
假设两算子真实单次成功率为 $q_1, q_2$，则期望成功次数为 $\mathbb{E}[n_1] = N \cdot p_{\text{strat1}}^{(g)} \cdot q_1$。若按 $\frac{n_1}{n_1+n_2}$ 更新，即便 $q_1 = q_2$，期望比值依然等于旧概率 $p_{\text{strat1}}^{(g)}$（形成鞅过程漂移与马太效应），一旦某一策略早期占优被推至 $0.90$，另一策略因抽样次数 $N_2 \to 0.1N$ 而永远无法翻盘。
* **基于 Jeffreys 先验 $\text{Beta}(0.5, 0.5)$ 的后验单次成功率估计**：
当 $n_1 + n_2 > 0$ 时，计算拉普拉斯平滑单次成功率 $\hat{s}_1^{(g)}, \hat{s}_2^{(g)}$ 并以学习率 $\alpha = 0.15$ 平滑更新：

$$\hat{s}_1^{(g)} = \frac{n_1 + 0.5}{N_1 + 1.0}, \qquad \hat{s}_2^{(g)} = \frac{n_2 + 0.5}{N_2 + 1.0}$$


$$p_{\text{strat1}}^{(g+1)} = \text{clip}\left((1 - \alpha) p_{\text{strat1}}^{(g)} + \alpha \frac{\hat{s}_1^{(g)}}{\hat{s}_1^{(g)} + \hat{s}_2^{(g)}}, \; 0.25, \; 0.90\right)$$



当 $n_{\text{eig}} + n_{\text{std}} > 0$ 时，同理更新特征交叉概率 $p_{\text{eigen}}^{(g+1)}$：

$$\hat{s}_{\text{eig}}^{(g)} = \frac{n_{\text{eig}} + 0.5}{N_{\text{eig}} + 1.0}, \qquad \hat{s}_{\text{std}}^{(g)} = \frac{n_{\text{std}} + 0.5}{N_{\text{std}} + 1.0}$$


$$p_{\text{eigen}}^{(g+1)} = \text{clip}\left((1 - \alpha) p_{\text{eigen}}^{(g)} + \alpha \frac{\hat{s}_{\text{eig}}^{(g)}}{\hat{s}_{\text{eig}}^{(g)} + \hat{s}_{\text{std}}^{(g)}}, \; 0.05, \; 0.95\right)$$



**数学性质**：分子加 $0.5$、分母加 $1.0$ 严格对应二项分布似然函数在无信息 Jeffreys 先验 $\text{Beta}(0.5, 0.5)$ 下的贝叶斯后验均值估计，既消除了样本量不对称偏差，又彻底杜绝了 $N_k = 0$ 时的除零异常。


3. **适应度改进量加权的 Lehmer 历史记忆池更新（$H = 5$）**：
当 $\vert{}\mathcal{S}_{CR}\vert{} > 0$ 且 $\sum_{k=1}^{\vert{}\mathcal{S}_{CR}\vert{}} \Delta f_k > 10^{-12}$ 时，定义归一化改进权重 $\omega_k = \frac{\Delta f_k}{\sum_{j} \Delta f_j}$，更新第 $k_{\text{mem}}$ 个记忆槽：

$$M_{CR, k_{\text{mem}}}^{(g+1)} = \sum_{k=1}^{\vert{}\mathcal{S}_{CR}\vert{}} \omega_k S_{CR, k}, \qquad M_{F, k_{\text{mem}}}^{(g+1)} = \frac{\sum_{k=1}^{\vert{}\mathcal{S}_F\vert{}} \omega_k S_{F, k}^2}{\sum_{k=1}^{\vert{}\mathcal{S}_F\vert{}} \omega_k S_{F, k}}, \qquad k_{\text{mem}} \leftarrow (k_{\text{mem}} + 1) \bmod H$$



---

### 三、 算法时间与空间复杂度分析（Computational Complexity Analysis）

在 SCI 论文的 **Section 3.6: Computational Complexity** 中，单代时间复杂度可分解为以下四项：

1. **种群排序与精英筛选**：每代对 $N$ 个个体按适应度排序，复杂度为 $\mathcal{O}(N \log N)$；
2. **对数秩收缩协方差与特征分解（每 $G_e = 5$ 代执行一次）**：计算 $K \times D$ 精英矩阵的加权协方差需 $\mathcal{O}(K D^2)$，实对称矩阵特征分解 `eigh` 需 $\mathcal{O}(D^3)$，分摊至每代的平均复杂度为 $\mathcal{O}\left(\frac{1}{G_e}(K D^2 + D^3)\right)$；
3. **异构双策略变异与特征子空间正交投影**：每个个体若执行特征子空间交叉，包含两次矩阵-向量乘法 $\left(\mathbf{B}^{(g)}\right)^T \mathbf{v}$ 与 $\mathbf{B}^{(g)} \hat{\mathbf{u}}$，单次耗时 $\mathcal{O}(D^2)$，$N$ 个个体的总耗时为 $\mathcal{O}(p_{\text{eigen}} N D^2)$；
4. **统计更新与记忆池维护**：耗时 $\mathcal{O}(N)$。

因此，EADS 在 $G_{\max}$ 代进化过程中的**总时间复杂度**为：


$$T_{\text{EADS}} = \mathcal{O}\left(G_{\max} \cdot \left(N \cdot T_{\text{eval}} + N \log N + N D^2 + \frac{1}{G_e} D^3\right)\right)$$


其中 $T_{\text{eval}}$ 为单次目标函数评估耗时。在工程优化（如 CFD 气动仿真）或高维复杂函数评测中，$T_{\text{eval}} \gg D^2$，EADS 增加的线性代数运算开销几乎可以忽略不计，却能换取收敛精度的数量级跃升。**空间复杂度**方面，除存储规模为 $N \times D$ 的种群与归档库 $\mathcal{A}$ 外，仅需维护一个 $D \times D$ 的正交基矩阵 $\mathbf{B}^{(g)}$，总空间复杂度为 $\mathcal{O}(N D + D^2)$。

---
