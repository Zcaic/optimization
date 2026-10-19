### 一、 核心数学符号体系 (Nomenclature)

| 符号 | 数学定义与物理意义 |
| --- | --- |
| $D \in \mathbb{N}^+$ | 优化问题的决策空间维度 (Dimension of Decision Space) |
| $N \in \mathbb{N}^+$ | 种群规模 (Population Size)，满足满秩采样约束 $N \ge \lceil (D+1)/0.35 \rceil$ |
| $t, T_{\max}$ | 当前演化代数与最大演化代数 ($t \in \{1, 2, \dots, T_{\max}\}$) |
| $\mathbf{x}_i^{(t)}, \mathbf{v}_i^{(t)}, \mathbf{u}_i^{(t)}$ | 第 $t$ 代第 $i$ 个目标向量 (Target)、变异向量 (Mutant) 与试验向量 (Trial Vector) |
| $K \in \mathbb{N}^+$ | 参与协方差估计的优势个体截断规模，$K = \max(D+1, \lfloor 0.35 N \rfloor)$ |
| $w_k \in (0, 1)$ | 第 $k$ 优个体的对数秩权重系数 (Log-Rank Weight Coefficient) |
| $\mathbf{C}_w^{(t)}, \mathbf{C}_{\text{reg}}^{(t)}$ | 对数秩加权协方差矩阵与 Ledoit-Wolf 保迹收缩正则化协方差矩阵 |
| $\mathbf{B}^{(t)} \in \mathbb{R}^{D \times D}$ | 正交特征向量矩阵 (Orthogonal Eigenbasis Matrix)，满足 $(\mathbf{B}^{(t)})^T \mathbf{B}^{(t)} = \mathbf{I}_D$ |
| $\mathcal{A}^{(t)}$ | 淘汰父代个体的外部归档集合 (External Archive)，最大容量 $\vert{}\mathcal{A}^{(t)}\vert{} \le N$ |
| $p^{(t)} \in [0.04, 0.20]$ | 时变退火精英池比例 (Time-Annealing Elite Pool Ratio) |
| $p_{\text{strat1}}^{(t)}, p_{\text{eigen}}^{(t)}$ | 变异策略 1 触发概率與特征子空间交叉触发概率 |
| $\mathbf{M}_{CR}, \mathbf{M}_F \in \mathbb{R}^H$ | 交叉概率与缩放因子的历史成功记忆池 (记忆容量 $H = 5$) |

### 二、 四大核心算子数学公式与理论机理

#### 算子 1：对数秩加权收缩正则化特征子空间提取 (Log-Rank Weighted Shrinkage-Regularized Eigenspace Extraction)

**1. 数学公式**
每隔 $G_e = 5$ 代，将当前种群 $\mathcal{P}^{(t)}$ 按适应度升序排列为 $f(\mathbf{x}_{(1:N)}^{(t)}) \le f(\mathbf{x}_{(2:N)}^{(t)}) \le \dots \le f(\mathbf{x}_{(N:N)}^{(t)})$，截取前 $K = \max(D+1, \lfloor 0.35 N \rfloor)$ 个优势个体构建样本矩阵。
首先，计算每个优势个体的对数秩衰减权重 $w_k$：


$$w_k = \frac{\ln(K + 0.5) - \ln(k)}{\sum_{j=1}^{K} \left(\ln(K + 0.5) - \ln(j)\right)}, \quad k = 1, 2, \dots, K$$


接着，计算加权分布质心 $\boldsymbol{\mu}_w^{(t)}$ 与对数秩加权经验协方差矩阵 $\mathbf{C}_w^{(t)}$：


$$\boldsymbol{\mu}_w^{(t)} = \sum_{k=1}^{K} w_k \mathbf{x}_{(k:N)}^{(t)}$$

$$\mathbf{C}_w^{(t)} = \sum_{k=1}^{K} w_k \left(\mathbf{x}_{(k:N)}^{(t)} - \boldsymbol{\mu}_w^{(t)}\right)\left(\mathbf{x}_{(k:N)}^{(t)} - \boldsymbol{\mu}_w^{(t)}\right)^T$$


为防止演化后期种群坍缩进低维流形导致 $\mathbf{C}_w^{(t)}$ 病态或奇异，引入 Ledoit-Wolf 型保迹线性收缩正则化：


$$\mathbf{C}_{\text{reg}}^{(t)} = (1 - \lambda)\mathbf{C}_w^{(t)} + \lambda \left(\frac{\text{Tr}(\mathbf{C}_w^{(t)})}{D}\right) \mathbf{I}_D, \quad \lambda = 10^{-4}$$


最后，对实对称正定矩阵 $\mathbf{C}_{\text{reg}}^{(t)}$ 执行谱分解 (Spectral Decomposition)：


$$\mathbf{C}_{\text{reg}}^{(t)} = \mathbf{B}^{(t)} \mathbf{\Lambda}^{(t)} (\mathbf{B}^{(t)})^T$$


其中 $\mathbf{\Lambda}^{(t)} = \text{diag}(\sigma_1^2, \dots, \sigma_D^2)$ 为特征值对角阵，$\mathbf{B}^{(t)} = [\mathbf{b}_1, \dots, \mathbf{b}_D] \in \mathbb{R}^{D \times D}$ 为标准正交特征基矩阵。

**2. 数学原理与几何意义**

* **二阶曲率逆向感知**：在二阶泰勒展开下，水平集 $\{ \mathbf{x} \mid f(\mathbf{x}) \le c \}$ 内高适应度个体的空间分布协方差近似正比于目标函数海森矩阵的逆 $\mathbf{H}^{-1}$（即 $\mathbf{C}_w^{(t)} \propto \mathbf{H}^{-1}$）。因此，$\mathbf{C}_{\text{reg}}^{(t)}$ 的最大特征值对应的特征向量 $\mathbf{b}_D$ 严格对齐于狭长弯曲峡谷（如 Rosenbrock 函数）的切向主脊线。
* **对数秩滤波去噪**：相比传统等权协方差（赋予第 1 名与第 $K$ 名相同的权重 $1/K$），对数秩加权 $w_k \propto \ln(K+0.5)-\ln(k)$ 赋予顶尖精英指数级更高的方向话语权，滤除了排在 $35\%$ 边缘处次优个体的梯度法向噪声，同时保持 $K \ge D+1$ 的满秩自由度。
* **保迹谱下界保证**：收缩项 $\frac{\text{Tr}(\mathbf{C}_w^{(t)})}{D}\mathbf{I}_D$ 与原矩阵具有完全相同的迹（$\text{Tr}(\mathbf{C}_{\text{reg}}^{(t)}) = \text{Tr}(\mathbf{C}_w^{(t)})$），在不改变总体方差尺度的前提下，将最小特征值严格下界抬升至 $\lambda \bar{\sigma}^2 > 0$，从理论上杜绝了正交特征分解 `eigh` 的数值崩溃。

#### 算子 2：时变精英池引导的异构双策略变异算子 (Time-Annealing Elite-Guided Heterogeneous Dual-Strategy Mutation)

**1. 数学公式**
在第 $t$ 代，首先根据线性退火调度确定精英池截断比例 $p^{(t)}$ 与精英池规模 $K_p^{(t)}$：


$$p^{(t)} = p_{\max} - (p_{\max} - p_{\min}) \cdot \frac{t}{T_{\max}}, \quad K_p^{(t)} = \max\left(2, \lfloor N \cdot p^{(t)} \rfloor\right)$$


其中 $p_{\max} = 0.20$，$p_{\min} = 0.04$。随后从前 $K_p^{(t)}$ 个顶尖个体中等概率抽取 $\mathbf{x}_{p\text{best}}^{(t)}$。
对于每个目标个体 $\mathbf{x}_i^{(t)}$，生成均匀随机数 $u_{\text{mut}} \sim \mathcal{U}(0, 1)$，并根据动态选择概率 $p_{\text{strat1}}^{(t)}$ 执行异构双通道变异：


$$\mathbf{v}_i^{(t)} = \begin{cases} \mathbf{x}_i^{(t)} + F_i \left(\mathbf{x}_{p\text{best}}^{(t)} - \mathbf{x}_i^{(t)}\right) + F_i \left(\mathbf{x}_{r_1}^{(t)} - \tilde{\mathbf{x}}_{r_2}^{(t)}\right), & \text{if } u_{\text{mut}} < p_{\text{strat1}}^{(t)} \quad (\text{Strategy 1: Exploitation}) \\ \mathbf{x}_{p\text{best}}^{(t)} + F_i \left(\mathbf{x}_{r_1}^{(t)} - \mathbf{x}_{r_2}^{(t)}\right) + \boldsymbol{\delta}_i \odot \left(\mathbf{x}_{r_1}^{(t)} - \mathbf{x}_i^{(t)}\right), & \text{otherwise} \quad (\text{Strategy 2: Exploration}) \end{cases}$$


其中：

* $r_1, r_2 \in \{1, \dots, N\} \setminus \{i\}$ 为互斥随机个体索引，$\tilde{\mathbf{x}}_{r_2}^{(t)} \in \mathcal{P}^{(t)} \cup \mathcal{A}^{(t)}$ 为从当前种群与历史淘汰归档并集中随机抽取的个体；
* $\odot$ 表示 Hadamard 逐元素乘积，$\boldsymbol{\delta}_i = [\delta_{i,1}, \dots, \delta_{i,D}]^T$ 为重尾柯西随机向量，其各分量独立服从尺度参数 $\gamma = 0.05$ 的柯西分布：

$$\delta_{i,j} \sim \mathcal{C}(0, \gamma), \quad f_{\mathcal{C}}(\delta; 0, \gamma) = \frac{1}{\pi \gamma \left[1 + (\delta/\gamma)^2\right]}$$



**2. 数学原理与几何意义**

* **选择压力时变退火 ($p^{(t)}: 0.20 \to 0.04$)**：在演化早期（$t \ll T_{\max}$），较大的 $p^{(t)} \approx 0.20$ 将引导中心分散至前 $20\%$ 的不同局部极值盆地，有效阻止种群在多峰旋转地形（如 Griewank 与 Rastrigin）中过早坍缩至同一局部凹坑；在演化末期（$t \to T_{\max}$），$p^{(t)} \to 0.04$ 将引导焦点收束至最顶尖的 $2 \sim 8$ 个精英解，实现指数级局部精修。
* **多尺度柯西重尾跳变 (Self-Scaled Cauchy Basin Hopping)**：策略 2 直接以精英个体 $\mathbf{x}_{p\text{best}}^{(t)}$ 为跃迁基点，并附加重尾项 $\boldsymbol{\delta}_i \odot (\mathbf{x}_{r_1}^{(t)} - \mathbf{x}_i^{(t)})$。由于柯西分布二阶矩发散（方差无穷大，尾部以幂律 $O(\delta^{-2})$ 衰减），它能以显著高于高斯分布的概率产生跨数量级的大步长脉冲，协助停滞个体跨越深谷势垒；同时，乘子 $(\mathbf{x}_{r_1}^{(t)} - \mathbf{x}_i^{(t)})$ 赋予了扰动天然的**空间自缩放特性**——当后期种群收敛、$\Vert{}\mathbf{x}_{r_1}^{(t)} - \mathbf{x}_i^{(t)}\Vert{}_2 \to 0$ 时，重尾扰动半径随之自动归零，不会破坏末期精细收敛。

#### 算子 3：地形感知的自适应特征子空间双模式交叉 (Landscape-Aware Adaptive Eigen-Subspace Crossover)

**1. 数学公式**
生成随机数 $u_{\text{eig}} \sim \mathcal{U}(0, 1)$ 及强制交叉维度索引 $j_{\text{rand}} \sim \mathcal{U}\{1, \dots, D\}$：

* **模式 A（正交特征子空间旋转交叉，当 $u_{\text{eig}} < p_{\text{eigen}}^{(t)}$ 时触发）**：
1. **正交投影解耦**：将父代向量与变异向量映射至特征坐标系：

$$\hat{\mathbf{x}}_i^{(t)} = (\mathbf{B}^{(t)})^T \mathbf{x}_i^{(t)}, \quad \hat{\mathbf{v}}_i^{(t)} = (\mathbf{B}^{(t)})^T \mathbf{v}_i^{(t)}$$


2. **主轴独立二项式重组**：在特征子空间内执行维度级交叉：

$$\hat{u}_{i, j}^{(t)} = \begin{cases} \hat{v}_{i, j}^{(t)}, & \text{if } \text{rand}_{i, j}(0, 1) < CR_i \text{ or } j = j_{\text{rand}} \\ \hat{x}_{i, j}^{(t)}, & \text{otherwise} \end{cases}, \quad j = 1, \dots, D$$


3. **逆正交重构**：将重组向量逆映射回原决策空间：

$$\mathbf{u}_i^{(t)} = \mathbf{B}^{(t)} \hat{\mathbf{u}}_i^{(t)}$$




* **模式 B（原笛卡尔坐标系标准交叉，当 $u_{\text{eig}} \ge p_{\text{eigen}}^{(t)}$ 时触发）**：

$$u_{i, j}^{(t)} = \begin{cases} v_{i, j}^{(t)}, & \text{if } \text{rand}_{i, j}(0, 1) < CR_i \text{ or } j = j_{\text{rand}} \\ x_{i, j}^{(t)}, & \text{otherwise} \end{cases}, \quad j = 1, \dots, D$$



**2. 数学原理与几何意义**

* **旋转不可分地形的坐标解耦**：设对角选维算子为 $\mathbf{M}_i = \text{diag}(m_{i,1}, \dots, m_{i,D})$（其中 $m_{i,j} \in \{0, 1\}$）。模式 A 的等价线性代数表达式为：

$$\mathbf{u}_i^{(t)} = \mathbf{B}^{(t)} \mathbf{M}_i (\mathbf{B}^{(t)})^T \mathbf{v}_i^{(t)} + \mathbf{B}^{(t)} (\mathbf{I}_D - \mathbf{M}_i) (\mathbf{B}^{(t)})^T \mathbf{x}_i^{(t)}$$



当目标函数经过未知正交旋转矩阵 $\mathbf{Q}$ 变换（如 $f(\mathbf{Q}(\mathbf{x}-\mathbf{o}))$）时，传统笛卡尔交叉（$\mathbf{B} = \mathbf{I}_D$）只能沿着与坐标轴平行的超立方体棱边移动，在斜向峡谷中产生严重的“之字形碰壁（Zig-Zagging）”。而当 $\mathbf{B}^{(t)} \approx \mathbf{Q}^T$ 时，模式 A 的交叉超平面严格平行于旋转后的等高线主轴，彻底消除了旋转变量间的交叉耦合冲突。
* **可分地形的无损兼容**：在严格坐标轴对齐的可分多峰函数（如 F3 Shifted Rastrigin）中，$x_1, \dots, x_D$ 彼此正交独立，任何非单位阵旋转反而会引入人为耦合干扰。双模式切换机制允许算法通过自适应概率 $p_{\text{eigen}}^{(t)}$ 自动识别两类地形，实现“可分地形走原坐标、旋转地形走特征空间”的自适应分流。

#### 算子 4：双通道闭环概率追踪与 Lehmer 成功历史记忆更新 (Dual-Channel Closed-Loop Adaptation & Success-History Memory)

**1. 数学公式**
定义世代成功改进量 $\Delta f_i^{(t)} = \max\left(0, f(\mathbf{x}_i^{(t)}) - f(\mathbf{u}_i^{(t)})\right)$。统计第 $t$ 代两类变异策略的成功次数 $n_{\text{strat1}}^{(t)}, n_{\text{strat2}}^{(t)}$ 以及两类交叉模式的成功次数 $n_{\text{eig}}^{(t)}, n_{\text{std}}^{(t)}$。

* **双通道一阶无限脉冲响应 (IIR) 指数平滑概率更新**：
当 $n_{\text{strat1}}^{(t)} + n_{\text{strat2}}^{(t)} > 0$ 时：

$$p_{\text{strat1}}^{(t+1)} = \text{clip}\left((1 - \alpha) p_{\text{strat1}}^{(t)} + \alpha \cdot \frac{n_{\text{strat1}}^{(t)}}{n_{\text{strat1}}^{(t)} + n_{\text{strat2}}^{(t)}}, \; 0.10, \; 0.90\right)$$


$$p_{\text{eigen}}^{(t+1)} = \text{clip}\left((1 - \alpha) p_{\text{eigen}}^{(t)} + \alpha \cdot \frac{n_{\text{eig}}^{(t)}}{n_{\text{eig}}^{(t)} + n_{\text{std}}^{(t)}}, \; 0.05, \; 0.95\right)$$



其中阻尼平滑率 $\alpha = 0.15$，双向截断阈值 $[0.10, 0.90]$ 与 $[0.05, 0.95]$ 保证探索通道永不归零。
* **适应度改进加权 Lehmer 记忆池更新**：
令成功参数集合为 $\mathcal{S}_{CR}, \mathcal{S}_F$，各成功个体的归一化贡献权重为 $w_k = \frac{\Delta f_k^{(t)}}{\sum_{m=1}^{\vert{}\mathcal{S}\vert{}} \Delta f_m^{(t)}}$。采用加权算术均值更新交叉率记忆槽 $M_{CR, k_{\text{mem}}}$，采用加权二阶 Lehmer 均值更新缩放因子记忆槽 $M_{F, k_{\text{mem}}}$：

$$M_{CR, k_{\text{mem}}}^{(t+1)} = \sum_{k=1}^{\vert{}\mathcal{S}_{CR}\vert{}} w_k \cdot S_{CR, k}, \qquad M_{F, k_{\text{mem}}}^{(t+1)} = \frac{\sum_{k=1}^{\vert{}\mathcal{S}_F\vert{}} w_k \cdot S_{F, k}^2}{\sum_{k=1}^{\vert{}\mathcal{S}_F\vert{}} w_k \cdot S_{F, k}}$$



随后循环移动记忆指针 $k_{\text{mem}} \leftarrow (k_{\text{mem}} + 1) \bmod H$。

**2. 数学原理与几何意义**

* **一阶低通滤波抗抖动**：单代成功率具有较强的伯努利采样方差，指数移动平均（IIR 低通滤波器，时间常数 $\tau \approx 1/\alpha \approx 6.7$ 代）滤除了单代随机波动，使 $p_{\text{eigen}}^{(t)}$ 在可分多峰地形上平稳收敛至下界 $0.05$，而在旋转复合地形上快速攀升至上界 $0.95$。
* **Lehmer 二阶矩抗早熟偏置**：算术均值与二阶 Lehmer 均值满足柯西-施瓦茨不等式 $\text{Mean}_{WL}(\mathcal{S}_F) \ge \text{Mean}_{WA}(\mathcal{S}_F)$。在演化中后期，沿着当前梯度微调的小步长 $F_k$ 虽然容易取得微小成功，但大步长 $F_k$ 才能带来大的适应度跃升 $\Delta f_k$。加权 Lehmer 均值通过平方比值结构放大了较大成功步长 $S_{F,k}$ 的权重，防止缩放因子 $M_F$ 过早衰减至 0。

### 三、 IEEE 伪代码

> **Algorithm 1:** Eigen-Adaptive Dual-Strategy Differential Evolution (EADS)
> **Input:** Objective function $f(\mathbf{x})$, dimension $D$, bounds $[\mathbf{lb}, \mathbf{ub}]$, population size $N$, maximum epochs $T_{\max}$, memory size $H = 5$, archive rate $r_{\text{arc}} = 1.0$, eigen-update interval $G_e = 5$.
> **Output:** Global best solution $\mathbf{x}_{\text{best}}$ and minimum fitness $f(\mathbf{x}_{\text{best}})$.
> 1: **Initialize** population $\mathcal{P}^{(0)} = \{\mathbf{x}_1^{(0)}, \dots, \mathbf{x}_N^{(0)}\}$ uniformly in $[\mathbf{lb}, \mathbf{ub}]$ and evaluate $f(\mathbf{x}_i^{(0)})$.
> 2: **Initialize** memory $\mathbf{M}_{CR} \leftarrow [0.5]^H$, $\mathbf{M}_F \leftarrow [0.5]^H$, pointer $k_{\text{mem}} \leftarrow 1$, archive $\mathcal{A} \leftarrow \emptyset$.
> 3: **Initialize** adaptive probabilities $p_{\text{strat1}} \leftarrow 0.5$, $p_{\text{eigen}} \leftarrow 0.5$, orthogonal basis $\mathbf{B} \leftarrow \mathbf{I}_D$.
> 4: **Sort** $\mathcal{P}^{(0)}$ in ascending order of objective fitness.
> 5: **for** $t = 1$ **to** $T_{\max}$ **do**
> 6: $\quad$ **if** $t \bmod G_e == 0$ **then** $\quad$ *// Operator 1: Log-Rank Weighted Eigenspace Extraction*
> 7: $\quad\quad$ Set elite sample size $K \leftarrow \max(D + 1, \lfloor 0.35 N \rfloor)$ and extract top-$K$ individuals $\{\mathbf{x}_{(1:N)}, \dots, \mathbf{x}_{(K:N)}\}$.
> 8: $\quad\quad$ Compute log-rank weights $w_k \leftarrow \frac{\ln(K+0.5)-\ln(k)}{\sum_{j=1}^K (\ln(K+0.5)-\ln(j))}$ for $k = 1, \dots, K$.
> 9: $\quad\quad$ Compute weighted mean $\boldsymbol{\mu}_w \leftarrow \sum_{k=1}^K w_k \mathbf{x}_{(k:N)}$ and covariance $\mathbf{C}_w \leftarrow \sum_{k=1}^K w_k (\mathbf{x}_{(k:N)} - \boldsymbol{\mu}_w)(\mathbf{x}_{(k:N)} - \boldsymbol{\mu}_w)^T$.
> 10: $\quad\quad$ Apply shrinkage regularization: $\mathbf{C}_{\text{reg}} \leftarrow 0.9999 \mathbf{C}_w + 0.0001 \left(\frac{\text{Tr}(\mathbf{C}_w)}{D}\right) \mathbf{I}_D$.
> 11: $\quad\quad$ Perform eigendecomposition $\mathbf{C}_{\text{reg}} = \mathbf{B} \mathbf{\Lambda} \mathbf{B}^T$ to update orthogonal basis $\mathbf{B}$.
> 12: $\quad$ **end if**
> 13: $\quad$ Initialize $\mathcal{S}_{CR} \leftarrow \emptyset, \mathcal{S}_F \leftarrow \emptyset, \Delta \mathcal{F} \leftarrow \emptyset$, and counters $n_{\text{strat1}}, n_{\text{strat2}}, n_{\text{eig}}, n_{\text{std}} \leftarrow 0$.
> 14: $\quad$ Update elite pool ratio $p^{(t)} \leftarrow 0.20 - (0.20 - 0.04) \cdot (t / T_{\max})$ and pool size $K_p^{(t)} \leftarrow \max(2, \lfloor N \cdot p^{(t)} \rfloor)$.
> 15: $\quad$ **for** $i = 1$ **to** $N$ **do**
> 16: $\quad\quad$ Sample memory index $r \sim \mathcal{U}\{1, \dots, H\}$.
> 17: $\quad\quad$ Generate $CR_i \leftarrow \text{clip}(\mathcal{N}(M_{CR, r}, 0.1^2), 0, 1)$ and $F_i \sim \mathcal{C}(M_{F, r}, 0.1)$ truncated to $(0, 1]$.
> 18: $\quad\quad$ Select $\mathbf{x}_{p\text{best}}$ randomly from top $K_p^{(t)}$ elites, and select distinct $\mathbf{x}_{r_1} \in \mathcal{P}^{(t)} \setminus \{\mathbf{x}_i\}$.
> 19: $\quad\quad$ **if** $\text{rand}(0, 1) < p_{\text{strat1}}$ **then** $\quad$ *// Operator 2: Heterogeneous Dual-Strategy Mutation*
> 20: $\quad\quad\quad$ Select $\tilde{\mathbf{x}}_{r_2} \in (\mathcal{P}^{(t)} \cup \mathcal{A}) \setminus \{\mathbf{x}_i, \mathbf{x}_{r_1}\}$ and compute $\mathbf{v}_i \leftarrow \mathbf{x}_i + F_i(\mathbf{x}_{p\text{best}} - \mathbf{x}_i) + F_i(\mathbf{x}_{r_1} - \tilde{\mathbf{x}}_{r_2})$.
> 21: $\quad\quad$ **else**
> 22: $\quad\quad\quad$ Select $\mathbf{x}_{r_2} \in \mathcal{P}^{(t)} \setminus \{\mathbf{x}_i\}$, draw $\boldsymbol{\delta}_i \sim \mathcal{C}(0, 0.05)^D$, and compute $\mathbf{v}_i \leftarrow \mathbf{x}_{p\text{best}} + F_i(\mathbf{x}_{r_1} - \mathbf{x}_{r_2}) + \boldsymbol{\delta}_i \odot (\mathbf{x}_{r_1} - \mathbf{x}_i)$.
> 23: $\quad\quad$ **end if**
> 24: $\quad\quad$ Enforce boundary constraints on $\mathbf{v}_i$ and draw forced index $j_{\text{rand}} \sim \mathcal{U}\{1, \dots, D\}$.
> 25: $\quad\quad$ **if** $\text{rand}(0, 1) < p_{\text{eigen}}$ **then** $\quad$ *// Operator 3: Landscape-Aware Eigen-Subspace Crossover*
> 26: $\quad\quad\quad$ Project $\hat{\mathbf{v}}_i \leftarrow \mathbf{B}^T \mathbf{v}_i$, $\hat{\mathbf{x}}_i \leftarrow \mathbf{B}^T \mathbf{x}_i$, perform binomial crossover on $\hat{\mathbf{v}}_i, \hat{\mathbf{x}}_i$ with $CR_i, j_{\text{rand}}$ to obtain $\hat{\mathbf{u}}_i$, and reconstruct $\mathbf{u}_i \leftarrow \mathbf{B}\hat{\mathbf{u}}_i$.
> 27: $\quad\quad$ **else**
> 28: $\quad\quad\quad$ Perform standard Cartesian binomial crossover on $\mathbf{v}_i, \mathbf{x}_i$ with $CR_i, j_{\text{rand}}$ to obtain $\mathbf{u}_i$.
> 29: $\quad\quad$ **end if**
> 30: $\quad\quad$ Enforce boundary constraints on $\mathbf{u}_i$ and evaluate $f(\mathbf{u}_i)$.
> 31: $\quad\quad$ **if** $f(\mathbf{u}_i) < f(\mathbf{x}_i)$ **then** $\quad$ *// Greedy Selection & Archive Update*
> 32: $\quad\quad\quad$ Append $CR_i \to \mathcal{S}_{CR}$, $F_i \to \mathcal{S}_F$, $(f(\mathbf{x}_i) - f(\mathbf{u}_i)) \to \Delta \mathcal{F}$, and increment corresponding win counters $(n_{\text{strat1}}\text{ or }n_{\text{strat2}})$ and $(n_{\text{eig}}\text{ or }n_{\text{std}})$.
> 33: $\quad\quad\quad$ Insert $\mathbf{x}_i$ into archive $\mathcal{A}$ (replacing a random member if $\vert{}\mathcal{A}\vert{} \ge \lfloor r_{\text{arc}} N \rfloor$), and update $\mathbf{x}_i \leftarrow \mathbf{u}_i$.
> 34: $\quad\quad$ **end if**
> 35: $\quad$ **end for**
> 36: $\quad$ Update $p_{\text{strat1}} \leftarrow \text{clip}\left(0.85 p_{\text{strat1}} + 0.15 \frac{n_{\text{strat1}}}{n_{\text{strat1}} + n_{\text{strat2}}}, 0.10, 0.90\right)$ and $p_{\text{eigen}} \leftarrow \text{clip}\left(0.85 p_{\text{eigen}} + 0.15 \frac{n_{\text{eig}}}{n_{\text{eig}} + n_{\text{std}}}, 0.05, 0.95\right)$. $\quad$ *// Operator 4*
> 37: $\quad$ **if** $\mathcal{S}_{CR} \neq \emptyset$ and $\sum \Delta \mathcal{F} > 10^{-12}$ **then**
> 38: $\quad\quad$ Update $M_{CR, k_{\text{mem}}} \leftarrow \text{Mean}_{WA}(\mathcal{S}_{CR}, \Delta \mathcal{F})$, $M_{F, k_{\text{mem}}} \leftarrow \text{Mean}_{WL}(\mathcal{S}_F, \Delta \mathcal{F})$, and $k_{\text{mem}} \leftarrow (k_{\text{mem}} \bmod H) + 1$.
> 39: $\quad$ **end if**
> 40: $\quad$ **Sort** $\mathcal{P}^{(t)}$ in ascending order of objective fitness.
> 41: **end for**
> 42: **return** $\mathbf{x}_{\text{best}} = \mathbf{x}_{(1:N)}^{(T_{\max})}$ and $f(\mathbf{x}_{\text{best}})$.


### 四、 源代码
```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import mealpy
from mealpy import Optimizer, FloatVar
from mealpy.swarm_based import PSO
from mealpy.evolutionary_based import DE, SHADE

class EADS(Optimizer):
    def __init__(self, epoch=300, pop_size=50, H=5, arc_rate=1.0, p_best_rate=0.11, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.H = H
        self.arc_rate = arc_rate
        self.p_best_rate = p_best_rate
        self.sort_flag = True  

    def initialize_variables(self):
        self.dim = self.problem.n_dims
        self.M_CR = np.full(self.H, 0.5)
        self.M_F = np.full(self.H, 0.5)
        self.k_mem = 0
        self.archive = []
        self.max_archive_size = int(self.pop_size * self.arc_rate)
        self.p_strat1 = 0.5
        self.p_eigen = 0.5
        self.B_matrix = np.eye(self.dim)

    def _update_eigen_basis(self):
        n_elites = max(self.dim + 1, int(0.35 * self.pop_size))
        elites = np.array([agent.solution for agent in self.pop[:n_elites]])
        
        # ---------------- [改进 2-A: 对数秩加权协方差矩阵] ----------------
        weights = np.log(n_elites + 0.5) - np.log(np.arange(1, n_elites + 1))
        weights = weights / np.sum(weights)
        mean_elite = np.sum(elites * weights[:, np.newaxis], axis=0)
        diff = elites - mean_elite
        cov = np.dot((diff * weights[:, np.newaxis]).T, diff)
        # ------------------------------------------------------------------
        
        trace_m = np.trace(cov) / self.dim if self.dim > 0 else 1.0
        cov_reg = 0.9999 * cov + 0.0001 * trace_m * np.eye(self.dim)
        try:
            _, eigen_vectors = np.linalg.eigh(cov_reg)
            self.B_matrix = eigen_vectors
        except np.linalg.LinAlgError:
            self.B_matrix = np.eye(self.dim)

    def evolve(self, epoch):
        if epoch % 5 == 0:
            self._update_eigen_basis()
            
        S_CR, S_F, delta_f = [], [], []
        strat1_wins, strat2_wins = 0, 0
        eigen_wins, standard_wins = 0, 0
        
        # ---------------- [改进 2-B: 动态 p-best 精英池线性缩减] ----------------
        p_rate = 0.20 - (0.20 - 0.04) * (epoch / self.epoch)
        p_count = max(2, int(self.pop_size * p_rate))
        # ------------------------------------------------------------------------
        
        for i in range(self.pop_size):
            r = self.generator.integers(0, self.H)
            cr = float(np.clip(self.generator.normal(self.M_CR[r], 0.1), 0.0, 1.0))
            while True:
                f_val = float(self.generator.standard_cauchy() * 0.1 + self.M_F[r])
                if f_val > 0:
                    f_val = min(1.0, f_val)
                    break
                    
            pbest_idx = self.generator.integers(0, p_count)
            x_pbest = self.pop[pbest_idx].solution
            cands_r1 = [idx for idx in range(self.pop_size) if idx != i]
            r1 = self.generator.choice(cands_r1)
            x_r1 = self.pop[r1].solution
            
            use_strat1 = (self.generator.uniform() < self.p_strat1)
            if use_strat1:
                # 策略 1 (开采): current-to-pbest/1 with archive
                pool = list(range(self.pop_size))
                if len(self.archive) > 0:
                    pool += [self.pop_size + idx for idx in range(len(self.archive))]
                cands_r2 = [idx for idx in pool if idx != i and idx != r1]
                r2 = self.generator.choice(cands_r2)
                x_r2 = self.pop[r2].solution if r2 < self.pop_size else self.archive[r2 - self.pop_size]
                v = self.pop[i].solution + f_val * (x_pbest - self.pop[i].solution) + f_val * (x_r1 - x_r2)
            else:
                # 策略 2 (勘探): 精英基向量 + 柯西重尾扰动
                r2 = self.generator.choice(cands_r1)
                x_r2 = self.pop[r2].solution
                cauchy_step = self.generator.standard_cauchy(self.dim) * 0.05
                v = x_pbest + f_val * (x_r1 - x_r2) + cauchy_step * (x_r1 - self.pop[i].solution)
                
            v = self.correct_solution(v)
            
            # 自适应特征子空间交叉
            use_eigen = (self.generator.uniform() < self.p_eigen)
            if use_eigen:
                v_prime = np.dot(self.B_matrix.T, v)
                x_prime = np.dot(self.B_matrix.T, self.pop[i].solution)
                mask = self.generator.uniform(size=self.dim) < cr
                mask[self.generator.integers(0, self.dim)] = True
                u_prime = np.where(mask, v_prime, x_prime)
                u = np.dot(self.B_matrix, u_prime)
            else:
                mask = self.generator.uniform(size=self.dim) < cr
                mask[self.generator.integers(0, self.dim)] = True
                u = np.where(mask, v, self.pop[i].solution)
                
            u = self.correct_solution(u)
            new_agent = self.generate_agent(u)
            
            if self.compare_target(new_agent.target, self.pop[i].target, self.problem.minmax):
                diff = abs(self.pop[i].target.fitness - new_agent.target.fitness)
                delta_f.append(diff)
                S_CR.append(cr)
                S_F.append(f_val)
                if use_strat1: strat1_wins += 1
                else: strat2_wins += 1
                if use_eigen: eigen_wins += 1
                else: standard_wins += 1
                
                if self.max_archive_size > 0:
                    if len(self.archive) < self.max_archive_size:
                        self.archive.append(self.pop[i].solution.copy())
                    else:
                        rep_idx = self.generator.integers(0, self.max_archive_size)
                        self.archive[rep_idx] = self.pop[i].solution.copy()
                self.pop[i] = new_agent
                
        # 概率自适应滑动更新
        if (strat1_wins + strat2_wins) > 0:
            self.p_strat1 = 0.85 * self.p_strat1 + 0.15 * (strat1_wins / (strat1_wins + strat2_wins))
            self.p_strat1 = np.clip(self.p_strat1, 0.1, 0.9)
            
        if (eigen_wins + standard_wins) > 0:
            self.p_eigen = 0.85 * self.p_eigen + 0.15 * (eigen_wins / (eigen_wins + standard_wins))
            self.p_eigen = np.clip(self.p_eigen, 0.05, 0.95)
            
        # Lehmer 均值更新记忆池
        if len(S_CR) > 0:
            sum_df = np.sum(delta_f)
            if sum_df > 1e-12:
                w = np.array(delta_f) / sum_df
                self.M_CR[self.k_mem] = np.sum(w * np.array(S_CR))
                self.M_F[self.k_mem] = np.sum(w * (np.array(S_F)**2)) / np.sum(w * np.array(S_F))
                self.k_mem = (self.k_mem + 1) % self.H

# ==============================================================================
# test
# ==============================================================================
dim = 10
epoch = 600
pop_size = 50  # D=10, 50, D=30, 100, D=60, 200
n_runs = 3
np.random.seed(42)

o1 = np.random.uniform(-30, 30, dim)
o2 = np.random.uniform(-5, 5, dim)
o3 = np.random.uniform(-2.5, 2.5, dim)
o4 = np.random.uniform(-10, 10, dim)
o5 = np.random.uniform(-2.5, 2.5, dim)
o6 = np.random.uniform(-10, 10, dim)
o7 = np.random.uniform(-150, 150, dim)
o8 = np.random.uniform(-2, 2, dim)

H1 = np.random.randn(dim, dim); Q1, _ = np.linalg.qr(H1)
H2 = np.random.randn(dim, dim); Q2, _ = np.linalg.qr(H2)
H3 = np.random.randn(dim, dim); Q3, _ = np.linalg.qr(H3)

def f1_shifted_sphere(x):
    z = x - o1
    return float(np.sum(z**2))

def f2_shifted_rosenbrock(x):
    z = x - o2 + 1.0
    return float(np.sum(100.0 * (z[1:] - z[:-1]**2.0)**2.0 + (1 - z[:-1])**2.0))

def f3_shifted_rastrigin(x):
    z = x - o3
    return float(10.0 * dim + np.sum(z**2.0 - 10.0 * np.cos(2.0 * np.pi * z)))

def f4_shifted_ackley(x):
    z = x - o4
    d = len(z)
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / d)) - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / d) + 20.0 + np.e)

def f5_shifted_rot_rastrigin(x):
    z = np.dot(x - o5, Q1)
    return float(10.0 * dim + np.sum(z**2.0 - 10.0 * np.cos(2.0 * np.pi * z)))

def f6_shifted_rot_ackley(x):
    z = np.dot(x - o6, Q2)
    d = len(z)
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / d)) - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / d) + 20.0 + np.e)

def f7_shifted_griewank(x):
    z = x - o7
    sum_part = np.sum(z**2) / 4000.0
    prod_part = np.prod(np.cos(z / np.sqrt(np.arange(1, dim + 1))))
    return float(sum_part - prod_part + 1.0)

def f8_shifted_rot_hybrid(x):
    z = np.dot(x - o8, Q3) + 1.0
    rosen = 100.0 * (z[1:] - z[:-1]**2.0)**2.0 + (1 - z[:-1])**2.0
    return float(np.sum(rosen**2 / 4000.0 - np.cos(rosen) + 1.0))

BENCHMARKS = [
    {"name": "F1: Shifted Sphere", "func": f1_shifted_sphere, "lb": -100.0, "ub": 100.0},
    {"name": "F2: Shifted Rosenbrock", "func": f2_shifted_rosenbrock, "lb": -30.0, "ub": 30.0},
    {"name": "F3: Shifted Rastrigin", "func": f3_shifted_rastrigin, "lb": -5.12, "ub": 5.12},
    {"name": "F4: Shifted Ackley", "func": f4_shifted_ackley, "lb": -32.768, "ub": 32.768},
    {"name": "F5: Shifted Rot Rastrigin", "func": f5_shifted_rot_rastrigin, "lb": -5.12, "ub": 5.12},
    {"name": "F6: Shifted Rot Ackley", "func": f6_shifted_rot_ackley, "lb": -32.768, "ub": 32.768},
    {"name": "F7: Shifted Griewank", "func": f7_shifted_griewank, "lb": -600.0, "ub": 600.0},
    {"name": "F8: Shifted Rot Hybrid", "func": f8_shifted_rot_hybrid, "lb": -10.0, "ub": 10.0},
]


if __name__ == "__main__":
    algorithms = {
        "PSO (Mealpy)": lambda: PSO.OriginalPSO(epoch=epoch, pop_size=pop_size),
        "DE (Mealpy)": lambda: DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=0.5, cr=0.9),
        "SHADE (Mealpy)": lambda: SHADE.OriginalSHADE(epoch=epoch, pop_size=pop_size, miu_f=0.5, miu_cr=0.5),
        "L-SHADE (Mealpy)": lambda: SHADE.OriginalL_SHADE(epoch=epoch, pop_size=pop_size, miu_f=0.5, miu_cr=0.5),
        "EADS (Proposed)": lambda: EADS(epoch=epoch, pop_size=pop_size, H=5, arc_rate=1.0)
    }

    results_summary = {}
    all_curves = {}

    print("\n" + "="*110)
    print(f"  8 test functions, D = {dim}, run times = {n_runs}, Epoch = {epoch})")
    print("="*110)

    for bm in BENCHMARKS:
        fname = bm["name"]
        print(f"\n>>> running: {fname}")
        results_summary[fname] = {}
        all_curves[fname] = {}
        
        prob_dict = {
            "obj_func": bm["func"],
            "bounds": FloatVar(lb=[bm["lb"]]*dim, ub=[bm["ub"]]*dim, name="delta"),
            "minmax": "min",
            "log_to": None,
        }

        for alg_name, model_builder in algorithms.items():
            run_fits = []
            run_curves = []
            for r in range(n_runs):
                model = model_builder()
                g_best = model.solve(prob_dict, seed=r*100+7)
                run_fits.append(g_best.target.fitness)
                run_curves.append(model.history.list_global_best_fit)
                
            run_fits = np.array(run_fits)
            mean_curve = np.mean(np.array(run_curves), axis=0)
            results_summary[fname][alg_name] = {"mean": np.mean(run_fits), "std": np.std(run_fits)}
            all_curves[fname][alg_name] = mean_curve
            print(f"  [{alg_name:18s}] 适应度均值: {np.mean(run_fits):.3e} | 标准差: {np.std(run_fits):.3e}")

    print("\n" + "="*115)
    print(f"{'function':<28} | {'PSO':<12} | {'DE':<12} | {'SHADE':<12} | {'L-SHADE':<12} | {'EADS':<12}")
    print("="*115)
    for bm in BENCHMARKS:
        fn = bm['name']
        p_str = f"{results_summary[fn]['PSO (Mealpy)']['mean']:.2e}"
        d_str = f"{results_summary[fn]['DE (Mealpy)']['mean']:.2e}"
        s_str = f"{results_summary[fn]['SHADE (Mealpy)']['mean']:.2e}"
        ls_str = f"{results_summary[fn]['L-SHADE (Mealpy)']['mean']:.2e}"
        e_str = f"{results_summary[fn]['EADS (Proposed)']['mean']:.2e}"
        print(f"{fn:<28} | {p_str:<12} | {d_str:<12} | {s_str:<12} | {ls_str:<12} | {e_str:<12}")
    print("="*115)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    colors = {
        "PSO (Mealpy)": "#7f7f7f", "DE (Mealpy)": "#1f77b4", 
        "SHADE (Mealpy)": "#ff7f0e", "L-SHADE (Mealpy)": "#9467bd",
        "EADS (Proposed)": "#d62728"
    }
    linestyles = {
        "PSO (Mealpy)": ":", "DE (Mealpy)": "--", 
        "SHADE (Mealpy)": "-.", "L-SHADE (Mealpy)": "--",
        "EADS (Proposed)": "-"
    }
    x_epochs = np.arange(1, epoch + 1)

    for idx, bm in enumerate(BENCHMARKS):
        ax = axes[idx]
        fname = bm["name"]
        for alg_name in algorithms.keys():
            curve = all_curves[fname][alg_name]
            plot_curve = np.maximum(curve, 1e-16)
            ax.plot(x_epochs, plot_curve, label=alg_name, color=colors[alg_name],
                    linestyle=linestyles[alg_name], linewidth=2.2 if "EADS" in alg_name else 1.4)
        ax.set_yscale('log')
        ax.set_title(fname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch (Iterations)', fontsize=9)
        ax.set_ylabel('Best Fitness (Log Scale)', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    output_filename = f"mealpy_shifted_rotated_8benchmarks_dim{dim}_imp2.png"
    # plt.savefig(output_filename, dpi=300)
    plt.show()
```