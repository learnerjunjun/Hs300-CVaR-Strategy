# Pre

By 11.12

## To-Do

- [x] 数据集更新

- [x] 历史数据分布拟合

  summary: 从4848只股票中随机选取5个股票的收益分布并非正态分布

  
  
- [ ] 建模

- [ ] 收益预测

- [ ] 简单训练预测

- [ ] 是否需要收益率矩阵的概率分布拟合

## Model

### Target

- 加入预测数据

  - 利用历史数据，训练收益预测模型；利用未来一段时间的预测数据，拟合概率分布

  - 给定一个预期收益率，计算最优化权重

- 加入CVaR

  - 由价值损失函数，计算Loss

  - 计算1%置信区间下的VaR，再计算CVaR

  - 在该权重下，使Loss < CVaR的概率为99%

- 加入投资者观点及定价估计偏差

### Hypothesis

- Black-Scholes Model


* CAPM
  - Markowitz
  - Sharpe
* Necessary
  * 单期模型

  * Market
    * 价格随机游走
    * 股票不支付股利等
    * 无交易成本
    * 无风险利率借贷
    * 允许卖空

  * Investor
    * Max Utility
    * 效用函数非线性
    * 二阶随机占优


### 建模_1

* Markowitz(1952)的提出的MV模型是一个静态优化模型，资产收益率是给定的、使用方差度量风险；

  基于Mean-Variance模型，我想改进收益率及风险的度量方式：即使用机器学习预测收益率，添加CVaR约束，并引入Black-Litterman模型来考虑投资者的观点，并生成最终的资产配置。

* 论文题目：基于机器学习和Black-Litterman模型的投资组合优化

* CAPM的模型假设为本文的主要假设：无风险利率借贷、风险厌恶等

  简化的数学模型如下：

  目标函数：

  $\text{max}\ \ w^Tμ - λ * CVaR_α(w)$  

  $\text{s.t.}$  

  $\sum w = 1$  
  $w ≥ 0$  
  $w^Tμ ≥ r_{min}$  
  $Pr(w^TR_{pred} ≤ -CVaR_α(w)) ≤ 1 - α$ 
  $w = (\sumτ)^{(-1)} (μ - π)$

  其中：

  * $R$：N x T的矩阵，表示历史资产收益率数据。
  * $R_{pred}$：N x T的矩阵，表示使用机器学习模型预测的未来资产收益率数据。
  * $w$：N x 1的向量，表示投资组合中每个资产的权重。
  * $μ$：N x 1的向量，表示资产的预期收益率。
  * Σ：N x N的协方差矩阵，表示资产之间的相关性和方差。
  * $λ$是风险规避系数，用于平衡收益和风险之间的权衡。
  * $CVaR_{α}(w)$是在置信水平α下的条件风险价值（Conditional Value-at-Risk），表示在最坏情况下的预期损失。
  * $r_{min}$是投资组合的最低收益率要求。
  * $Pr(w^TR_pred ≤ -CVaR_α(w)) ≤ 1 - α$是风险的约束条件，表示在投资组合收益率低于CVaR时的置信水平。
  * τ是Black-Litterman模型中的定价误差参数。
  * π是投资者的观点向量，表示对资产预期收益率的观点。

* 可能存在的问题

  求解可能比较复杂，这时可以适当不考虑Black-Litterman模型
  
### 解析法求解

#### Info_set

- **收益率矩阵 **$N\times T$
  $$
  R=
  \begin{bmatrix}
  r_1^1 & r_1^2 & r_1^3 & ... & r_1^T\\
  r_2^1 & r_2^2 & r_2^3 & ... & r_2^T\\
  r_3^1 & r_3^2 & r_3^3 & ... & r_3^T\\
  ... & ... & ... & ... & ...\\
  r_N^1 & r_N^2 & r_N^3 & ... & r_N^T\\
  \end{bmatrix}
  =
  \begin{bmatrix}
  R_1 & R_2 & R_3 &... & R_T
  \end{bmatrix}
  $$
  *Predicted:*
  $$
  \widehat{R}=
  \begin{bmatrix}
  \widehat{r}_1^1 & \widehat{r}_1^2 & \widehat{r}_1^3 & ... & \widehat{r}_1^T\\
  \widehat{r}_2^1 & \widehat{r}_2^2 & \widehat{r}_2^3 & ... & \widehat{r}_2^T\\
  \widehat{r}_3^1 & \widehat{r}_3^2 & \widehat{r}_3^3 & ... & \widehat{r}_3^T\\
  ... & ... & ... & ... & ...\\
  \widehat{r}_N^1 & \widehat{r}_N^2 & \widehat{r}_N^3 & ... & \widehat{r}_N^T\\
  \end{bmatrix}
  =
  \begin{bmatrix}
  \widehat{R}_1 & \widehat{R}_2 & \widehat{R}_3 &... & \widehat{R}_T
  \end{bmatrix}
  $$
  
- **权重** $N \times 1$
  $$
  w=\begin{bmatrix}
  w_1 & w_2 & w_3 & ... & w_N
  \end{bmatrix}
  $$

- **期望收益率矩阵**
  $$
  \mu=\begin{bmatrix}
  \mu_1 & \mu_2 & \mu_3 & ... & \mu_N
  \end{bmatrix}
  $$

- **协方差矩阵**
  $$
  \Sigma=
  \begin{bmatrix}
  \sigma_{11} & \sigma_{12} & \sigma_{13} & ... & \sigma_{1N}\\
  \sigma_{21} & \sigma_{22} & \sigma_{23} & ... & \sigma_{2N}\\
  \sigma_{31} & \sigma_{32} & \sigma_{33} & ... & \sigma_{3N}\\
  ... & ... & ... & ... & ...\\
  \sigma_{N1} & \sigma_{N2} & \sigma_{N3} & ... & \sigma_{NN}\\
  \end{bmatrix}
  $$
  
- **Loss Function**
  $$
  L(w,R)=-w^TR
  $$
  Joint probability density function of $R$ is $P(R)$:

  Distribution function of $L(w,R)$ is 

  $\varphi(\lambda)=P\{L(w,R) \leq \lambda\}$

  $\varphi(\lambda)=\int P(R)dR$

- **VaR and C-VaR**

  置信水平 $\alpha$ 

  $VaR$ ：在险价值 WR(Value—a t—Risk)是指在一定的置信水平下，证券组合在正常的市场波动下，在未来一段特定时间内的最大可能损失；
  
  为$L(w,R)$ 的 $\alpha$ 分位数
  $$
  VaR_{\alpha}=inf\{\lambda \in \R \vert \varphi(\lambda) \geq \alpha \}
  $$
  $CVaR$ 为 $L(w,R)$ 不小于 $VaR_{\alpha}$ 条件下的期望损失，具有次可加性；
  $$
  CVaR=E[L \vert L(w,R)\geq VaR_{\alpha}]
  $$

#### 优化问题建模

- **Target Function**

  最大化收益 or 最小化风险损失

  $\text{min} \ \  CVaR$

- **Constraint**

  $\text{s.t.}$




## Return Prediction

* Paper reading
* 补充数据集

  * 1998.04-2022.09 拓展到 2023.10
  * 筛选因子

    * info
    * 数据下载与清洗分类
* 滚动训练周期优化

  * 原始为6个月滚动一次
* 原有模型

  * XGBoost
  * Linear Regression
  * 后续拓展

    *  Deep learning
  * 模型预测效果比较
* 调参速度差异较大
  * 调参之后，整体超额收益率有提升；


## Constraint

* CVAR

## Black-Litterman

* Making decision

## Refer_web

- https://www.value-at-risk.net/value-at-risk/
- https://www.bananaspace.org/wiki/累积分布函数#:~:text=在概率论中%2C%20累积,随机变量的概率分布./
- 
