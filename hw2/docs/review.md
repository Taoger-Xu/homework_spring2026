# 2 回顾

## 2.1 策略梯度

回顾一下，强化学习的目标是学习一个能够最大化目标函数的 $\theta^*$：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta(\tau)}[r(\tau)]. \tag{1}
$$

其中，每条轨迹（rollout）$\tau$ 的长度均为 $H$，其概率为

$$
\begin{aligned}
\pi_\theta(\tau)
&=p(s_0,a_0,\ldots,s_{H-1},a_{H-1})\\
&=p(s_0)\pi_\theta(a_0\mid s_0)
\prod_{t=1}^{H-1}p(s_t\mid s_{t-1},a_{t-1})
\pi_\theta(a_t\mid s_t),
\end{aligned} \tag{2}
$$

并且

$$
r(\tau)=r(s_0,a_0,\ldots,s_{H-1},a_{H-1})
=\sum_{t=0}^{H-1}r(s_t,a_t). \tag{3}
$$

策略梯度方法直接对该目标函数求梯度：

$$
\nabla_\theta J(\theta)
=\nabla_\theta\int\pi_\theta(\tau)r(\tau)\,d\tau, \tag{4}
$$

$$
=\int\pi_\theta(\tau)\nabla_\theta\log\pi_\theta(\tau)r(\tau)\,d\tau, \tag{5}
$$

$$
=\mathbb{E}_{\tau\sim\pi_\theta(\tau)}
[\nabla_\theta\log\pi_\theta(\tau)r(\tau)]. \tag{6}
$$

在实践中，可以使用一批由 $N$ 条采样轨迹组成的数据来近似关于轨迹 $\tau$ 的期望：

$$
\nabla_\theta J(\theta)\approx
\frac{1}{N}\sum_{i=1}^{N}
\nabla_\theta\log\pi_\theta(\tau^i)r(\tau^i), \tag{7}
$$

$$
=\frac{1}{N}\sum_{i=1}^{N}
\left(\sum_{t=0}^{H-1}\nabla_\theta
\log\pi_\theta(a_t^i\mid s_t^i)\right)
\left(\sum_{t=0}^{H-1}r(s_t^i,a_t^i)\right). \tag{8}
$$

由此可见，策略 $\pi_\theta$ 是一个以状态为条件、定义在动作空间上的概率分布。在智能体与环境的交互循环中，智能体从 $\pi_\theta(\cdot\mid s_t)$ 中采样动作 $a_t$，环境则返回奖励 $r(s_t,a_t)$。

## 2.2 方差缩减

### 2.2.1 从当前时刻起的回报（Reward-to-Go）

降低策略梯度方差的一种方法是利用因果关系，即策略不可能影响过去已经获得的奖励。由此可得到下面修改后的目标：此处的奖励之和不包含策略被查询的时间步之前所获得的奖励。这个奖励和是对 $Q$ 函数的一种采样估计，被称为“从当前时刻起的回报”（reward-to-go）。

$$
\nabla_\theta J(\theta)\approx
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{H-1}
\nabla_\theta\log\pi_\theta(a_t^i\mid s_t^i)
\left(\sum_{t'=t}^{H-1}r(s_{t'}^i,a_{t'}^i)\right). \tag{9}
$$

### 2.2.2 折扣

将折扣因子 $\gamma$ 乘到奖励上，可以理解为鼓励智能体更加关注时间上较近的奖励，而较少关注更遥远的未来奖励。这也可以看作一种降低方差的方法，因为考虑越遥远的未来，可能产生的方差就越大。

在实践中，可以通过两种方式引入折扣因子。第一种方式对完整轨迹中的奖励进行折扣：

$$
\nabla_\theta J(\theta)\approx
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{H-1}
\nabla_\theta\log\pi_\theta(a_t^i\mid s_t^i)
\left(\sum_{t'=0}^{H-1}\gamma^{t'}r(s_{t'}^i,a_{t'}^i)\right), \tag{10}
$$

第二种方式对“从当前时刻起的回报”进行折扣：

$$
\nabla_\theta J(\theta)\approx
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{H-1}
\nabla_\theta\log\pi_\theta(a_t^i\mid s_t^i)
\left(\sum_{t'=t}^{H-1}\gamma^{t'-t}r(s_{t'}^i,a_{t'}^i)\right). \tag{11}
$$

虽然这两种形式看起来相似，但它们产生的效果并不相同。

在第一种形式中，折扣是以绝对时间为基准施加的。每个奖励的权重仅取决于它在整条轨迹中出现得有多晚。因此，策略主要关注轨迹的较早部分，并逐渐减少对较晚访问状态的关注。

在第二种形式中，折扣是相对于当前时间步 $t$ 施加的。无论一个状态在轨迹中的什么位置出现，它都会一致地鼓励策略偏好那些能够更快带来奖励的动作。

在实践中，我们几乎总是使用第二种形式（你知道为什么吗？）。

### 2.2.3 基线

另一种降低方差的方法，是从奖励总和中减去一个相对于 $\tau$ 而言为常数的基线：

$$
\nabla_\theta J(\theta)
=\nabla_\theta\mathbb{E}_{\tau\sim\pi_\theta(\tau)}[r(\tau)-b]. \tag{12}
$$

这不会使策略梯度产生偏差，因为

$$
\nabla_\theta\mathbb{E}_{\tau\sim\pi_\theta(\tau)}[b]
=\mathbb{E}_{\tau\sim\pi_\theta(\tau)}
[\nabla_\theta\log\pi_\theta(\tau)\cdot b]=0.
$$

在本次作业中，我们实现一个充当状态相关基线的价值函数 $V_\phi^\pi$。训练该价值函数，使其近似从某个特定状态开始的未来奖励之和：

$$
V_\phi^\pi(s_t)\approx
\sum_{t'=t}^{H-1}\mathbb{E}_{\pi_\theta}
\left[\gamma^{t'-t}r(s_{t'},a_{t'})\mid s_t\right]. \tag{13}
$$

于是，近似策略梯度变为

$$
\nabla_\theta J(\theta)\approx
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{H-1}
\nabla_\theta\log\pi_\theta(a_t^i\mid s_t^i)
\left(
\sum_{t'=t}^{H-1}\gamma^{t'-t}r(s_{t'}^i,a_{t'}^i)
-V_\phi^\pi(s_t^i)
\right). \tag{14}
$$

### 2.2.4 广义优势估计

下式

$$
\left(\sum_{t'=t}^{H-1}\gamma^{t'-t}r(s_{t'},a_{t'})\right)
-V_\phi^\pi(s_t)
$$

可以被解释为对优势函数的一种估计：

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t), \tag{15}
$$

其中，$Q^\pi(s_t,a_t)$ 使用蒙特卡洛回报进行估计，而 $V^\pi(s_t)$ 则使用学得的价值函数 $V_\phi^\pi$ 进行估计。

我们还可以用 $V_\phi^\pi$ 代替蒙特卡洛回报来估计优势，从而进一步降低方差：

$$
A^\pi(s_t,a_t)\approx\delta_t
=r(s_t,a_t)+\gamma V_\phi^\pi(s_{t+1})-V_\phi^\pi(s_t), \tag{16}
$$

其边界情况为
$\delta_{H-1}=r(s_{H-1},a_{H-1})-V_\phi^\pi(s_{H-1})$。不过，由于 $V_\phi^\pi$ 存在建模误差，这种方法会以在策略梯度估计中引入偏差为代价。我们也可以结合 $n$ 步蒙特卡洛回报与 $V_\phi^\pi$ 来估计优势：

$$
A_n^\pi(s_t,a_t)=
\sum_{t'=t}^{t+n-1}\gamma^{t'-t}r(s_{t'},a_{t'})
+\gamma^nV_\phi^\pi(s_{t+n})-V_\phi^\pi(s_t). \tag{17}
$$

增大 $n$ 会使优势估计更多地采用蒙特卡洛回报，从而降低偏差、增大方差；减小 $n$ 则会产生相反的效果。注意，当 $n=H-t$ 时，会恢复无偏但方差较高的蒙特卡洛优势估计；当 $n=1$ 时，则会恢复方差较低但偏差较高的估计 $\delta_t$。

我们可以将多个 $n$ 步优势估计组合成指数加权和，这称为广义优势估计（Generalized Advantage Estimation，GAE）。令 $\lambda\in[0,1]$，则

$$
A_{\mathrm{GAE}}^\pi(s_t,a_t)=
\frac{1-\lambda}{1-\lambda^{H-t-1}}
\sum_{n=1}^{H-t-1}\lambda^{n-1}A_n^\pi(s_t,a_t), \tag{18}
$$

其中，$\frac{1-\lambda}{1-\lambda^{H-t-1}}$ 是归一化常数。注意，较大的 $\lambda$ 会更强调 $n$ 较大的优势估计，较小的 $\lambda$ 则相反。因此，$\lambda$ 用于控制偏差与方差之间的权衡：增大 $\lambda$ 会降低偏差并增大方差。在无限时域（$H=\infty$）的情况下，可以证明

$$
A_{\mathrm{GAE}}^\pi(s_t,a_t)
=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}A_n^\pi(s_t,a_t), \tag{19}
$$

$$
=\sum_{t'=t}^{\infty}(\gamma\lambda)^{t'-t}\delta_{t'}. \tag{20}
$$

为简洁起见，这里省略了推导（详见 [GAE 论文](https://arxiv.org/pdf/1506.02438.pdf)）。

在有限时域情况下，可以写成

$$
A_{\mathrm{GAE}}^\pi(s_t,a_t)
=\sum_{t'=t}^{H-1}(\gamma\lambda)^{t'-t}\delta_{t'}, \tag{21}
$$

这为高效实现广义优势估计器提供了一种方法，因为我们可以递归计算：

$$
A_{\mathrm{GAE}}^\pi(s_t,a_t)
=\delta_t+\gamma\lambda A_{\mathrm{GAE}}^\pi(s_{t+1},a_{t+1}). \tag{22}
$$

## 2.3 优势归一化

策略梯度方法中另一个常用技巧，是将一个批次内的优势归一化，使其均值为 0、标准差为 1。严格来说，这会引入偏差，因为归一化依赖于我们采样到的那一批轨迹。要理解这一点，可以考虑批次大小为 1 的极端情况：此时归一化后的优势始终为 0。不过在实践中，这通常是一种能够有效提升性能的启发式方法；而且当批次足够大时，这种偏差可以忽略不计。

具体而言，优势归一化会将一个批次中的优势估计从

$$
A^\pi(s_t,a_t) \tag{23}
$$

转换为

$$
\frac{A^\pi(s_t,a_t)-\mu}{\sigma+\varepsilon}, \tag{24}
$$

其中，$\mu$ 和 $\sigma$ 分别是一个轨迹批次内优势估计的均值和标准差，$\varepsilon$ 是为了保证数值稳定性而添加到分母中的一个很小的常数（例如 $\varepsilon=10^{-8}$）。
