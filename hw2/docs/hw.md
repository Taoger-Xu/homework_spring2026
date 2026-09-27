# 作业 2：策略梯度

本文整理并翻译作业第 3～6 节的重要内容。实现时请重点检查
`src/agents/pg_agent.py` 中标记为 `TODO` 的部分；实验命令应在
`hw2` 目录下运行。

## 3. Vanilla Policy Gradients

### 3.1 实现要求

你需要在 `pg_agent.py` 中实现两种回报（return）估计方法。

#### 情况 1：整条轨迹的折扣累计回报

`calculate_q_vals` 中的 Case 1 使用整条轨迹的折扣累计回报，对应
vanilla policy gradient：

$$
R(\tau^i)=\sum_{t'=0}^{H-1}\gamma^{t'}r(s_{t'}^i,a_{t'}^i).
$$

同一条轨迹中的每个时间步都使用这个相同的总回报。

#### 情况 2：Reward-to-go

Case 2 使用从当前时间步开始计算的 reward-to-go：

$$
Q_t^i=\sum_{t'=t}^{H-1}\gamma^{t'-t}r(s_{t'}^i,a_{t'}^i).
$$

除了这两种回报估计方法，还需要完成代码中其余标记为 `TODO` 的部分。
在本节的小规模实验中，可以暂时跳过仅在 `use_baseline=True` 时才执行的代码；
baseline 将在第 4 节实现。

### 3.2 实验 1：CartPole

在离散动作环境 `CartPole-v0` 上运行以下 8 组实验：

```bash
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  --exp_name cartpole
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  -rtg --exp_name cartpole_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  -na --exp_name cartpole_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  -rtg -na --exp_name cartpole_rtg_na

uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
  --exp_name cartpole_lb
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
  -rtg --exp_name cartpole_lb_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
  -na --exp_name cartpole_lb_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
  -rtg -na --exp_name cartpole_lb_rtg_na
```

关键参数：

- `-n`：策略梯度迭代次数。
- `-b`：batch size，即每次迭代使用当前策略采集的状态—动作对数量。
- `-rtg`：启用 reward-to-go；不提供时默认为关闭。
- `-na`：启用 advantage normalization，把一个 batch 内的 advantage
  标准化为均值 0、标准差 1。
- `--exp_name`：实验名称，也用于数据日志目录名。

其他命令行参数还可以控制学习率、网络结构和 batch size 等设置。

#### 需要提交

- 上述 8 组实验的日志。无论小 batch 还是大 batch，表现最好的配置都应收敛到
  最大回报 200。
- 两张图：
  - 比较不含 `lb` 前缀的 4 组小 batch 实验；
  - 比较含 `cartpole_lb` 前缀的 4 组大 batch 实验。
- 两张图均绘制“平均回报 vs. 环境步数”。横轴必须使用日志指标
  `Train_EnvstepsSoFar`，不能使用策略梯度迭代次数。
- 简要回答：
  1. 不使用 advantage normalization 时，整条轨迹回报与 reward-to-go
     哪一种表现更好？
  2. 为什么通常更偏好其中一种估计方法？
  3. advantage normalization 是否有帮助？
  4. batch size 是否造成了影响？
- 给出实际运行实验的完整命令，包括所有相对默认值做出的修改。

## 4. 使用神经网络 Baseline

### 4.1 实现要求

现在需要实现一个依赖状态的神经网络价值函数，作为 baseline。为此，需要补全
第 3 节中暂时跳过的 `TODO`：

- 在 `PGAgent.update` 中，baseline 网络与 policy gradient 更新一起训练。
- 在 `pg_agent.py` 的 `estimate_advantage` 中，用 reward-to-go 减去
  baseline 的预测，得到 advantage 估计：

$$
A_t^i=
\left(\sum_{t'=t}^{H-1}\gamma^{t'-t}r(s_{t'}^i,a_{t'}^i)\right)
-V_\phi^\pi(s_t^i).
$$

- 每次 policy 更新时，对 baseline 网络执行多次梯度更新；更新次数由
  `baseline_gradient_steps` 决定。

### 4.2 实验 2：HalfCheetah

使用带 baseline 的 policy gradient，在 `HalfCheetah-v4` 上学习控制器：

```bash
# 不使用 baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
  --discount 0.95 -lr 0.01 --exp_name cheetah

# 使用 baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
  --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 \
  --exp_name cheetah_baseline
```

这里有意没有使用 `-na`。在实际中，advantage normalization 是一个很有效的
技巧；对于本作业测试的大多数简单环境，它甚至会让 baseline 显得没有必要。

#### 需要提交

- 上述两组实验的日志。
- 带 baseline 的版本在训练结束时应达到 300 以上的平均回报。不同随机种子会
  产生波动；正确实现也可能偶尔出现较差结果，但通常不应需要超过 2～3 次运行
  才达到该阈值，否则应重新检查实现。
- 两张图：
  - baseline loss 的学习曲线；
  - evaluation return 的学习曲线。
- 再运行一组实验，降低 baseline 梯度步数 `-bgs` 或 baseline 学习率 `-blr`，
  并说明它对以下两项的影响：
  1. baseline 的学习曲线；
  2. policy 的表现。
- 给出所有实验的完整命令，包括所有相对默认值做出的修改。

可选：重新加入 `-na`，观察性能提升；还可以把 `video_log_freq` 设为 10，
在 WandB 中查看 HalfCheetah 行走的视频。

## 5. 实现 Generalized Advantage Estimation（GAE）

使用上一节实现的价值函数，实现一个简化版的 GAE-$\lambda$。需要补全
`pg_agent.py` 中 `estimate_advantage` 的最后一个 `TODO`。

### 实验 3：LunarLander

使用带 GAE 的 policy gradient，在动作带噪声的 `LunarLander-v2` 上学习控制器。
固定其他超参数不变，仅搜索：

$$
\lambda\in\{0, 0.95, 0.98, 0.99, 1\}.
$$

把下面命令中的 `<lambda>` 替换为相应数值，共运行 5 组实验：

```bash
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go \
  --use_baseline --gae_lambda <lambda> \
  --exp_name lunar_lander_lambda<lambda>
```

不要修改 batch size、学习率等其他超参数。

#### 需要提交

- 上述 5 个 $\lambda$ 取值对应的实验日志。
- 最好的一次实验应在训练过程中至少一次达到 150 以上的平均回报。结果存在
  随机波动，必要时可以重新运行。
- 用一张图绘制全部 LunarLander 实验的学习曲线，并用文字描述 $\lambda$
  对任务表现的影响。
- 用一两句话回答：$\lambda=0$ 和 $\lambda=1$ 分别对应什么？并把这一点与
  LunarLander 的实验表现联系起来。
- 给出所有实验的完整命令，包括所有相对默认值做出的修改。

## 6. 超参数与样本效率

Policy gradient 的一个常见问题是样本效率较低：可能要与环境交互几十万乃至
数百万步才能学到良好策略。虽然提高样本效率整体上仍是开放问题，但合理选择
超参数通常能显著改善结果。

训练中需要考虑的主要设置包括：

1. 折扣因子（discount factor）；
2. 网络大小；
3. batch size：太小会造成高方差，太大则可能浪费样本，因为每次梯度更新后
   都必须重新采集整个 batch；
4. 学习率；
5. 是否使用 reward-to-go；
6. 是否对 advantage 做标准化；
7. 是否使用 GAE，以及使用何种 $\lambda$。

### 实验 4：InvertedPendulum

默认配置如下：

```bash
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 \
  --exp_name pendulum
```

你的任务是调节超参数，使算法用比默认配置更少的环境步数达到最大表现，即回报
1000。注意，目标不是最小化 policy gradient 的迭代次数，因为一次迭代会消耗
约一个 batch 的环境步数。

具体目标：**在 100,000 个环境步以内达到回报 1000。**

你可以修改前面列出的任意超参数，也可以修改其他认为相关的参数。实验名可以
自行指定，但必须以 `pendulum` 开头，以便识别。

#### 需要提交

- 表现最佳的一次运行日志；它必须在 100K 环境步以内至少一次达到平均回报
  1000。
- 最佳超参数组合及其完整运行命令，并简要讨论调参过程中哪些超参数最重要。
- 在同一组学习曲线中比较最佳配置与默认配置的平均回报，横轴使用环境步数。

由于 RL 结果会随随机种子变化，正式研究通常应报告多个随机种子的平均结果，
以确保统计显著性并避免只挑选最好结果。不过，为降低本作业的计算负担，这里
只要求提交表现最好的一次运行。

## 最终检查清单

- [ ] `pg_agent.py` 中相关 `TODO` 已完成。
- [ ] CartPole 的 8 份日志、2 张图和 4 个简答题已准备好。
- [ ] HalfCheetah 的基础/带 baseline 日志及额外消融实验已完成。
- [ ] HalfCheetah 的 baseline loss 与 eval return 曲线已准备好。
- [ ] LunarLander 的 5 个 $\lambda$ 实验和汇总图已完成。
- [ ] InvertedPendulum 在 100K 环境步内达到回报 1000，并与默认配置对比。
- [ ] 每组实验都记录了实际使用的完整命令。
