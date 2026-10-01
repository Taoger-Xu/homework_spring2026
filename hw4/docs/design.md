# HW4 训练流程与代码结构

HW4 是基于 LoRA policy 的语言模型强化学习作业。阅读代码时，可以沿着下面的链路追踪数据：

```text
tasks → rollout → models/logprobs → rl → train → eval / Gradescope
```

其中 `train.py` 负责串联各模块；`models/logprobs.py` 在采样阶段和策略更新阶段都用于计算 token 概率。

## 什么是 LoRA policy

在这个作业中，语言模型就是强化学习的策略（policy）。给定 prompt 和已经生成的 token，模型输出下一个 token 的概率分布 `πθ(a_t | h_t)`；从中采样 token 就是执行一次 action，连续采样形成完整回答。

HW4 的策略由冻结的基座语言模型和可训练的 LoRA adapter 组成。LoRA 在选定的线性层上添加低秩参数，可简化表示为：

```text
实际使用的权重 = 冻结的基座权重 + 可训练的 LoRA 增量
```

LoRA 不是独立生成回答的小模型。它改变基座模型的内部计算，因此即使基座权重不变，整个策略输出的 token 概率也会随着训练改变。`models/load.py` 只让名字含 `lora_` 的参数保留 `requires_grad=True`；`train.py` 也只把这些可训练参数交给优化器。所以 REINFORCE 和 GRPO 的 loss 针对完整策略的输出概率计算，但 `optimizer.step()` 实际更新的是 LoRA 参数。

一次 `format_copy` 训练可以这样理解：

```text
输入“复制整数 42”
→ 基座模型 + 当前 LoRA 生成回答
→ Task 根据完整回答给出 reward
→ REINFORCE / GRPO 根据 reward 和 advantage 计算 loss
→ 反向传播并更新 LoRA
→ 下一次生成时，策略的 token 概率可能发生变化
```

采样和更新使用的**当前 policy** 是基座模型加当前 LoRA。计算 KL 约束时，采样器暂时关闭 LoRA adapter，同一个冻结的基座模型便充当 **reference policy**。两者的区别是是否启用正在训练的 LoRA 参数。

## 目录职责

| 路径 | 作用 |
| --- | --- |
| `hw4/` | 作业根目录，包含依赖配置 `pyproject.toml`、运行说明 `README.md` 和代码包。 |
| `hw4/hw4/` | Python 主包，放置训练、评估和打包逻辑。 |
| `hw4/hw4/tasks/` | `base.py` 定义 `Task` / `TaskExample` 接口；`format_copy.py` 提供整数复制与严格 XML 格式任务；`math_hard.py` 提供 MATH level-5 数学题及 `\boxed{NUMBER}` 答案格式。任务负责生成题目、计算奖励和评估指标。 |
| `hw4/hw4/models/` | `load.py` 加载 Qwen、tokenizer、LoRA policy 和 reference model；`logprobs.py` 计算逐 token log-prob、completion mask 和基于采样 token 的 policy/reference KL 估计。 |
| `hw4/hw4/rollout/` | `hf_sampler.py` 使用 Hugging Face `generate` 采样；`sampler_base.py` 定义采样输出；`rollout_buffer.py` 保存训练字段并按行切分 minibatch。 |
| `hw4/hw4/rl/` | `base.py` 定义共享配置和算法接口；`reinforce.py` 实现单次 on-policy REINFORCE 更新；`grpo.py` 实现带 PPO ratio clipping 的 GRPO 更新；`__init__.py` 导出算法。 |
| `hw4/hw4/utils/` | 答案解析、随机种子、PyTorch 辅助操作，以及 W&B / 本地日志工具。 |
| `hw4/hw4/train.py` | 训练总控：读取配置、加载模型与任务、采样、评分、计算 advantage、更新 policy、记录日志、评估并保存 checkpoint。 |
| `hw4/hw4/eval.py` | 加载指定 checkpoint，独立运行任务评估并输出指标。 |
| `hw4/hw4/config.py` | `TrainConfig` 集中定义 batch/group size、生成长度、学习率、KL 系数和 PPO 参数等超参数。 |
| `hw4/hw4/gradescope_bundle.py` | 收集训练产物并整理为 Gradescope 提交包。 |
| `hw4/scripts/` | 云端运行入口；`modal_train.py` 配置 Modal GPU、Volume，并调用训练、评估和打包命令。 |

## 一次训练更新的数据流

1. `Task.sample_train_batch` 采样一批题目和 prompt。
2. `HFSampler` 让当前 LoRA policy 对每道题生成 `group_size` 个回答，形成 `RolloutOutput`。回答按题目分组、组内连续排列。
3. `Task.reward` 对每个回答评分。`train.py` 将同题回答的奖励转换为组内相对 advantage，并可对整个 rollout 的 advantage 再做标准化。
4. `train.py` 把采样输出与 reward、advantage 合并成 `RolloutBatch`。它包含 `input_ids`、`attention_mask`、`completion_mask`、采样时的 `old_logprobs`、`ref_logprobs`、`rewards` 和 `advantages`。
5. `rollout_buffer.iter_minibatches` 用相同的行号切分所有字段，确保每条回答的 token、概率、奖励和 advantage 对齐。
6. `GRPO` 或 `Reinforce` 用当前 policy 重新计算 completion token 的 log-prob，再根据 advantage、KL 项及各自的目标函数更新 LoRA 参数。
7. 训练过程按配置记录日志、保存 checkpoint 并评估；`eval.py` 可单独复评 checkpoint，`gradescope_bundle.py` 整理提交产物。

`RolloutOutput` 是尚未评分的采样结果；`RolloutBatch` 是加入 reward 和 advantage 后交给 RL 算法的数据。令 `N = batch_size × group_size`、`L` 为填充后的序列长度，则主要张量形状为：序列 `[N, L]`，逐 token log-prob 与 completion mask `[N, L-1]`，reward 和 advantage `[N]`。mask 只选择生成回答的有效 token，排除 prompt 和 padding。

## 当前实现状态

学生练习中的核心函数已填写，包括逐 token 概率与 KL 估计、rollout minibatch、REINFORCE 和 GRPO 更新，以及 advantage 计算。代码仍需通过短训练和完整实验验证实际效果；源文件中保留的部分 `TODO(student)` 注释是原始题目说明。
