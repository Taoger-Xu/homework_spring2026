from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch


@dataclass
class RolloutBatch:
    # RolloutOutput 是采样器的原始输出：包含生成文本、题目元数据和 token 概率，尚未评分。
    # RolloutBatch 是训练用数据：在采样结果上加入 rewards 和 advantages，供 RL 算法更新策略。
    # 两者都按回答逐行排列；每一行的 token、概率、奖励和优势必须对应同一条回答。
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    # Optional debug
    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # 批量大小必须为正，否则无法按固定步长切分。
    if minibatch_size <= 0:
        raise ValueError(f"minibatch_size must be positive, got {minibatch_size}")

    # 第一维是本次 rollout 生成的回答总数 N。
    n = batch.input_ids.shape[0]

    # 所有字段共用这一组行号，保证回答、概率、奖励和优势始终对齐。
    if shuffle:
        # 使用传入的随机数生成器打乱行号，便于复现实验。
        indices = torch.randperm(n, generator=generator, device=batch.input_ids.device)
    else:
        # 不打乱时，按原始顺序遍历所有回答。
        indices = torch.arange(n, device=batch.input_ids.device)

    # 每次取最多 minibatch_size 条；最后一批可以更小。
    for start in range(0, n, minibatch_size):
        # 取得当前小批量在完整 rollout 中的行号。
        idx = indices[start : start + minibatch_size]
        # Python 列表字段需要普通整数下标，而不是张量下标。
        list_idx = idx.tolist()

        # 用相同的行号切出所有字段，组成可直接用于策略更新的小批量。
        minibatch = RolloutBatch(
            input_ids=batch.input_ids[idx],
            attention_mask=batch.attention_mask[idx],
            completion_mask=batch.completion_mask[idx],
            old_logprobs=batch.old_logprobs[idx],
            ref_logprobs=batch.ref_logprobs[idx],
            rewards=batch.rewards[idx],
            advantages=batch.advantages[idx],
            task_names=(
                [batch.task_names[i] for i in list_idx]
                if batch.task_names is not None
                else None
            ),
            completion_texts=(
                [batch.completion_texts[i] for i in list_idx]
                if batch.completion_texts is not None
                else None
            ),
        )

        # 如需在 GPU 上更新，只搬运当前小批量，而非整个 rollout。
        if device is not None:
            minibatch = minibatch.to(device)

        # 交给 RL 算法处理；下次迭代再产生下一批。
        yield minibatch
