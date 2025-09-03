# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import logprobs_from_logits

from ..base import BaseRollout

__all__ = ["NaiveRollout"]


class NaiveRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        """A naive rollout. It requires the module to be compatible with huggingface APIs. That is:
        The module should define __call__ to receive input_ids, attention_mask and position_ids.
        It outputs a structure that contains logits field.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
        """
        super().__init__()
        self.config = config
        self.module = module

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences with optional SAE feature steering"""
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()

        # 检查是否启用SAE
        sae_enabled = prompts.meta_info.get("sae_enabled", False)
        sae_strengths = None
        
        if sae_enabled:
            print("🔥 SAE enabled in naive rollout")
            # 这里可以添加SAE强度预测逻辑
            # 目前使用随机强度作为示例
            num_features = prompts.meta_info.get("sae_num_features", 512)
            strength_scale = prompts.meta_info.get("sae_strength_scale", 1.0)
            
            # 生成随机强度（实际应用中应该通过强度预测器生成）
            sae_strengths = torch.randn(batch_size, num_features, device=idx.device) * strength_scale
            print(f"📊 Generated SAE strengths: mean={sae_strengths.mean():.4f}, std={sae_strengths.std():.4f}")

        prev_attention_mask = torch.ones(size=(batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        logits_lst = []
        for step in range(self.config.response_length):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            idx_cond = idx
            # forward the model to get the logits for the index in the sequence
            # we use huggingface APIs here
            output = self.module(input_ids=idx_cond, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=sae_enabled)
            
            # 应用SAE特征叠加（如果启用）
            if sae_enabled and sae_strengths is not None:
                try:
                    # 获取hidden states
                    if hasattr(output, 'hidden_states') and output.hidden_states is not None:
                        steering_layer = prompts.meta_info.get("sae_steering_layer", -1)
                        hidden_states = output.hidden_states[steering_layer]  # [batch_size, seq_len, hidden_size]
                        
                        # 简单的SAE特征叠加（实际应用中需要真正的SAE解码器）
                        # 这里使用线性变换作为示例
                        if not hasattr(self, '_sae_decoder'):
                            hidden_size = hidden_states.shape[-1]
                            num_features = sae_strengths.shape[-1]
                            self._sae_decoder = torch.nn.Linear(num_features, hidden_size).to(hidden_states.device)
                        
                        # 解码SAE特征
                        sae_features = self._sae_decoder(sae_strengths)  # [batch_size, hidden_size]
                        
                        # 将特征添加到最后一个token
                        modified_hidden_states = hidden_states.clone()
                        modified_hidden_states[:, -1, :] += sae_features
                        
                        # 重新计算logits（需要模型的lm_head）
                        if hasattr(self.module, 'lm_head'):
                            # 只对最后一个token的hidden states计算logits
                            last_hidden = modified_hidden_states[:, -1, :]  # [batch_size, hidden_size]
                            logits = self.module.lm_head(last_hidden)  # [batch_size, vocab_size]
                        else:
                            logits = output.logits
                        
                        print(f"✅ Applied SAE steering at step {step}")
                    else:
                        logits = output.logits
                except Exception as e:
                    print(f"⚠️ SAE steering failed at step {step}: {e}")
                    logits = output.logits
            else:
                logits = output.logits
            
            # pluck the logits at the final step and scale by desired temperature
            # 检查logits的维度，如果已经是2D就不需要再索引
            if len(logits.shape) == 3:
                logits = logits[:, -1, :]  # (bs, seq_len, vocab_size) -> (bs, vocab_size)
            elif len(logits.shape) == 2:
                pass  # 已经是 (bs, vocab_size)
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            logits = logits / self.config.temperature  # (bs, vocab_size)
            # optionally crop the logits to only the top k options
            if self.config.top_k is not None:
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            if self.config.do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

            for token_id in eos_token_id:
                prev_attention_mask = torch.logical_and(idx_next != token_id, prev_attention_mask.bool())
            prev_attention_mask.to(attention_mask.dtype)

            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            logits_lst.append(logits)

        logits = torch.stack(logits_lst, dim=1)  # (bs, response_length, vocab_size)
        prompts = idx[:, :prompt_length]  # (bs, prompt_length)
        response = idx[:, prompt_length:]  # (bs, response_length)
        log_probs = logprobs_from_logits(logits=logits, labels=response)
        batch = TensorDict(
            {
                "input_ids": prompts,
                "responses": response,
                "sequences": idx,
                "old_log_probs": log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        self.module.train()

        # 添加SAE相关信息到meta_info
        meta_info = {}
        if sae_enabled and sae_strengths is not None:
            meta_info["sae_strengths"] = sae_strengths.cpu()
            meta_info["sae_enabled"] = True
            print("✅ SAE rollout completed with strengths recorded")
        else:
            meta_info["sae_enabled"] = False

        return DataProto(batch=batch, meta_info=meta_info)
