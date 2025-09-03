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
SAE Enhanced Rollout Implementation
集成SAE特征叠加的rollout实现
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from typing import Optional, Dict, Any

from verl import DataProto
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.rollout.base import BaseRollout

from .strength_predictor import SAEEnhancedModel, SAEStrengthConfig

__all__ = ["SAEEnhancedRollout"]


class SAEEnhancedRollout(BaseRollout):
    """
    集成SAE特征叠加的Rollout类
    在生成过程中动态预测和应用SAE特征强度
    """
    
    def __init__(self, 
                 enhanced_model: SAEEnhancedModel, 
                 config: Dict[str, Any],
                 sae_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            enhanced_model: SAE增强模型
            config: 生成配置
            sae_config: SAE相关配置
        """
        super().__init__()
        self.config = config
        self.enhanced_model = enhanced_model
        self.sae_config = sae_config or {}
        
        # SAE配置参数
        self.enable_sae_steering = self.sae_config.get("enable_steering", True)
        self.steering_layer = self.sae_config.get("steering_layer", -1)
        self.strength_scale = self.sae_config.get("strength_scale", 1.0)
        
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        生成序列，集成SAE特征叠加
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # 用于构建attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.enhanced_model.eval()

        # 检查meta_info中是否启用了SAE
        sae_enabled = prompts.meta_info.get("sae_enabled", self.enable_sae_steering)
        
        # 预测SAE特征强度（只在开始时预测一次）
        sae_strengths = None
        if sae_enabled:
            try:
                sae_strengths = self.enhanced_model.predict_strengths(
                    input_ids=idx, 
                    attention_mask=attention_mask
                )
                # 应用强度缩放
                strength_scale = prompts.meta_info.get("sae_strength_scale", self.strength_scale)
                sae_strengths = sae_strengths * strength_scale
                print(f"🔥 SAE strengths predicted: mean={sae_strengths.mean():.4f}, std={sae_strengths.std():.4f}")
            except Exception as e:
                print(f"⚠️ SAE strength prediction failed: {e}")
                sae_enabled = False

        prev_attention_mask = torch.ones(
            size=(batch_size, 1), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )

        logits_lst = []
        
        for step in range(self.config.response_length):
            idx_cond = idx
            
            # 获取模型输出，包含hidden states
            output = self.enhanced_model.base_model(
                input_ids=idx_cond, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                output_hidden_states=True
            )
            
            # 应用SAE特征叠加
            if sae_enabled and sae_strengths is not None:
                try:
                    steering_layer = prompts.meta_info.get("sae_steering_layer", self.steering_layer)
                    hidden_states = output.hidden_states[steering_layer]
                    steered_hidden_states = self.enhanced_model.apply_sae_steering(
                        hidden_states, sae_strengths
                    )
                    
                    # 重新计算logits（需要通过LM head）
                    if hasattr(self.enhanced_model.base_model, 'lm_head'):
                        logits = self.enhanced_model.base_model.lm_head(steered_hidden_states)
                    else:
                        # 对于一些模型，可能需要不同的方式获取logits
                        logits = output.logits
                except Exception as e:
                    print(f"⚠️ SAE steering failed at step {step}: {e}")
                    logits = output.logits
            else:
                logits = output.logits
            
            # 只取最后一个token的logits
            logits = logits[:, -1, :] / self.config.temperature  # (bs, vocab_size)
            
            # Top-k采样
            if self.config.top_k is not None:
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            
            # 应用softmax
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个token
            if self.config.do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            # 更新attention mask
            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

            # 检查EOS token
            for token_id in eos_token_id:
                prev_attention_mask = torch.logical_and(
                    idx_next != token_id, 
                    prev_attention_mask.bool()
                )
            prev_attention_mask = prev_attention_mask.to(attention_mask.dtype)

            # 更新position_ids
            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

            # 追加新token
            idx = torch.cat((idx, idx_next), dim=1)
            logits_lst.append(logits)

        # 构建输出
        logits = torch.stack(logits_lst, dim=1)  # (bs, response_length, vocab_size)
        prompts_out = idx[:, :prompt_length]  # (bs, prompt_length)
        response = idx[:, prompt_length:]  # (bs, response_length)
        log_probs = logprobs_from_logits(logits=logits, labels=response)
        
        batch = TensorDict(
            {
                "input_ids": prompts_out,
                "responses": response,
                "sequences": idx,
                "old_log_probs": log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # 添加SAE相关信息到meta_info
        meta_info = {}
        if sae_strengths is not None:
            meta_info["sae_strengths"] = sae_strengths.cpu()
            meta_info["sae_enabled"] = True
        else:
            meta_info["sae_enabled"] = False
        
        self.enhanced_model.train()

        return DataProto(batch=batch, meta_info=meta_info)


def create_sae_rollout(base_model: nn.Module,
                      sae_model: nn.Module,
                      rollout_config: Dict[str, Any],
                      sae_config: Dict[str, Any]) -> SAEEnhancedRollout:
    """
    创建SAE增强rollout的工厂函数
    """
    from .strength_predictor import create_sae_enhanced_model, SAEStrengthConfig
    
    # 创建强度预测器配置
    strength_config = SAEStrengthConfig(
        hidden_size=sae_config.get("hidden_size", 4096),
        num_features=sae_config.get("num_features", 512),
        predictor_hidden_size=sae_config.get("predictor_hidden_size", 1024),
        num_layers=sae_config.get("predictor_layers", 2),
        max_strength=sae_config.get("max_strength", 5.0),
        min_strength=sae_config.get("min_strength", -5.0)
    )
    
    # 创建增强模型
    enhanced_model = create_sae_enhanced_model(
        base_model=base_model,
        sae_model=sae_model,
        strength_config=strength_config,
        target_layer=sae_config.get("target_layer", -1)
    )
    
    # 创建rollout
    rollout = SAEEnhancedRollout(
        enhanced_model=enhanced_model,
        config=rollout_config,
        sae_config=sae_config
    )
    
    return rollout
