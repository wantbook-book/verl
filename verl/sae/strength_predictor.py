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
SAE Strength Predictor Module
预测针对特定问题的SAE特征叠加强度
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SAEStrengthConfig:
    """SAE强度预测器配置"""
    hidden_size: int = 4096  # LLM hidden size
    num_features: int = 512  # SAE特征数量
    predictor_hidden_size: int = 1024  # 预测器隐层大小
    num_layers: int = 2  # 预测器层数
    dropout: float = 0.1
    activation: str = "relu"  # 激活函数类型
    max_strength: float = 5.0  # 最大强度值
    min_strength: float = -5.0  # 最小强度值
    

class SAEStrengthPredictor(nn.Module):
    """
    SAE特征强度预测器
    输入：问题的embedding表示
    输出：每个SAE特征的叠加强度
    """
    
    def __init__(self, config: SAEStrengthConfig):
        super().__init__()
        self.config = config
        
        # 构建MLP预测器
        layers = []
        input_size = config.hidden_size
        
        for i in range(config.num_layers):
            if i == config.num_layers - 1:
                # 最后一层输出特征强度
                layers.append(nn.Linear(input_size, config.num_features))
            else:
                layers.append(nn.Linear(input_size, config.predictor_hidden_size))
                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "gelu":
                    layers.append(nn.GELU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())
                layers.append(nn.Dropout(config.dropout))
                input_size = config.predictor_hidden_size
        
        self.predictor = nn.Sequential(*layers)
        
        # 用于限制强度范围的tanh激活
        self.strength_range = (config.max_strength - config.min_strength) / 2
        self.strength_center = (config.max_strength + config.min_strength) / 2
    
    def forward(self, question_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            question_embeddings: 问题embedding [batch_size, hidden_size]
            
        Returns:
            strengths: SAE特征强度 [batch_size, num_features]
        """
        # 通过MLP预测原始强度
        raw_strengths = self.predictor(question_embeddings)
        
        # 使用tanh限制强度范围
        strengths = torch.tanh(raw_strengths) * self.strength_range + self.strength_center
        
        return strengths


class SAEEnhancedModel(nn.Module):
    """
    集成SAE特征叠加的增强模型
    包含原始LLM + SAE强度预测器
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 sae_model: nn.Module,
                 strength_config: SAEStrengthConfig,
                 target_layer: int = -1):
        super().__init__()
        
        self.base_model = base_model
        self.sae_model = sae_model
        self.strength_predictor = SAEStrengthPredictor(strength_config)
        self.target_layer = target_layer  # 在哪一层添加SAE特征
        self.strength_config = strength_config
        
        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 只训练强度预测器
        for param in self.strength_predictor.parameters():
            param.requires_grad = True
    
    def get_question_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        获取问题的embedding表示
        使用attention mask进行平均pooling
        """
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # 最后一层
            
            # 使用attention mask进行平均pooling
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            question_embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
        return question_embedding
    
    def predict_strengths(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        预测SAE特征强度
        """
        question_embedding = self.get_question_embedding(input_ids, attention_mask)
        strengths = self.strength_predictor(question_embedding)
        return strengths
    
    def apply_sae_steering(self, 
                          hidden_states: torch.Tensor, 
                          strengths: torch.Tensor) -> torch.Tensor:
        """
        应用SAE特征引导
        
        Args:
            hidden_states: 原始隐层状态 [batch_size, seq_len, hidden_size]
            strengths: SAE特征强度 [batch_size, num_features]
            
        Returns:
            steered_hidden_states: 引导后的隐层状态
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 解码SAE特征
        with torch.no_grad():
            # 假设SAE有decode方法
            if hasattr(self.sae_model, 'decode'):
                sae_features = self.sae_model.decode(strengths)  # [batch_size, hidden_size]
            elif hasattr(self.sae_model, 'decoder'):
                sae_features = self.sae_model.decoder(strengths)
            else:
                # 简单的线性变换作为fallback
                if not hasattr(self, '_fallback_decoder'):
                    self._fallback_decoder = nn.Linear(self.strength_config.num_features, hidden_size).to(strengths.device)
                sae_features = self._fallback_decoder(strengths)
        
        # 将SAE特征添加到最后一个token的hidden states
        steered_hidden_states = hidden_states.clone()
        steered_hidden_states[:, -1, :] += sae_features
        
        return steered_hidden_states


def create_sae_enhanced_model(base_model: nn.Module,
                             sae_model: nn.Module,
                             strength_config: SAEStrengthConfig,
                             target_layer: int = -1) -> SAEEnhancedModel:
    """
    创建SAE增强模型的工厂函数
    """
    # 创建增强模型
    enhanced_model = SAEEnhancedModel(
        base_model=base_model,
        sae_model=sae_model,
        strength_config=strength_config,
        target_layer=target_layer
    )
    
    return enhanced_model
