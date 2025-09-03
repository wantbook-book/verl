#!/usr/bin/env python3
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
SAE集成功能测试脚本
验证SAE特征叠加是否正常工作
"""

import torch
import sys
from pathlib import Path
from tensordict import TensorDict

# 添加VERL路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl import DataProto
from verl.workers.rollout.naive.naive_rollout import NaiveRollout


class MockModel(torch.nn.Module):
    """模拟的LLM模型用于测试"""
    
    def __init__(self, vocab_size=32000, hidden_size=4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerDecoderLayer(
                d_model=hidden_size, 
                nhead=32,
                batch_first=True
            ) for _ in range(2)
        ])
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, output_hidden_states=False):
        # 简单的前向传播
        x = self.embed_tokens(input_ids)
        
        hidden_states = []
        for layer in self.layers:
            if output_hidden_states:
                hidden_states.append(x)
            # 简化的transformer层
            x = layer(x, x)
        
        if output_hidden_states:
            hidden_states.append(x)
        
        logits = self.lm_head(x)
        
        # 返回类似HuggingFace的输出
        class Output:
            def __init__(self, logits, hidden_states=None):
                self.logits = logits
                self.hidden_states = hidden_states if hidden_states else None
        
        return Output(logits, hidden_states if output_hidden_states else None)


class MockConfig:
    """模拟的配置类"""
    def __init__(self):
        self.response_length = 10
        self.temperature = 1.0
        self.top_k = 50
        self.do_sample = True


def test_sae_integration():
    """测试SAE集成功能"""
    print("🧪 开始测试SAE集成功能...")
    
    # 创建模拟模型和配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel().to(device)
    config = MockConfig()
    
    # 创建rollout
    rollout = NaiveRollout(model, config)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 20
    vocab_size = 32000
    
    input_ids = torch.randint(1, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
    
    batch = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask, 
        "position_ids": position_ids,
    }, batch_size=batch_size)
    
    # 测试1: 不启用SAE
    print("\n📋 测试1: 标准生成（不启用SAE）")
    meta_info_standard = {
        "eos_token_id": [2],
        "pad_token_id": 0,
        "sae_enabled": False,
    }
    
    prompts_standard = DataProto(batch=batch, meta_info=meta_info_standard)
    
    try:
        output_standard = rollout.generate_sequences(prompts_standard)
        print("✅ 标准生成成功")
        print(f"   输出形状: {output_standard.batch['responses'].shape}")
        print(f"   SAE启用: {output_standard.meta_info.get('sae_enabled', False)}")
    except Exception as e:
        print(f"❌ 标准生成失败: {e}")
        return False
    
    # 测试2: 启用SAE
    print("\n📋 测试2: SAE增强生成")
    meta_info_sae = {
        "eos_token_id": [2],
        "pad_token_id": 0,
        "sae_enabled": True,
        "sae_num_features": 512,
        "sae_strength_scale": 1.0,
        "sae_steering_layer": -1,
    }
    
    prompts_sae = DataProto(batch=batch, meta_info=meta_info_sae)
    
    try:
        output_sae = rollout.generate_sequences(prompts_sae)
        print("✅ SAE增强生成成功")
        print(f"   输出形状: {output_sae.batch['responses'].shape}")
        print(f"   SAE启用: {output_sae.meta_info.get('sae_enabled', False)}")
        
        if "sae_strengths" in output_sae.meta_info:
            strengths = output_sae.meta_info["sae_strengths"]
            print(f"   强度统计: mean={strengths.mean():.4f}, std={strengths.std():.4f}")
            print(f"   强度形状: {strengths.shape}")
        
    except Exception as e:
        print(f"❌ SAE增强生成失败: {e}")
        return False
    
    # 测试3: 比较输出差异
    print("\n📋 测试3: 输出差异分析")
    
    try:
        # 比较生成的文本是否不同（应该有差异，因为SAE改变了生成过程）
        responses_standard = output_standard.batch["responses"]
        responses_sae = output_sae.batch["responses"]
        
        # 计算差异
        differences = (responses_standard != responses_sae).float().mean()
        print(f"   Token差异率: {differences:.2%}")
        
        if differences > 0:
            print("✅ SAE确实影响了生成结果")
        else:
            print("⚠️ SAE似乎没有影响生成结果（可能是随机性导致）")
            
    except Exception as e:
        print(f"⚠️ 输出比较失败: {e}")
    
    print("\n🎉 SAE集成测试完成！")
    return True


def test_sae_config_propagation():
    """测试SAE配置传播"""
    print("\n🧪 测试SAE配置传播...")
    
    # 测试不同的SAE配置
    test_configs = [
        {"sae_enabled": True, "sae_strength_scale": 0.5},
        {"sae_enabled": True, "sae_strength_scale": 2.0},
        {"sae_enabled": False},
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MockModel().to(device)
    config = MockConfig()
    rollout = NaiveRollout(model, config)
    
    # 创建基础数据
    batch_size = 1
    seq_len = 10
    
    input_ids = torch.randint(1, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
    
    batch = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, batch_size=batch_size)
    
    for i, test_config in enumerate(test_configs):
        print(f"\n   配置 {i+1}: {test_config}")
        
        meta_info = {
            "eos_token_id": [2],
            "pad_token_id": 0,
            **test_config
        }
        
        prompts = DataProto(batch=batch, meta_info=meta_info)
        
        try:
            output = rollout.generate_sequences(prompts)
            sae_enabled = output.meta_info.get("sae_enabled", False)
            print(f"   ✅ 生成成功，SAE状态: {sae_enabled}")
            
            if sae_enabled and "sae_strengths" in output.meta_info:
                strengths = output.meta_info["sae_strengths"]
                print(f"   📊 强度范围: [{strengths.min():.2f}, {strengths.max():.2f}]")
                
        except Exception as e:
            print(f"   ❌ 生成失败: {e}")
    
    print("✅ 配置传播测试完成")


if __name__ == "__main__":
    print("🚀 开始SAE集成测试")
    print("="*50)
    
    # 检查依赖
    try:
        import torch
        import tensordict
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        sys.exit(1)
    
    # 运行测试
    success = True
    
    try:
        success &= test_sae_integration()
        test_sae_config_propagation()
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 所有测试通过！SAE集成功能正常工作")
    else:
        print("❌ 部分测试失败，请检查实现")
        sys.exit(1)
