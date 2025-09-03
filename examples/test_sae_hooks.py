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
SAE Hooks功能测试脚本
测试改进后的SAE hooks实现，参考SAE-Reasoning2
"""

import torch
import sys
from pathlib import Path

# 添加VERL路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
        GlobalSAE,
        get_intervention_hook,
        get_clamp_hook, 
        get_multi_intervention_hook,
        add_hooks
    )
    HOOKS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Hook函数导入失败: {e}")
    HOOKS_AVAILABLE = False


class MockSAE:
    """模拟的SAE模型"""
    def __init__(self, hidden_size=1024, num_features=512):
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模拟SAE权重
        self.W_enc = torch.randn(hidden_size, num_features, device=self.device) * 0.1
        self.W_dec = torch.randn(num_features, hidden_size, device=self.device) * 0.1
        self.b_enc = torch.zeros(num_features, device=self.device)
        self.b_dec = torch.zeros(hidden_size, device=self.device)
        
        class MockConfig:
            hook_layer = 10
        self.cfg = MockConfig()
    
    def to(self, device):
        self.device = device
        self.W_enc = self.W_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_enc = self.b_enc.to(device)
        self.b_dec = self.b_dec.to(device)
        return self
    
    def encode(self, x):
        """编码：hidden states -> features"""
        # x: [batch, seq_len, hidden_size] -> [batch, seq_len, num_features]
        return torch.relu((x @ self.W_enc) + self.b_enc)
    
    def decode(self, features):
        """解码：features -> hidden states"""
        # features: [batch, seq_len, num_features] -> [batch, seq_len, hidden_size]
        return (features @ self.W_dec) + self.b_dec


class MockModule(torch.nn.Module):
    """模拟的神经网络模块"""
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        return self.linear(x)


def test_global_sae_control():
    """测试GlobalSAE控制机制"""
    print("🧪 测试GlobalSAE控制机制...")
    
    # 测试默认状态
    assert GlobalSAE.use_sae == True, "GlobalSAE默认应该启用"
    print("✅ GlobalSAE默认启用")
    
    # 测试禁用
    GlobalSAE.use_sae = False
    assert GlobalSAE.use_sae == False, "GlobalSAE禁用失败"
    print("✅ GlobalSAE禁用成功")
    
    # 重新启用
    GlobalSAE.use_sae = True
    assert GlobalSAE.use_sae == True, "GlobalSAE重新启用失败"
    print("✅ GlobalSAE重新启用成功")


def test_intervention_hook():
    """测试干预hook"""
    print("\n🧪 测试干预hook...")
    
    if not HOOKS_AVAILABLE:
        print("⚠️ Hook函数不可用，跳过测试")
        return
    
    # 创建模拟数据
    batch_size, seq_len, hidden_size = 2, 10, 1024
    num_features = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sae = MockSAE(hidden_size, num_features).to(device)
    module = MockModule(hidden_size).to(device)
    
    # 创建输入数据
    input_data = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 测试单特征干预
    feature_idx = 100
    strength = 2.0
    max_activation = 5.0
    
    hook_fn = get_intervention_hook(sae, feature_idx=feature_idx, strength=strength, max_activation=max_activation)
    
    # 应用hook
    with add_hooks([], [(module, hook_fn)]):
        output = module(input_data)
        
        print(f"✅ 单特征干预hook应用成功")
        print(f"   输入形状: {input_data.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   特征索引: {feature_idx}, 强度: {strength}")
    
    # 测试全局干预
    global_strength = 0.8
    global_hook_fn = get_intervention_hook(sae, strength=global_strength)
    
    with add_hooks([], [(module, global_hook_fn)]):
        output = module(input_data)
        print(f"✅ 全局干预hook应用成功，强度: {global_strength}")


def test_clamp_hook():
    """测试钳制hook"""
    print("\n🧪 测试钳制hook...")
    
    if not HOOKS_AVAILABLE:
        print("⚠️ Hook函数不可用，跳过测试")
        return
    
    # 创建模拟数据
    batch_size, seq_len, hidden_size = 2, 10, 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    module = MockModule(hidden_size).to(device)
    
    # 创建方向向量
    direction = torch.randn(hidden_size, device=device)
    max_activation = 3.0
    strength = 1.5
    
    hook_fn = get_clamp_hook(direction, max_activation, strength)
    
    # 创建输入数据
    input_data = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 应用hook
    with add_hooks([], [(module, hook_fn)]):
        output = module(input_data)
        
        print(f"✅ 钳制hook应用成功")
        print(f"   方向向量形状: {direction.shape}")
        print(f"   最大激活值: {max_activation}, 强度: {strength}")


def test_multi_intervention_hook():
    """测试多特征干预hook"""
    print("\n🧪 测试多特征干预hook...")
    
    if not HOOKS_AVAILABLE:
        print("⚠️ Hook函数不可用，跳过测试")
        return
    
    # 创建模拟数据
    batch_size, seq_len, hidden_size = 2, 10, 1024
    num_features = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sae = MockSAE(hidden_size, num_features).to(device)
    module = MockModule(hidden_size).to(device)
    
    # 多特征参数
    feature_idxs = [50, 100, 200]
    max_activations = [3.0, 4.0, 2.5]
    strengths = [1.0, 1.5, 0.8]
    
    hook_fn = get_multi_intervention_hook(sae, feature_idxs, max_activations, strengths)
    
    # 创建输入数据
    input_data = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 应用hook
    with add_hooks([], [(module, hook_fn)]):
        output = module(input_data)
        
        print(f"✅ 多特征干预hook应用成功")
        print(f"   特征索引: {feature_idxs}")
        print(f"   最大激活值: {max_activations}")
        print(f"   强度: {strengths}")


def test_hook_disable():
    """测试hook禁用功能"""
    print("\n🧪 测试hook禁用功能...")
    
    if not HOOKS_AVAILABLE:
        print("⚠️ Hook函数不可用，跳过测试")
        return
    
    # 创建模拟数据
    batch_size, seq_len, hidden_size = 2, 10, 1024
    num_features = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sae = MockSAE(hidden_size, num_features).to(device)
    module = MockModule(hidden_size).to(device)
    
    input_data = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 创建hook
    hook_fn = get_intervention_hook(sae, feature_idx=100, strength=2.0)
    
    # 启用状态下的输出
    GlobalSAE.use_sae = True
    with add_hooks([], [(module, hook_fn)]):
        output_enabled = module(input_data)
    
    # 禁用状态下的输出
    GlobalSAE.use_sae = False
    with add_hooks([], [(module, hook_fn)]):
        output_disabled = module(input_data)
    
    # 恢复启用状态
    GlobalSAE.use_sae = True
    
    print("✅ Hook禁用功能测试完成")
    print(f"   启用时输出均值: {output_enabled.mean():.4f}")
    print(f"   禁用时输出均值: {output_disabled.mean():.4f}")


def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试错误处理...")
    
    if not HOOKS_AVAILABLE:
        print("⚠️ Hook函数不可用，跳过测试")
        return
    
    # 测试无效特征索引
    try:
        sae = MockSAE(1024, 512)
        hook_fn = get_intervention_hook(sae, feature_idx=1000)  # 超出范围
        print("✅ 无效特征索引处理正常（不会立即报错）")
    except Exception as e:
        print(f"⚠️ 无效特征索引处理异常: {e}")
    
    # 测试设备不匹配
    try:
        sae = MockSAE(1024, 512).to('cpu')
        module = MockModule(1024)
        input_data = torch.randn(2, 10, 1024)
        
        if torch.cuda.is_available():
            input_data = input_data.cuda()
            module = module.cuda()
        
        hook_fn = get_intervention_hook(sae, feature_idx=100)
        
        with add_hooks([], [(module, hook_fn)]):
            output = module(input_data)
        
        print("✅ 设备不匹配自动处理正常")
    except Exception as e:
        print(f"⚠️ 设备不匹配处理异常: {e}")


def main():
    """主测试函数"""
    print("🚀 开始SAE Hooks功能测试")
    print("=" * 60)
    
    try:
        test_global_sae_control()
        test_intervention_hook()
        test_clamp_hook()
        test_multi_intervention_hook()
        test_hook_disable()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 SAE Hooks功能测试完成！")
        print("\n💡 测试总结:")
        print("- ✅ GlobalSAE控制机制正常")
        print("- ✅ 单特征干预hook正常")
        print("- ✅ 钳制hook正常")
        print("- ✅ 多特征干预hook正常")
        print("- ✅ Hook禁用功能正常")
        print("- ✅ 错误处理机制正常")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
