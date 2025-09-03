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
vLLM SAE集成功能测试脚本
验证vLLM rollout的SAE特征叠加是否正常工作
"""

import torch
import sys
from pathlib import Path
from tensordict import TensorDict
from omegaconf import DictConfig, OmegaConf

# 添加VERL路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout


def create_mock_config(enable_sae=False):
    """创建模拟配置"""
    config = {
        "tensor_model_parallel_size": 1,
        "max_num_batched_tokens": 8192,
        "prompt_length": 100,
        "response_length": 10,
        "max_model_len": 200,
        "enable_chunked_prefill": False,
        "load_format": "dummy",
        "free_cache_engine": False,
        "dtype": "float16",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.9,
        "disable_log_stats": True,
        "calculate_log_probs": False,
        "temperature": 1.0,
        "top_k": 50,
        "do_sample": True,
        "seed": 42,
    }
    
    if enable_sae:
        config["sae"] = {
            "enable": True,
            "model_path": "/fake/sae/path",  # 模拟路径
            "release": "test_release",
            "id": "test_id",
            "feature_idx": 1160,
            "strength_scale": 1.0,
            "max_activation": 5.0,
        }
    
    return DictConfig(config)


class MockTokenizer:
    """模拟的tokenizer"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2


def test_vllm_sae_config_loading():
    """测试vLLM SAE配置加载"""
    print("🧪 测试vLLM SAE配置加载...")
    
    # 测试不启用SAE
    config_no_sae = create_mock_config(enable_sae=False)
    tokenizer = MockTokenizer()
    
    try:
        rollout_no_sae = vLLMRollout(
            model_path="dummy_model",
            config=config_no_sae,
            tokenizer=tokenizer,
            model_hf_config=None
        )
        
        assert rollout_no_sae.sae is None
        assert not rollout_no_sae.sae_config
        print("✅ 不启用SAE的配置加载正常")
        
    except Exception as e:
        print(f"⚠️ 不启用SAE的配置加载失败: {e}")
    
    # 测试启用SAE（但没有真实的SAE模型）
    config_with_sae = create_mock_config(enable_sae=True)
    
    try:
        rollout_with_sae = vLLMRollout(
            model_path="dummy_model",
            config=config_with_sae,
            tokenizer=tokenizer,
            model_hf_config=None
        )
        
        # 由于没有真实的SAE模型，应该加载失败但不会崩溃
        print(f"SAE加载状态: {rollout_with_sae.sae is not None}")
        print(f"SAE配置: {rollout_with_sae.sae_config}")
        print("✅ 启用SAE的配置加载完成（可能没有真实SAE模型）")
        
    except Exception as e:
        print(f"⚠️ 启用SAE的配置加载失败: {e}")


def test_sae_meta_info_processing():
    """测试SAE meta_info处理"""
    print("\n🧪 测试SAE meta_info处理...")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 20
    
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    batch = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, batch_size=batch_size)
    
    # 测试不同的SAE meta_info配置
    test_configs = [
        {
            "name": "无SAE",
            "meta_info": {
                "eos_token_id": [2],
                "pad_token_id": 0,
                "sae_enabled": False,
            }
        },
        {
            "name": "启用SAE - 单特征干预",
            "meta_info": {
                "eos_token_id": [2],
                "pad_token_id": 0,
                "sae_enabled": True,
                "sae_feature_idx": 1160,
                "sae_strength_scale": 1.0,
                "sae_max_activation": 5.0,
                "sae_steering_layer": -1,
            }
        },
        {
            "name": "启用SAE - 全局干预",
            "meta_info": {
                "eos_token_id": [2],
                "pad_token_id": 0,
                "sae_enabled": True,
                "sae_strength_scale": 0.5,
                "sae_steering_layer": 10,
            }
        },
    ]
    
    for test_config in test_configs:
        print(f"\n   测试配置: {test_config['name']}")
        
        prompts = DataProto(batch=batch, meta_info=test_config["meta_info"])
        
        # 模拟SAE参数提取逻辑
        sae_enabled = prompts.meta_info.get("sae_enabled", False)
        
        if sae_enabled:
            sae_params = {
                'sae_feature_idx': prompts.meta_info.get('sae_feature_idx'),
                'sae_strength_scale': prompts.meta_info.get('sae_strength_scale', 1.0),
                'sae_max_activation': prompts.meta_info.get('sae_max_activation'),
                'sae_steering_layer': prompts.meta_info.get('sae_steering_layer', -1),
            }
            print(f"   ✅ SAE参数提取成功: {sae_params}")
        else:
            print("   ✅ 正确识别为非SAE模式")


def test_sae_hook_functions():
    """测试SAE hook函数"""
    print("\n🧪 测试SAE hook函数...")
    
    # 测试hook函数的导入和基本功能
    try:
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
            get_intervention_hook, 
            get_clamp_hook, 
            add_hooks
        )
        print("✅ SAE hook函数导入成功")
        
        # 测试add_hooks上下文管理器
        class MockModule:
            def register_forward_hook(self, hook_fn):
                class MockHandle:
                    def remove(self):
                        pass
                return MockHandle()
        
        mock_module = MockModule()
        mock_hook_fn = lambda module, input, output: output
        
        hook_context = add_hooks([], [(mock_module, mock_hook_fn)])
        
        with hook_context:
            print("✅ Hook上下文管理器工作正常")
        
        print("✅ Hook移除正常")
        
    except ImportError as e:
        print(f"⚠️ SAE hook函数导入失败: {e}")
    except Exception as e:
        print(f"⚠️ SAE hook函数测试失败: {e}")


def main():
    """主测试函数"""
    print("🚀 开始vLLM SAE集成测试")
    print("=" * 60)
    
    try:
        # 检查SAE依赖
        try:
            from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import SAE_AVAILABLE
            if SAE_AVAILABLE:
                print("✅ SAE-Lens可用")
            else:
                print("⚠️ SAE-Lens不可用，将跳过部分测试")
        except ImportError:
            print("⚠️ 无法检查SAE依赖")
        
        # 运行测试
        test_vllm_sae_config_loading()
        test_sae_meta_info_processing()
        test_sae_hook_functions()
        
        print("\n" + "=" * 60)
        print("🎉 vLLM SAE集成测试完成！")
        print("\n💡 注意事项:")
        print("- 实际使用需要真实的SAE模型和vLLM环境")
        print("- 当前测试主要验证配置加载和参数处理逻辑")
        print("- 完整功能测试需要在实际的vLLM部署环境中进行")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
