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
SAE增强PPO训练示例脚本
展示如何使用SAE特征叠加功能进行强化学习训练
"""

import os
import sys
import argparse
from pathlib import Path

# 添加VERL路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import OmegaConf

# 导入VERL训练函数
from verl.trainer.main_ppo import run_ppo


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAE-enhanced PPO training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="examples/sae_ppo_config.yaml",
        help="Path to SAE PPO configuration file"
    )
    parser.add_argument(
        "--sae-model-path",
        type=str,
        required=True,
        help="Path to SAE model"
    )
    parser.add_argument(
        "--base-model-path", 
        type=str,
        required=True,
        help="Path to base LLM model"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True, 
        help="Path to training data"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        help="Path to validation data"
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        help="Path to reward model"
    )
    parser.add_argument(
        "--strength-scale",
        type=float,
        default=1.0,
        help="SAE strength scaling factor"
    )
    parser.add_argument(
        "--disable-sae",
        action="store_true",
        help="Disable SAE features (run standard PPO)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sae_ppo_outputs",
        help="Output directory for checkpoints and logs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 更新配置
    config.model.path = args.base_model_path
    config.data.train_files = [args.train_data]
    
    if args.val_data:
        config.data.val_files = [args.val_data]
    
    if args.reward_model_path:
        config.reward_model.path = args.reward_model_path
    
    # SAE配置
    if not args.disable_sae:
        config.sae.enable = True
        config.sae.model_path = args.sae_model_path
        config.sae.strength_scale = args.strength_scale
        print(f"🔥 SAE enabled with model: {args.sae_model_path}")
        print(f"   Strength scale: {args.strength_scale}")
    else:
        config.sae.enable = False
        print("⚠️ SAE disabled, running standard PPO")
    
    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    config.trainer.output_dir = args.output_dir
    
    # 打印配置摘要
    print("\n" + "="*50)
    print("SAE-Enhanced PPO Training Configuration")
    print("="*50)
    print(f"Base Model: {config.model.path}")
    print(f"SAE Enabled: {config.sae.enable}")
    if config.sae.enable:
        print(f"SAE Model: {config.sae.model_path}")
        print(f"Strength Scale: {config.sae.strength_scale}")
        print(f"Target Layer: {config.sae.target_layer}")
    print(f"Training Data: {config.data.train_files}")
    print(f"Output Dir: {args.output_dir}")
    print("="*50 + "\n")
    
    # 验证路径
    if not os.path.exists(args.base_model_path):
        raise FileNotFoundError(f"Base model not found: {args.base_model_path}")
    
    if config.sae.enable and not os.path.exists(args.sae_model_path):
        raise FileNotFoundError(f"SAE model not found: {args.sae_model_path}")
    
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    
    # 运行训练
    try:
        run_ppo(config)
        print("🎉 SAE-enhanced PPO training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
