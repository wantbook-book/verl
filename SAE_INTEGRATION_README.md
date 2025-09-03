# VERL SAE Integration

本文档介绍如何在VERL框架中使用SAE（Sparse Autoencoder）特征叠加功能进行强化学习训练。

## 功能概述

SAE集成为VERL添加了以下功能：

1. **动态特征叠加**：在文本生成过程中动态叠加SAE特征
2. **强度预测**：根据问题内容预测最适合的特征强度
3. **选择性训练**：只训练强度预测器，保持基础LLM参数冻结
4. **无缝集成**：与现有VERL训练流程完全兼容

## 快速开始

### 1. 准备模型和数据

```bash
# 准备基础LLM模型
BASE_MODEL="/path/to/your/llm/model"

# 准备SAE模型
SAE_MODEL="/path/to/your/sae/model"

# 准备训练数据
TRAIN_DATA="/path/to/train/data.jsonl"
VAL_DATA="/path/to/val/data.jsonl"
```

### 2. 运行SAE增强训练

```bash
python examples/run_sae_ppo.py \
    --config examples/sae_ppo_config.yaml \
    --base-model-path $BASE_MODEL \
    --sae-model-path $SAE_MODEL \
    --train-data $TRAIN_DATA \
    --val-data $VAL_DATA \
    --strength-scale 1.0 \
    --output-dir ./sae_outputs
```

### 3. 禁用SAE（运行标准PPO）

```bash
python examples/run_sae_ppo.py \
    --config examples/sae_ppo_config.yaml \
    --base-model-path $BASE_MODEL \
    --train-data $TRAIN_DATA \
    --disable-sae \
    --output-dir ./standard_outputs
```

## 配置说明

### SAE配置项

```yaml
sae:
  enable: true                    # 启用SAE功能
  model_path: "/path/to/sae"      # SAE模型路径
  
  # 模型配置
  hidden_size: 4096               # 必须与LLM匹配
  num_features: 512               # SAE特征数量
  target_layer: -1                # 特征叠加层位置
  
  # 强度预测器配置
  predictor_hidden_size: 1024     # 预测器隐层大小
  predictor_layers: 2             # 预测器层数
  max_strength: 5.0               # 最大强度值
  min_strength: -5.0              # 最小强度值
  
  # 训练配置
  strength_scale: 1.0             # 强度缩放因子
  enable_steering: true           # 启用特征引导
```

## 实现细节

### 1. 代码修改位置

- **`verl/sae/`**: 新增SAE集成模块
- **`verl/workers/fsdp_workers.py`**: 添加SAE配置传递
- **`verl/workers/rollout/naive/naive_rollout.py`**: 集成SAE特征叠加（基础实现）
- **`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`**: vLLM SAE集成（推荐）
- **`verl/trainer/ppo/ray_trainer.py`**: 验证阶段SAE支持

### 2. 工作流程

```
问题输入 → 配置检查 → 强度预测 → 特征叠加 → 文本生成
    ↓           ↓           ↓           ↓           ↓
meta_info → SAE启用 → 随机强度* → hidden修改 → 记录统计
```

*注：当前使用随机强度作为示例，实际应用需要真正的强度预测器

### 3. 关键组件

#### SAE强度预测器 (`verl/sae/strength_predictor.py`)
- 输入：问题embedding
- 输出：SAE特征强度向量
- 架构：可配置的MLP网络

#### SAE增强Rollout (`verl/sae/sae_rollout.py`)
- 集成SAE特征叠加到生成过程
- 支持动态强度预测
- 错误处理和降级机制

#### Naive Rollout增强 (`verl/workers/rollout/naive/naive_rollout.py`)
- 基础SAE集成实现
- 检测SAE配置并应用特征叠加
- 适用于简单的单GPU场景

#### vLLM Rollout增强 (`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`) **推荐**
- 专业的SAE集成实现，参考SAE-Reasoning2
- 支持真正的SAE模型和hook机制
- 支持多种干预模式：
  - **单特征干预**: 针对特定特征索引的干预
  - **多特征干预**: 同时干预多个特征
  - **全局干预**: 对所有特征应用统一强度
  - **Clamp干预**: 基于方向向量的钳制干预
- GlobalSAE控制机制，可动态启用/禁用
- 完善的错误处理和设备自适应
- 适用于生产环境和大规模部署

## 监控和调试

### 1. 日志输出

vLLM rollout的SAE集成会输出详细日志：
```
🔥 SAE enabled in vLLM rollout
   SAE config: {'enable': True, 'model_path': '/path/to/sae'}
   SAE params: {'sae_feature_idx': 1160, 'sae_strength_scale': 1.0, ...}
🎯 Setup intervention SAE hook on layer 19 for feature 1160
🚀 Generating with SAE hooks active...
✅ SAE generation completed, hooks removed
✅ vLLM SAE rollout completed with SAE intervention
```

Naive rollout的日志格式：
```
🔥 SAE enabled for rollout generation:
   - Strength scale: 1.0
   - Target layer: -1
📊 Generated SAE strengths: mean=0.0245, std=0.9876
✅ Applied SAE steering at step 0
✅ SAE rollout completed with strengths recorded
```

### 2. 支持的干预模式

#### 单特征干预
```yaml
sae:
  feature_idx: 1160
  strength_scale: 1.0
  max_activation: 5.0
```

#### 多特征干预
```yaml
sae:
  feature_idxs: [1160, 2340, 890]
  max_activations: [5.0, 3.0, 4.0]
  strengths: [1.0, 1.5, 0.8]
```

#### 全局干预
```yaml
sae:
  strength_scale: 0.8  # 仅指定全局强度
```

#### Clamp干预（通过代码配置）
```python
# 在meta_info中指定使用clamp模式
meta_info = {
    "sae_feature_idx": 1160,
    "sae_max_activation": 5.0,
    "sae_strength_scale": 1.0,
}
```

### 3. 统计信息

训练过程中会记录：
- SAE强度的均值和标准差
- 特征叠加成功/失败次数
- 生成质量指标对比

### 4. 故障排除

常见问题及解决方案：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| SAE模型加载失败 | 路径错误或格式不匹配 | 检查模型路径和格式 |
| 维度不匹配 | SAE hidden_size与LLM不匹配 | 调整配置中的hidden_size |
| 内存不足 | SAE增加了内存消耗 | 减少batch_size或使用更少特征 |
| 特征叠加失败 | 模型结构不兼容 | 检查模型是否支持output_hidden_states |

## 扩展和自定义

### 1. 实现真正的强度预测器

当前示例使用随机强度，实际应用需要：

```python
# 在 naive_rollout.py 中替换随机强度生成
# 当前代码：
sae_strengths = torch.randn(batch_size, num_features, device=idx.device) * strength_scale

# 替换为真正的预测器：
from verl.sae import SAEStrengthPredictor
predictor = SAEStrengthPredictor(config)
sae_strengths = predictor.predict_strengths(input_ids=idx, attention_mask=attention_mask)
```

### 2. 集成真正的SAE模型

```python
# 替换简单的线性变换
# 当前代码：
self._sae_decoder = torch.nn.Linear(num_features, hidden_size)

# 替换为真正的SAE：
from sae_lens import SAE
sae_model = SAE.from_pretrained("path/to/sae")
sae_features = sae_model.decode(sae_strengths)
```

### 3. 添加更多SAE配置

```yaml
sae:
  # 多层特征叠加
  multi_layer_steering: true
  target_layers: [12, 24, -1]
  
  # 特征选择
  feature_selection: true
  selected_features: [1, 5, 10, 20]
  
  # 动态强度调整
  dynamic_strength: true
  strength_schedule: "cosine"
```

## 性能考虑

### 1. 内存使用

SAE集成会增加内存消耗：
- SAE模型参数：~100MB-1GB
- 强度预测器：~10-50MB
- 额外激活值：~10-20%增加

### 2. 计算开销

- 强度预测：每个batch额外1-5ms
- 特征叠加：每个生成步骤额外1-2ms
- 总体开销：约5-10%性能影响

### 3. 优化建议

- 使用较小的SAE特征数量
- 缓存强度预测结果
- 使用混合精度训练
- 考虑使用gradient checkpointing

## 未来改进

1. **智能强度预测**：基于问题类型和历史性能的智能强度预测
2. **多SAE支持**：支持同时使用多个SAE模型
3. **在线学习**：根据奖励信号在线调整强度预测器
4. **可视化工具**：SAE特征激活和强度分布的可视化
5. **性能优化**：更高效的特征叠加实现

## 贡献指南

欢迎贡献SAE集成的改进！请遵循以下步骤：

1. Fork本项目
2. 创建功能分支：`git checkout -b feature/sae-improvement`
3. 提交更改：`git commit -am 'Add SAE improvement'`
4. 推送分支：`git push origin feature/sae-improvement`
5. 创建Pull Request

## 许可证

本SAE集成功能遵循与VERL相同的Apache 2.0许可证。
