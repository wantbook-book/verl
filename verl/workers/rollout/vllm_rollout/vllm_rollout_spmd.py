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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

# SAE相关导入
try:
    from sae_lens import SAE
    SAE_AVAILABLE = True
except ImportError:
    SAE = None
    SAE_AVAILABLE = False
import functools
# GlobalSAE控制机制
class GlobalSAE:
    use_sae = True

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


def get_intervention_hook(sae, feature_idx, strength=1.0, max_activation=1.0):
    """
    创建SAE干预hook，参考SAE-Reasoning2实现
    
    Args:
        sae: SAE模型
        feature_idx: 特征索引
        strength: 干预强度
        max_activation: 最大激活值
    """
    def hook_fn(module, input, output):
        if not GlobalSAE.use_sae:
            return output
            
        # 处理输出格式
        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()
        
        try:
            # 确保SAE在正确的设备上
            if sae.device != activations.device:
                sae.device = activations.device
                sae.to(sae.device)
            
            # 编码-解码-重构误差方法
            features = sae.encode(activations)
            reconstructed = sae.decode(features)
            error = activations.to(features.dtype) - reconstructed
            
            features[..., feature_idx] = max_activation * strength
            
            # 重构激活并添加误差
            activations_hat = sae.decode(features) + error
            activations_hat = activations_hat.type_as(activations)
            
            if torch.is_tensor(output):
                return activations_hat
            else:
                return (activations_hat,) + output[1:] if len(output) > 1 else (activations_hat,)
                
        except Exception as e:
            print(f"⚠️ SAE intervention hook error: {e}")
            return output
    
    return hook_fn


def get_clamp_hook(direction, max_activation, strength):
    """
    创建特征钳制hook，参考SAE-Reasoning2实现
    
    Args:
        direction: 特征方向向量
        max_activation: 最大激活值
        strength: 强度
    """
    def hook_fn(module, input, output):
        if not GlobalSAE.use_sae:
            return output
            
        # 处理输出格式
        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()
        
        try:
            # 标准化方向向量
            direction_normalized = direction / torch.norm(direction)
            direction_normalized = direction_normalized.type_as(activations)
            
            # 计算投影大小
            proj_magnitude = torch.sum(activations * direction_normalized, dim=-1, keepdim=True)
            
            # 计算正交分量
            orthogonal_component = activations - proj_magnitude * direction_normalized
            
            # 钳制并重构
            clamped = orthogonal_component + direction_normalized * max_activation * strength
            
            if torch.is_tensor(output):
                return clamped
            else:
                return (clamped,) + output[1:] if len(output) > 1 else (clamped,)
                
        except Exception as e:
            print(f"⚠️ SAE clamp hook error: {e}")
            return output
    
    return hook_fn


def get_multi_intervention_hook(
    sae: SAE,
    feature_idxs: list[int],
    max_activations: list[float],
    strengths: list[float],
):
    def hook_fn(module, input, output):
        if not GlobalSAE.use_sae:
            return output

        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()

        if sae.device != activations.device:
            sae.device = activations.device
            sae.to(sae.device)

        # import torch.distributed as dist
        # # 只在rank 0执行SAE计算
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     features = sae.encode(activations)
        #     reconstructed = sae.decode(features)
        #     error = activations.to(features.dtype) - reconstructed
            
        #     for feature_idx, max_activation, strength in zip(feature_idxs, max_activations, strengths):
        #         features[..., feature_idx] = max_activation * strength
            
        #     activations_hat = sae.decode(features) + error
        #     activations_hat = activations_hat.type_as(activations)
        # else:
        #     # 其他rank直接返回原activations
        #     activations_hat = activations

        # # 广播结果到所有ranks
        # if dist.is_initialized():
            # dist.broadcast(activations_hat, src=0)

        # TP>1，在这里已经聚合了，会重复执行？
        features = sae.encode(activations)
        reconstructed = sae.decode(features)
        error = activations.to(features.dtype) - reconstructed

        for feature_idx, max_activation, strength in zip(feature_idxs, max_activations, strengths):
            features[..., feature_idx] = max_activation * strength

        activations_hat = sae.decode(features) + error
        activations_hat = activations_hat.type_as(activations)

        if torch.is_tensor(output):
            return activations_hat
        else:
            return (activations_hat,) + output[1:] if len(output) > 1 else (activations_hat,)

    return hook_fn

@contextmanager
def add_hooks(
    module_forward_pre_hooks: list,
    module_forward_hooks: list,
    **kwargs
):
    """
    上下文管理器，用于添加和移除hooks，参考SAE-Reasoning2实现
    
    Args:
        module_forward_pre_hooks: 前置hook列表 [(module, hook_fn), ...]
        module_forward_hooks: 后置hook列表 [(module, hook_fn), ...]
        **kwargs: 传递给hook函数的额外参数
    """
    try:
        handles = []
        # 注册前置hooks
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        
        # 注册后置hooks
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        # 清理所有hooks
        for h in handles:
            h.remove()




# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        # SAE相关属性
        self.sae = None
        self.sae_config = {}
        self._sae_hooks = []
        breakpoint()
        self._load_sae_from_config(config)

    def _load_sae_from_config(self, config: DictConfig):
        """从配置加载SAE模型"""
        if not SAE_AVAILABLE:
            return
        
        # 检查配置中是否有SAE设置
        sae_config = getattr(config, 'sae', {})
        if not sae_config or not sae_config.get('enable', False):
            return
        
        try:
            sae_model_path = sae_config.get('path')
            if sae_model_path:
                print(f"🔄 Loading SAE from {sae_model_path}...")
                self.sae = SAE.load_from_pretrained(path=sae_model_path)
                self.sae_config = sae_config
                print(f"✅ SAE loaded successfully")
            else:
                # 支持从release和id加载
                sae_release = sae_config.get('release')
                sae_id = sae_config.get('id')
                if sae_release and sae_id:
                    print(f"🔄 Loading SAE from release: {sae_release}, id: {sae_id}")
                    self.sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
                    self.sae_config = sae_config
                    print(f"✅ SAE loaded successfully")
                    
        except Exception as e:
            print(f"⚠️ Failed to load SAE: {e}")
            self.sae = None
            self.sae_config = {}

    def _setup_sae_hooks(self) -> list:
        """设置SAE hooks，参考SAE-Reasoning2实现"""
        if self.sae is None:
            return []
        
        sae_hooks = []
        
        try:
            # 获取vLLM模型
            lm_model = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner.model
            
            # 获取SAE配置参数
            feature_idxs = self.sae_config.get('feature_idxs')  # 支持多特征
            if feature_idxs:
                feature_idxs = list(map(int, feature_idxs.split(',')))
                if len(feature_idxs) == 1:
                    feature_idx = feature_idxs[0]

            strengths = self.sae_config.get('strengths')
            if strengths:
                strengths = list(map(float, strengths.split(',')))
                if len(strengths) == 1:
                    strength = strengths[0]

            max_activations = self.sae_config.get('max_activations')
            if max_activations:
                max_activations = list(map(float, max_activations.split(',')))
                if len(max_activations) == 1:
                    max_activation = max_activations[0]

            # 确定hook层
            hook_layer = self.sae.cfg.hook_layer
            target_module = lm_model.model.layers[hook_layer]
            
            # 选择合适的hook类型
            if feature_idxs is not None and isinstance(feature_idxs, (list, tuple)):
                # 多特征干预
                hook_fn = get_multi_intervention_hook(
                    self.sae, 
                    feature_idxs=feature_idxs,
                    max_activations=max_activations,
                    strengths=strengths
                )
                print(f"🎯 Setup multi-feature SAE hook on layer {hook_layer} for features {feature_idxs}")
                
            else:
                # 单特征干预
                # if max_activation is not None:
                #     # 使用clamp hook
                #     direction = self.sae.W_dec[feature_idx].clone()
                #     hook_fn = get_clamp_hook(direction, max_activation, strength)
                #     print(f"🎯 Setup clamp SAE hook on layer {hook_layer} for feature {feature_idx}")
                # else:
                # 使用intervention hook
                hook_fn = get_intervention_hook(
                    self.sae, 
                    feature_idx=feature_idx, 
                    strength=strength,
                    max_activation=max_activation
                )
                print(f"🎯 Setup intervention SAE hook on layer {hook_layer} for feature {feature_idx}")
            
            sae_hooks.append((target_module, hook_fn))
            
        except Exception as e:
            print(f"⚠️ Failed to setup SAE hooks: {e}")
        
        return sae_hooks

    def _apply_sae_hooks(self):
        """
        应用SAE hooks到vLLM模型，返回hook上下文管理器
        
        Args:
            meta_info: 包含SAE配置的元信息
            
        Returns:
            hook_context: hook上下文管理器
        """
        if self.sae is None:
            return None
        
        try:
            # 设置SAE hooks
            sae_hooks = self._setup_sae_hooks()
            
            if not sae_hooks:
                return None
            
            # 返回hook上下文管理器，使用SAE-Reasoning2风格的add_hooks
            return add_hooks(
                module_forward_pre_hooks=[],  # 我们主要使用forward hooks
                module_forward_hooks=sae_hooks
            )
            
        except Exception as e:
            print(f"⚠️ Failed to apply SAE hooks: {e}")
            return None

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        
        # 检查是否启用SAE特征叠加
        breakpoint()
        sae_enabled = self.sae_config.get('enable', False) and self.sae is not None
        
        if sae_enabled:
            print("🔥 SAE enabled in vLLM rollout")
            print(f"   SAE config: {self.sae_config}")
            
            # 启用GlobalSAE
            GlobalSAE.use_sae = True
            
        else:
            # 禁用GlobalSAE
            GlobalSAE.use_sae = False
            if self.sae_config.get('enable', False) and self.sae is None:
                print("⚠️ SAE enabled in config but no SAE model loaded")

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # 应用SAE特征叠加（如果启用）
        sae_hook_context = None
        if sae_enabled:
            try:
                print("🎯 Applying SAE hooks to vLLM model...")
                sae_hook_context = self._apply_sae_hooks()
                if sae_hook_context:
                    print("✅ SAE hooks context created")
                else:
                    print("⚠️ Failed to create SAE hooks context")
                    sae_enabled = False
            except Exception as e:
                print(f"⚠️ Failed to apply SAE hooks: {e}")
                sae_enabled = False

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # 使用SAE hooks上下文管理器进行生成
            if sae_hook_context:
                with sae_hook_context:
                    print("🚀 Generating with SAE hooks active...")
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )
                print("✅ SAE generation completed, hooks removed")
            else:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # 添加SAE相关信息到meta_info
        meta_info = {}
        if sae_enabled:
            meta_info["sae_enabled"] = True
            meta_info["sae_config"] = self.sae_config
            if 'sae_params' in locals():
                meta_info.update(sae_params)
            print("✅ vLLM SAE rollout completed with SAE intervention")
        else:
            meta_info["sae_enabled"] = False

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
