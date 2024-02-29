# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.distributed as dist
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import get_peft_model, prepare_model_for_kbit_training
from torch import nn

from xtuner.engine._strategy.deepspeed import (get_sequence_parallel_world_size,
                                               get_sequence_parallel_group)
from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, make_inputs_require_grad,
                    traverse_dict)
import math
from transformers import AutoConfig, AutoModelForCausalLM
import torch


class SupervisedFinetune(BaseModel):

    def __init__(self,
                 llm,
                 lora=None,
                 peft_model=None,
                 use_activation_checkpointing=True,
                 use_varlen_attn=False):
        super().__init__()
        with LoadWoInit():
            # todo
            # model_max_length = 32768
            # config = AutoConfig.from_pretrained(llm.pretrained_model_name_or_path, trust_remote_code=True)
            # orig_rope_scaling = getattr(config, "rope_scaling", None)
            # if orig_rope_scaling is None:
            #     orig_rope_scaling = {"factor": 1}
            # orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
            # orig_ctx_len = getattr(config, "max_position_embeddings", None)
            # if orig_ctx_len:
            #     orig_ctx_len *= orig_rope_scaling_factor
            #     if model_max_length > orig_ctx_len:
            #         scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
            #         config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            
            # self.llm = AutoModelForCausalLM.from_pretrained(
            #     llm.pretrained_model_name_or_path,
            #     config=config,
            #     trust_remote_code=True,
            #     torch_dtype=torch.bfloat16,
            #     attn_implementation='flash_attention_2')

            self.llm = self._build_from_cfg_or_module(llm)

        self.llm.config.use_cache = False
        dispatch_modules(self.llm, use_varlen_attn=use_varlen_attn)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)

            # enable gradient checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        if isinstance(lora, dict) or isinstance(lora, Config) or isinstance(
                lora, ConfigDict):
            self.lora = BUILDER.build(lora)
        else:
            self.lora = lora
        self.peft_model = peft_model
        self.use_lora = lora is not None
        if self.use_lora:
            self._prepare_for_lora(peft_model, use_activation_checkpointing)

        self._is_init = True
        # Determines whether to calculate attention based on the
        # seq_len dimension (use_varlen_attn = False) or the actual length of
        # the sequence.
        self.use_varlen_attn = use_varlen_attn
        for name, module in self.named_modules():
            module.name = name

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def _prepare_for_lora(self,
                          peft_model=None,
                          use_activation_checkpointing=True):
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if self.lora.target_modules is None:
            modules = find_all_linear_names(self.llm)
            self.lora.target_modules = modules

        self.llm = get_peft_model(self.llm, self.lora)
        if peft_model is not None:
            _ = load_checkpoint(self, peft_model)

    def init_weights(self):
        pass

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict
    
    def compute_sequence_parallel_loss(self, data):
        outputs = self.llm(**data)
        sequence_parallel_group = get_sequence_parallel_group()

        labels = data['labels']
        num_tokens = (labels != -100).sum()
        loss = outputs.loss * num_tokens
        if num_tokens == 0:
            # convert nan to 0 just for logging
            loss = torch.nan_to_num(outputs.loss)
        dist.all_reduce(loss, group=sequence_parallel_group)
        dist.all_reduce(num_tokens, group=sequence_parallel_group)

        loss = loss / num_tokens
        loss_dict = {'loss': loss}
        return loss_dict

    def compute_loss(self, data, data_samples=None):
        # return self.compute_sequence_parallel_loss(data)
        if get_sequence_parallel_world_size() > 1:
            return self.compute_sequence_parallel_loss(data)
        else:
            outputs = self.llm(**data)
            loss_dict = {'loss': outputs.loss}
            return loss_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if not self.use_lora:
            return state_dict
        to_return = get_peft_model_state_dict(self.llm, state_dict=state_dict)
        return OrderedDict(to_return)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
