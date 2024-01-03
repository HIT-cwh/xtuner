# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmengine import MessageHub
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import PeftType, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .utils import LoadWoInit, find_all_linear_names, traverse_dict
from mmengine import print_log


class SupervisedFinetune(BaseModel):

    def __init__(self,
                 llm,
                 lora=None,
                 peft_model=None,
                 use_activation_checkpointing=True,
                 use_local_attn=True,
                 debug=False):
        super().__init__()
        with LoadWoInit():
            # if 'internlm' in llm.pretrained_model_name_or_path:
            hf_config = AutoConfig.from_pretrained(
                llm.pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            # assert hf_config.rms_norm_eps == 1e-6
            # hf_config.rms_norm_eps = 1e-5
            # print('set rms_norm_eps to 1e-5')
            assert hf_config.rotary['type'] == "dynamic"
            hf_config.rotary['type'] = "origin"
            print_log('set rotary type to origin', 'current')
            self.llm = AutoModelForCausalLM.from_pretrained(
                torch_dtype=torch.bfloat16,
                pretrained_model_name_or_path=llm.pretrained_model_name_or_path,
                trust_remote_code=True,
                config=hf_config)
            # else:
            #     self.llm = self._build_from_cfg_or_module(llm)
        for name, module in self.llm.named_modules():
            module.name = name
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)

            # enable gradient checkpointing for memory efficiency
            self.llm.gradient_checkpointing_enable()

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
        self.use_local_attn = use_local_attn
        self.cnt = 0
        self.debug = debug

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
        if self.debug and self.cnt == 4:
            rank = dist.get_rank()
            if rank == 0:
                torch.save(self.llm.state_dict(), 'saved/iter1.pth')
            else:
                import time
                time.sleep(200)
            assert False

        if self.use_local_attn:
            message_hub = MessageHub.get_instance('for_flash_attn')
            rank = dist.get_rank()
            # saved_dict = torch.load(f'/mnt/petrelfs/caoweihan/projects/train_internlm/saved/rank_{rank}_model_in.pth', map_location='cpu')
            # cu_seqlens = saved_dict['cu_seqlens'].cuda()
            # cu_seqlens = [cu_seqlens[0]]
            # input_ids = saved_dict['input_ids'].cuda()
            # indexes = saved_dict['indexes'].cuda()
            # max_seqlen = indexes.max().item() + 1
            # data['input_ids'].data = input_ids.data
            # message_hub.update_info(f'cumulative_len_rank_{rank}',
            #                         cu_seqlens)
            # message_hub.update_info(f'indexes_rank_{rank}',
            #                         indexes)
            # message_hub.update_info(f'max_seqlen_rank_{rank}',
            #                         max_seqlen)
            # data.pop('cumulative_len')
            # data.pop('indexes')
            # data.pop('max_seqlen')

            message_hub.update_info(f'cumulative_len_rank_{rank}',
                                    data.pop('cumulative_len'))
            message_hub.update_info(f'indexes_rank_{rank}',
                                    data.pop('indexes'))
            message_hub.update_info(f'max_seqlen_rank_{rank}',
                                    data.pop('max_seqlen'))
        else:
            data.pop('cumulative_len', None)
            data.pop('indexes', None)
            data.pop('max_seqlen', None)
        self.cnt += 1

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

    def compute_loss(self, data, data_samples=None):
        # rank = dist.get_rank()
        # shift_labels = torch.load(f'/mnt/petrelfs/caoweihan/projects/train_internlm/saved/rank_{rank}_shift_labels.pth', map_location='cpu')
        # shift_labels = shift_labels[:-1]
        # shift_labels = torch.cat([torch.tensor([-100], dtype=shift_labels.dtype, device=shift_labels.device), shift_labels])
        # data['labels'] = shift_labels.to(dtype=data['labels'].dtype, device=data['labels'].device).reshape(1, -1)
        outputs = self.llm(**data)
        if self.debug:
            rank = dist.get_rank()
            if not hasattr(self, 'loss_cnt'):
                self.loss_cnt = 1
            else:
                self.loss_cnt += 1
            torch.save(outputs.loss, f'./saved/rank_{rank}_loss_cnt_{self.loss_cnt}_insft.pth')
        #     torch.save(outputs.logits, f'./saved/rank_{rank}_logits_cnt_{self.loss_cnt}.pth')

        loss_dict = {'loss': outputs.loss}
        # assert False
        return loss_dict

    def state_dict(self, destination=None, prefix='', keep_vars=False):

        def get_peft_model_state_dict(model,
                                      state_dict=None,
                                      adapter_name='default'):
            # Modified from `https://github.com/huggingface/peft/blob/main/src
            # /peft/utils/save_and_load.py`

            config = model.peft_config[adapter_name]
            if state_dict is None:
                state_dict = model.state_dict()
            if config.peft_type == PeftType.LORA:
                # adapted from `https://github.com/microsoft/LoRA/blob/main/
                # loralib/utils.py`
                # to be used directly with the state dict which is necessary
                # when using DeepSpeed or FSDP
                bias = config.bias
                if bias == 'none':
                    to_return = {
                        k: state_dict[k]
                        for k in state_dict if 'lora_' in k
                    }
                elif bias == 'all':
                    to_return = {
                        k: state_dict[k]
                        for k in state_dict if 'lora_' in k or 'bias' in k
                    }
                elif bias == 'lora_only':
                    to_return = {}
                    for k in state_dict:
                        if 'lora_' in k:
                            to_return[k] = state_dict[k]
                            bias_name = k.split('lora_')[0] + 'bias'
                            if bias_name in state_dict:
                                to_return[bias_name] = state_dict[bias_name]
                else:
                    raise NotImplementedError
                to_return = {
                    k: v
                    for k, v in to_return.items()
                    if (('lora_' in k and adapter_name in k) or ('bias' in k))
                }
            else:
                # Currently we only support lora
                raise NotImplementedError
            if model.modules_to_save is not None:
                for key, value in state_dict.items():
                    if any(f'{module_name}.modules_to_save.{adapter_name}' in
                           key for module_name in model.modules_to_save):
                        to_return[key] = value

            return to_return

        if not self.use_lora:
            return super().state_dict()
        state_dict = super().state_dict()
        to_return = get_peft_model_state_dict(self.llm, state_dict=state_dict)
        return OrderedDict(to_return)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
