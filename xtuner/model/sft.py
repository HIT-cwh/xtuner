# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from packaging import version
from peft import get_peft_model, prepare_model_for_kbit_training
from torch import nn

from xtuner.engine._strategy.deepspeed import (get_sequence_parallel_group,
                                               get_sequence_parallel_rank,
                                               get_sequence_parallel_world_size
                                               )
from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, make_inputs_require_grad,
                    traverse_dict)


class _VocabSequenceParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_seq_parallel_logits, target, label_smoothing=0.0):
        # vocab_seq_parallel_logits: [S/P, B, V]
        # target: [S/P, B]
        # return: [S, B]

        # Need softmax for backward
        softmax = torch.nn.functional.softmax(
            vocab_seq_parallel_logits, dim=-1)
        ctx.vocab_size = vocab_seq_parallel_logits.size(2)
        loss = torch.nn.functional.nll_loss(
            softmax.log().view(-1, ctx.vocab_size),
            target.view(-1),
            reduction='none')
        if dist.get_rank() in (0, 1):
            torch.save(loss.cpu(), f'rank_{dist.get_rank()}.pth')

        ctx.seqlen = vocab_seq_parallel_logits.size(
            0) * get_sequence_parallel_world_size()
        batch_size = vocab_seq_parallel_logits.size(1)

        loss_all = torch.empty(
            ctx.seqlen,
            batch_size,
            dtype=vocab_seq_parallel_logits.dtype,
            device=vocab_seq_parallel_logits.device)
        if version.parse(torch.__version__) >= version.parse('1.13'):
            torch.distributed.all_gather_into_tensor(
                loss_all, loss, group=get_sequence_parallel_group())
        else:
            torch.distributed._all_gather_base(
                loss_all, loss, group=get_sequence_parallel_group())

        ctx.save_for_backward(softmax, target)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors

        step_seqlen = ctx.seqlen // get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        grad_output_part = grad_output[step_seqlen * sp_rank:step_seqlen *
                                       (sp_rank + 1), :]

        grad_input = softmax
        grad_2d = grad_input.view(-1, ctx.vocab_size)
        arange_1d = torch.arange(
            start=0, end=grad_2d.size()[0], device=grad_2d.device)

        grad_2d[arange_1d, target.view(-1)] -= 1
        grad_input.mul_(grad_output_part.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits,
                                          target,
                                          label_smoothing=0.0):
    return _VocabSequenceParallelCrossEntropy.apply(vocab_parallel_logits,
                                                    target, label_smoothing)


class SupervisedFinetune(BaseModel):

    def __init__(self,
                 llm,
                 lora=None,
                 peft_model=None,
                 use_activation_checkpointing=True,
                 use_varlen_attn=False):
        super().__init__()
        with LoadWoInit():
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

    def compute_loss(self, data, data_samples=None):
        labels = data.pop('labels')
        outputs = self.llm(**data)

        logits = outputs.logits  # b, s/p, dim
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.transpose(0, 1).contiguous()  # s/p, b, dim
        shift_labels = shift_labels.transpose(
            0, 1).contiguous()  # [b s/p] => [s/p b]
        loss = vocab_sequence_parallel_cross_entropy(shift_logits.float(),
                                                     shift_labels)
        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        num_tokens = (shift_labels != -100).sum()
        sequence_parallel_group = get_sequence_parallel_group()
        dist.all_reduce(num_tokens, group=sequence_parallel_group)
        loss = loss.sum() / num_tokens

        # labels = data['labels']
        # num_tokens = (labels != -100).sum()
        # sequence_parallel_group = get_sequence_parallel_group()
        # with open('debug.txt', 'a+') as f:
        #     f.write(f'{num_tokens}, {dist.get_rank()} \n')
        # loss = outputs.loss * num_tokens
        # dist.all_reduce(loss, group=sequence_parallel_group)
        # dist.all_reduce(num_tokens, group=sequence_parallel_group)
        # loss = loss / num_tokens
        loss_dict = {'loss': loss}
        # loss_dict = {'loss': outputs.loss}
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
