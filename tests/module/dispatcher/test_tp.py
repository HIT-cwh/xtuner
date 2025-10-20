import torch
from torch.testing._internal.common_distributed import DistributedTestBase
from xtuner.v1.module.dispatcher.base import NaiveDispatcher, DispacherInterface
from xtuner.v1.module.dispatcher.torch_all2all import TorchAll2AllDispatcher
from xtuner.v1.module.dispatcher.moe_tp import MoETPDispatcher
import parametrize


import os


EP_SIZE = 8


# def mock_experts(hidden_states: torch.Tensor, tokens_per_exprts: torch.Tensor):
#     return hidden_states

def mock_experts(x, w1, w2):
    hidden_states = torch.matmul(x, w1)
    hidden_states = torch.matmul(hidden_states, w2)
    return hidden_states


class TestNoETorchAll2AllDispatcher(DistributedTestBase):
    @parametrize.parametrize("dtype,device", [(torch.float32, "cuda")])
    def test_dispatch_and_combine(self, dtype, device):
        self.create_pg(device)
        num_experts = 16
        noep_dispatcher = NaiveDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
        )

        all2all_dispatcher = MoETPDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
            process_group=torch.distributed.group.WORLD
        )

        seq_len = 32
        hidden_size = 128
        topk_experts = 4
        hidden_states = torch.randn(seq_len, hidden_size).to(device).to(dtype)
        topk_idx = torch.randint(0, num_experts, (seq_len, topk_experts)).to(device).to(torch.int32)
        torch.manual_seed(0)
        topk_weights = torch.randn(seq_len, topk_experts).to(device).to(torch.float32)
        rank = torch.distributed.get_rank()

        w1 = torch.randn(128, 64, device=device, dtype=dtype)
        w2 = torch.randn(64, 128, device=device, dtype=dtype)
        w1_tp = w1.clone()
        w1_tp = w1_tp[:, 8 * rank:8 * (rank + 1)]
        w2_tp = w2.clone()
        w2_tp = w2_tp[8 * rank:8 * (rank + 1), :]

        noep_results = self._dispatcher_call(
            dispatcher=noep_dispatcher,
            hidden_states=hidden_states,
            topk_ids=topk_idx,
            topk_weights=topk_weights,
            w1=w1,
            w2=w2,
        )
        all2all_results = self._dispatcher_call(
            dispatcher=all2all_dispatcher,
            hidden_states=hidden_states,
            topk_ids=topk_idx,
            topk_weights=topk_weights,
            w1=w1_tp,
            w2=w2_tp,
        )
        torch.distributed.breakpoint()

        self.assertTrue(torch.allclose(noep_results["hidden_states"], all2all_results["hidden_states"], atol=1e-6, rtol=1e-4))

    def _dispatcher_call(
            self,
            dispatcher: DispacherInterface,
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
    ):
        pre_dispatched = dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
        )
        dispatched = dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=topk_weights,
            decoding=False,
        )
        post_dispatched = dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )
        # experts_results = mock_experts(
        #     hidden_states=post_dispatched["hidden_states"],
        #     tokens_per_exprts=post_dispatched["tokens_per_expert"],
        # )
        experts_results = mock_experts(
            x=post_dispatched["hidden_states"],
            w1=w1,
            w2=w2,
        )
        pre_combined = dispatcher.combine_preprocess(
            hidden_states=experts_results,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
        )
        combined = dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        return dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
