import math
from typing import Literal, TypedDict, cast

import ray
import torch
import random
import heapq

from xtuner.v1.data_proto.sequence_context import SequenceContext

from .worker import TrainingWorker


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    advantage: float


def ceildiv(a, b):
    return -(a // -b)


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


@ray.remote
class TrainingController:
    def __init__(self, workers: list[TrainingWorker]) -> None:
        self.workers = workers

    def _get_pack_infos(self, dataset, num_tokens, target, random=None):
        inds = list(range(len(dataset)))
        if random is not None:
            random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        longest = 0

        pack_infos = []
        for shfl_i in inds:
            if num_tokens[shfl_i] + sum(length_buffer) <= target:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                longest = max(longest, num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        "indices": item_buffer,
                        "longest": int(longest),
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                longest = num_tokens[shfl_i]

        if len(item_buffer) > 0:
            info = {
                "indices": item_buffer,
                "longest": int(longest),
            }

            pack_infos.append(info)

        return pack_infos

    def _packing(self, data_batches, pack_max_length):
        pack_infos = self._get_pack_infos(
            data_batches,
            [data["seq_ctx"].input_ids.numel() for data in data_batches],
            pack_max_length,
        )
        packed_data_batches = []
        for pack_info in pack_infos:
            indices = pack_info["indices"]
            total_len = sum([data_batches[i]["seq_ctx"].input_ids.shape[1] for i in indices])
            pad_len = pack_max_length - total_len
            seq_ctx_list = [data_batches[i]["seq_ctx"] for i in indices]
            label_list = [data_batches[i]["shifted_labels"] for i in indices]
            advantage_list = [data_batches[i]["advantage"] for i in indices]
            if pad_len > 0:
                # Reduce the attn calculation time by using multiple short sequence packs
                pad_tokens = tuple(
                    torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
                    for _ in range(pad_len // 1024)
                )
                if pad_len % 1024 > 0:
                    pad_tokens = pad_tokens + (
                        torch.zeros(1, pad_len % 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu"),
                    )
                pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")
                pad_seq_ctx.num_padding = pad_len
                pad_labels = torch.full(
                    (1, pad_len),
                    -100,
                    dtype=data_batches[0]["shifted_labels"].dtype,
                    device=data_batches[0]["shifted_labels"].device,
                )
                seq_ctx_list.append(pad_seq_ctx)
                label_list.append(pad_labels)
                advantage_list.extend(
                    [-100] * math.ceil(pad_len / 1024)
                )  # can be any number, pad tokens are excluded from the calculation of the loss function.

            seq_ctx = SequenceContext.pack(seq_ctx_list)
            shifted_labels = torch.cat(label_list, dim=1)  # (1, max_len)
            advantages = torch.tensor(advantage_list).float().unsqueeze(0)  # (1, num_samples)
            cu_seq_lens_q = seq_ctx.cu_seq_lens_q
            num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
            advantages = torch.repeat_interleave(advantages, num_tokens, dim=1)  # (1, max_len)

            packed_data_batches.append(
                {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": shifted_labels,
                    "advantages": advantages,
                }
            )
        return packed_data_batches

    def _grouped_by_max_length(self, packed_data_batches):
        # sort 过后可能第一个 batch 会有很多 pad tokens，因为最后一个 pack 可能只有少量真实数据。
        # 比如组成了 16 个 pack，第 16 个 pack 可能只有几条真实数据，剩下的都是 pad tokens。
        # 排序后这条 pack 会被放在最前面，导致 rank0 的第一个 step 消耗的有效 token 数往往少于其他 rank，是正常现象。
        return sorted(packed_data_batches, key=lambda x: x["seq_ctx"].max_length_q, reverse=True)

    def _pack_and_pad1(self, data_batches: list[list[ColateItem]], pack_max_length: int) -> list[ColateItem]:
        data_batches = cast(list[ColateItem], sum(data_batches, []))
        # random.shuffle(data_batches)
        ##########
        packed_data_batches_all_ranks = []
        max_len = 0
        for i in range(8):
            packed_data_batches = self._packing(data_batches[i*64:(i+1)*64], pack_max_length)
            packed_data_batches_all_ranks.append(packed_data_batches)
            max_len = max(max_len, len(packed_data_batches))
        for i in range(8):
            num_pad = max_len - len(packed_data_batches_all_ranks[i])
            if num_pad > 0:
                # Reduce the attn calculation time by using multiple short sequence packs
                pad_tokens = tuple(
                    torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
                    for _ in range(pack_max_length // 1024)
                )
                pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")
                pad_seq_ctx.num_padding = pack_max_length
                pad_shifted_labels = torch.full(
                    (1, pack_max_length),
                    -100,
                    dtype=packed_data_batches[0]["shifted_labels"].dtype,
                    device="cpu",
                )
                pad_advantages = torch.full(
                    (1, pack_max_length),
                    -100,
                    dtype=packed_data_batches[0]["advantages"].dtype,
                    device="cpu",
                )
                pad_data = {
                    "seq_ctx": pad_seq_ctx,
                    "shifted_labels": pad_shifted_labels,
                    "advantages": pad_advantages,
                }
                pad_data_samples = [pad_data for _ in range(num_pad)]
                packed_data_batches_all_ranks[i] = packed_data_batches_all_ranks[i] + pad_data_samples
        return packed_data_batches_all_ranks
        ###########
        packed_data_batches = self._packing(data_batches, pack_max_length)
        # packed_data_batches = self._grouped_by_max_length(packed_data_batches)

        num_packed_data_batches = len(packed_data_batches)
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]
        dp_size = len(self.workers) // data_replicate_size
        pad_num = math.ceil(num_packed_data_batches / dp_size) * dp_size - num_packed_data_batches
        if pad_num > 0:
            # Reduce the attn calculation time by using multiple short sequence packs
            pad_tokens = tuple(
                torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
                for _ in range(pack_max_length // 1024)
            )
            if pack_max_length % 1024 > 0:
                pad_tokens = pad_tokens + (
                    torch.zeros(
                        1, pack_max_length % 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu"
                    ),
                )
            pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")  # type: ignore
            pad_seq_ctx.num_padding = pack_max_length
            pad_shifted_labels = torch.full(
                (1, pack_max_length),
                -100,
                dtype=packed_data_batches[0]["shifted_labels"].dtype,
                device="cpu",
            )
            pad_advantages = torch.full(
                (1, pack_max_length),
                -100,
                dtype=packed_data_batches[0]["advantages"].dtype,
                device="cpu",
            )
            pad_data = {
                "seq_ctx": pad_seq_ctx,
                "shifted_labels": pad_shifted_labels,
                "advantages": pad_advantages,
            }
            pad_data_samples = [pad_data for _ in range(pad_num)]
            packed_data_batches = packed_data_batches + pad_data_samples
        return packed_data_batches
    
    def fit1(self, data_batches: list[list[ColateItem]], pack_max_length: int, optimizer_steps: int, rollout_idx: int):
        handles = []
        for worker_idx, worker in enumerate(self.workers):
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    data_batches=[],
                    rollout_idx=rollout_idx,
                )
            )
        ray.get(handles)
        return
    
    def _pack_and_pad(self, data_batches: list[list[ColateItem]], pack_max_length: int) -> list[ColateItem]:
        # dynamic batch
        data_batches = cast(list[ColateItem], sum(data_batches, []))
        num_micro_batches_max = 0
        for i in range(8):
            data = data_batches[i*64:(i+1)*64]
            seq_len_effective = []
            for j in range(64):
                seq_len_effective.append(data[j]['shifted_labels'].numel())
            total_seqlen = sum(seq_len_effective)
            num_micro_batches = min(64, ceildiv(total_seqlen, 20480))
            num_micro_batches_max = max(num_micro_batches_max, num_micro_batches)
        
        data_batches_all_ranks = []
        for i in range(8):
            data = data_batches[i*64:(i+1)*64]
            seq_len_effective = []
            for j in range(64):
                seq_len_effective.append(data[j]['shifted_labels'].numel())
            total_seqlen = sum(seq_len_effective)
            micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches_max, equal_size=False)
            micro_bsz_idx.sort(
                key=lambda partition: (
                    sum(seq_len_effective[idx] ** 2 for idx in partition),
                    min(partition) if partition else 0,
                ),
                reverse=True,
            )
            
            for jj, idxes in enumerate(micro_bsz_idx):
                seq_ctx_list = []
                shifted_labels_list = []
                advantage_list = []
                for idx in idxes:
                    seq_ctx_list.append(data[idx]["seq_ctx"])
                    shifted_labels_list.append(data[idx]["shifted_labels"])
                    advantage_list.append(data[idx]["advantage"])
                seq_ctx = SequenceContext.pack(seq_ctx_list)
                shifted_labels = torch.cat(shifted_labels_list, dim=1)  # (1, max_len)
                cu_seq_lens_q = seq_ctx.cu_seq_lens_q
                num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
                advantages = torch.tensor(advantage_list).float().unsqueeze(0)  # (1, num_samples)
                advantages = torch.repeat_interleave(advantages, num_tokens, dim=1)  # (1, max_len)
                if jj == 0:
                    data_batches_all_ranks.append([
                        {
                            "seq_ctx": seq_ctx,
                            "shifted_labels": shifted_labels,
                            "advantages": advantages,
                        }
                    ])
                else:
                    data_batches_all_ranks[i].append({
                        "seq_ctx": seq_ctx,
                        "shifted_labels": shifted_labels,
                        "advantages": advantages,
                    })
        return data_batches_all_ranks

    def fit(self, data_batches: list[list[ColateItem]], pack_max_length: int, optimizer_steps: int, rollout_idx: int):
        optimizer_steps = min(optimizer_steps, len(data_batches))
        n_groups_per_step = math.ceil(len(data_batches) / optimizer_steps)
        packed_data_batches_all_steps: list[list[ColateItem]] = []
        for i in range(optimizer_steps):
            packed_data_batches_all_steps.append(
                self._pack_and_pad(
                    data_batches[i * n_groups_per_step : (i + 1) * n_groups_per_step],
                    pack_max_length,
                )
            )
        # breakpoint()
        
        handles = []
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]
        dp_size = len(self.workers) // data_replicate_size
        for worker_idx, worker in enumerate(self.workers):

            packed_data_batches_all_steps_cur = [data[worker_idx] for data in packed_data_batches_all_steps]

            # packed_data_batches_all_steps_cur = [data[(worker_idx // data_replicate_size) :: dp_size] for data in packed_data_batches_all_steps]
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    data_batches=packed_data_batches_all_steps_cur,
                    rollout_idx=rollout_idx,
                )
            )
        ray.get(handles)
        return

        # packed_data_batches = self._packing(data_batches, pack_max_length)
        # packed_data_batches = self._grouped_by_max_length(packed_data_batches)

        # # todo: support round up
        # num_packed_data_batches = len(packed_data_batches)
        # data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]
        # dp_size = len(self.workers) // data_replicate_size
        # pad_num = math.ceil(num_packed_data_batches / dp_size) * dp_size - num_packed_data_batches
        # if pad_num > 0:
        #     # Reduce the attn calculation time by using multiple short sequence packs
        #     pad_tokens = tuple(
        #         torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
        #         for _ in range(pack_max_length // 1024)
        #     )
        #     if pack_max_length % 1024 > 0:
        #         pad_tokens = pad_tokens + (
        #             torch.zeros(
        #                 1, pack_max_length % 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu"
        #             ),
        #         )
        #     pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")  # type: ignore
        #     pad_seq_ctx.num_padding = pack_max_length
        #     pad_shifted_labels = torch.full(
        #         (1, pack_max_length),
        #         -100,
        #         dtype=packed_data_batches[0]["shifted_labels"].dtype,
        #         device="cpu",
        #     )
        #     pad_advantages = torch.full(
        #         (1, pack_max_length),
        #         -100,
        #         dtype=packed_data_batches[0]["advantages"].dtype,
        #         device="cpu",
        #     )
        #     pad_data = {
        #         "seq_ctx": pad_seq_ctx,
        #         "shifted_labels": pad_shifted_labels,
        #         "advantages": pad_advantages,
        #     }
        #     pad_data_samples = [pad_data for _ in range(pad_num)]
        #     packed_data_batches = packed_data_batches + pad_data_samples

        # print(f"len(packed_data_batches): {len(packed_data_batches)}")

        # handles = []
        # for worker_idx, worker in enumerate(self.workers):
        #     handles.append(
        #         worker.fit.remote(  # type: ignore[attr-defined]
        #             data_batches=packed_data_batches[(worker_idx // data_replicate_size) :: dp_size],
        #             rollout_idx=rollout_idx,
        #         )
        #     )
        # ray.get(handles)

    def offload(self, target: Literal["model", "optimizer", "all"] = "all"):
        if target == "model":
            ray.get([worker.offload_model.remote() for worker in self.workers])  # type: ignore
        elif target == "optimizer":
            ray.get([worker.offload_optimizer.remote() for worker in self.workers])  # type: ignore
        elif target == "all":
            ray.get([worker.offload_model.remote() for worker in self.workers])  # type: ignore
            ray.get([worker.offload_optimizer.remote() for worker in self.workers])  # type: ignore
        return

    def onload(self, target: Literal["model", "optimizer", "all"] = "all"):
        """Onload the model or optimizer of the training workers."""
        if target == "model":
            ray.get([worker.onload_model.remote() for worker in self.workers])  # type: ignore
        elif target == "optimizer":
            ray.get([worker.onload_optimizer.remote() for worker in self.workers])  # type: ignore
        elif target == "all":
            ray.get([worker.onload_model.remote() for worker in self.workers])  # type: ignore
            ray.get([worker.onload_optimizer.remote() for worker in self.workers])  # type: ignore
        return

    def update_rollout_info(self, info_dict):
        ray.get([worker.update_rollout_info.remote(**info_dict) for worker in self.workers])  # type: ignore[attr-defined]

    def update_weights(self):
        """Update the weights of the training workers."""
        handles = [worker.update_weights.remote() for worker in self.workers]
        ray.get(handles)
        return

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        handles = [worker.save_hf.remote(hf_dir, save_dtype) for worker in self.workers]  # type: ignore
        ray.get(handles)
        return
