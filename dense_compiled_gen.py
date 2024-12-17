from xtuner._lite.accelerate.generate import contiguous_batching_generate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from xtuner._lite.accelerate import dispatch_hf_code
from xtuner._lite.parallel import setup_parallel, get_tp_mesh
from xtuner._lite.parallel.megatron.qwen2 import _tp_qwen2
from torch.distributed._tensor import Replicate, distribute_tensor
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               parallelize_module)
import copy


tp_size = 2
bs = 32
setup_parallel(tp_size=tp_size)
tp_mesh = get_tp_mesh()

path = '/cpfs01/shared/public/caoweihan/.cache/hub/models--Qwen--Qwen2.5-7B/snapshots/09a0bac5707b43ec44508eab308b0846320c1ed4/'
# path = '/cpfs01/shared/public/caoweihan/.cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/'
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    trust_remote_code=True,
    # num_hidden_layers=1,
    ).cuda()
if tp_size > 1:
    _tp_qwen2(model.model, tp_mesh)
    model = parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan={
            'lm_head': ColwiseParallel(output_layouts=Replicate(), ),
        })
dispatch_hf_code(model)

for name, module in model.named_modules():
    module.name = name

piece = 'Who are '
token_ids = tokenizer.encode(piece, return_tensors='pt').cuda().view(1, -1)
token_ids_list = [copy.deepcopy(token_ids) for _ in range(bs)]

out = contiguous_batching_generate(model, token_ids_list, use_compile=True, max_new_tokens=300, tp_size=tp_mesh.size())
print(tokenizer.decode(out[0]))
# breakpoint()
