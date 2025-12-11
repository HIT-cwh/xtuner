from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
    FSDPConfig,
)
from xtuner.v1.train import TrainerConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig

# model config

EP_SIZE = 8
SP_SIZE = 1


INTRA_LAYER_MICRO_BATCH = 2
SEED = 1024
LR = 8e-5
LR_MIN = 8e-6
SEQ_LEN = 32768
GLOBAL_BS = 256
TOTAL_EPOCH = 1

CHECKPOINT_INTERVAL = 2000

# HF_MODEL_PATH = "/mnt/shared-storage-user/songdemin/user/puyudilivery/ckpts/ai4s-moe/Qwen3-MoE/pretrain/cosine_decay/VanGogh_30B_A3_cosine_decay_20250902b_4k_from_scratch_fp8_0_0/20250905142937/hf-125000"
# CACHE_DIR = "/mnt/shared-storage-user/songdemin/user/puyudilivery/ckpts/ai4s-moe/Qwen3-MoE/cache_new"
# WORK_DIR = "..."  # 需设置
# dataset and dataloader config
HF_MODEL_PATH = "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/89e5e822ba31507f5f79dc3422c7c5345c422737/"
CACHE_DIR = "./cache_ep8"
WORK_DIR = "./work_dirs/ib_benchmark/exp2_2_ep8_grouped_router_our_agrs_bench_acc"


dataset_config = [
    {
        "dataset": DatasetConfig(
            name="data1",
            anno_path="/mnt/shared-storage-user/caoweihan/data/open-thoughts/",
            sample_ratio=1.0,
            cache_dir=CACHE_DIR,
        ),
        "tokenize_fn": OpenaiTokenizeFunctionConfig(max_length=SEQ_LEN, chat_template="qwen3"),
    },
    
]

dataloader_config = DataloaderConfig(
    pack_max_length=SEQ_LEN,
    num_workers=4,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=LR, weight_decay=0.1)
lr_cfg = LRConfig(lr_type="cosine", lr_min=LR_MIN, warmup_ratio=0.03)

fsdp_cfg = FSDPConfig(
    torch_compile=True, 
    ep_size=EP_SIZE, 
)
model_cfg = get_model_config_from_hf(HF_MODEL_PATH)
model_cfg.ep_size = EP_SIZE
model_cfg.dispatcher = "agrs_custom"
model_cfg.router.use_grouped_router = True
model_cfg.router.router_n_groups = 8
# trainer config
trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    loss_cfg=CELossConfig(mode="liger", chunk_size=1024),
    global_batch_size=GLOBAL_BS,
    sp_size=SP_SIZE,
    intra_layer_micro_batch=INTRA_LAYER_MICRO_BATCH,
    total_epoch=1,
    load_from=HF_MODEL_PATH,
    seed=42,
    profile_step=1,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    hf_interval=CHECKPOINT_INTERVAL,
    work_dir=WORK_DIR,
    tokenizer_path=HF_MODEL_PATH,
    strict_load=True,
    skip_checkpoint_validation=True,
)
