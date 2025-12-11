set -ex

cd /mnt/shared-storage-user/suzhongling/xtuner_1204

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export CUDA_HOME=/usr/local/cuda
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=none
export NCCL_PXN_DISABLE=0
export NCCL_CUMEM_ENABLE=0

export TORCH_LOGS=recompiles 
export XTUNER_ROUTER_DEBUG=false 
export XTUNER_ACTIVATION_OFFLOAD=1
ulimit -u 65536
export TORCHINDUCTOR_COMPILE_THREADS=8
export NVSHMEM_IB_GID_INDEX=3

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONPATH=.

export QWEN3_MOE_PATH=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/c6a03d5f8cf5257df36b2dc539463e41ca39c507
export ALPACA_PATH=/mnt/shared-storage-user/llmrazor-share/data/alpaca

export TORCH_NCCL_AVOID_RECORD_STREAMS=True
export TORCHINDUCTOR_CACHE_DIR=/mnt/shared-storage-user/suzhongling/xtuner/cache/${NODE_RANK}


export XTUNER_USE_FA3=1
export NGPUS=256
export PACK_MAX_LENGTH=16384
export INTRA_LAYER_MICRO_BATCH=2
export EP_SIZE=8

export FIRST_K_DENSE_REPLACE=3
export NUM_LAYERS=61
export N_ROUTED_EXPERTS=256
export XTUNER_USE_NATIVE_RMSNORM=0
export USE_GROUPED_ROUTER=true
export ROUTER_N_GROUPS=8
export TORCH_COMPILE=true
export LOAD=true
export DISPATCHER=agrs_custom 

# Variables to be modified
export NVSHMEM_HOME=/mnt/shared-storage-user/llmrazor-share/data/suzhongling/environment/nvshmem/build/src
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$NVSHMEM_HOME/bin:$PATH

export NCCL_MAX_CTAS=24 # We need to control the max SM used by nccl
# export NCCL_MIN_CTAS=6
export DISTRIBUTED_COMMUNICATION_SM=8
export DISPATCHER=agrs_custom # Change dispatcher to customed agrs
export XTUNER_ENABLE_CUSTOM_COMMUNICATION=1
export SYMM_BUF_SIZE=0 # 4GB
# export USE_CUSTOM_AG_IN_FSDP=0 # 在 FSDP 中 使用自定义 All gather
# export USE_CUSTOM_RS_IN_FSDP=0 # 在 FSDP 中 使用自定义 Reduce scatter

export USE_CUSTOM_AG_IN_DISPATCHER=1 # 在 Dispatcher 中 使用自定义 All gather
export USE_CUSTOM_RS_IN_DISPATCHER=1 # 在 Dispatcher 中 使用自定义 Reduce scatter

# export BARRIER_FSDP_ON_COMP=0 # 在 FSDP compute 流中同步
export BARRIER_DISPATCHER_ON_COMP=0 # 在 Dispatcher compute 流中同步


export GRID_IB_AG=2
export GRID_IB_RS=2

export PYTHONPATH=$PYTHONPATH:/mnt/shared-storage-user/llmrazor-share/data/suzhongling/environment/ib_wrapper/local/lib/python3.12/dist-packages/ib_wrapper-2.0.0-py3.12-linux-x86_64.egg/
export PYTHONPATH=$PYTHONPATH:/mnt/shared-storage-user/suzhongling/AdaptiveGEMM/
export PYTHONPATH=$PYTHONPATH:/mnt/shared-storage-user/suzhongling/GroupedGEMM/local/lib/python3.12/dist-packages/grouped_gemm-1.1.4-py3.12-linux-x86_64.egg
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD
# cp _fsdp_collectives_custom_comm.py /usr/local/lib/python3.12/dist-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py
# cp _fsdp_param_group.py /usr/local/lib/python3.12/dist-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py


git config --global --add safe.directory /mnt/shared-storage-user/suzhongling/xtuner_1204

torchrun --nproc-per-node=8  \
    --master_addr=${MASTER_ADDR} \
    --master_port=6003 \
    --nnodes=${NODE_COUNT} \
    --node_rank=${NODE_RANK} \
    ci/scripts/ib_benchmark/customed_comm/ep8_tgs/test_sft_trainer.py \
    work_dirs/exp2_1_ep8_grouped_router_our_agrs_bench_tgs
