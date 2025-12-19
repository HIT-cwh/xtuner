set -ex
cd /mnt/shared-storage-user/suzhongling/xtuner_1204

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export CUDA_HOME=/usr/local/cuda
# export NCCL_DEBUG=INFO

export XTUNER_USE_FA3=1
export TORCH_LOGS=recompiles 
export XTUNER_ROUTER_DEBUG=false 
export XTUNER_ACTIVATION_OFFLOAD=0
ulimit -u 65536
export TORCHINDUCTOR_COMPILE_THREADS=8
export NVSHMEM_IB_GID_INDEX=3

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONPATH=.

export TORCH_NCCL_AVOID_RECORD_STREAMS=True
export TORCHINDUCTOR_CACHE_DIR=/mnt/shared-storage-user/caoweihan/cache/${NODE_RANK}

# Variables to be modified
export NVSHMEM_HOME=/mnt/shared-storage-user/llmrazor-share/data/suzhongling/environment/nvshmem/build/src
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$NVSHMEM_HOME/bin:$PATH

export NCCL_MAX_CTAS=24 # We need to control the max SM used by nccl
export DISTRIBUTED_COMMUNICATION_SM=4

export DISPATCHER=agrs 
export XTUNER_ENABLE_CUSTOM_COMMUNICATION=1
export BARRIER_FSDP_ON_COMP=1
export SCALE_RS_IN_FSDP=0
# # Check if either variable is set to 1 (non-zero)
# if [ "$BARRIER_FSDP_ON_COMP" = "1" ] || [ "$SCALE_RS_IN_FSDP" = "1" ]; then
#     export BARRIER_FSDP_ON_COMP=1
# fi

# Set symmetric buffer size for distributed training
# Value of 0 enables dynamic resizing during runtime based on actual requirements
export SYMM_BUF_SIZE=0 # Auto-resize symmetric buffer during runtime if smaller than required
export USE_CUSTOM_AG_IN_FSDP=1 # 使用自定义 All gather
export USE_CUSTOM_RS_IN_FSDP=1 # 使用自定义 Reduce scatter
export GRID_IB_AG=4
export GRID_IB_RS=4
export SELECT_COMM_SM_IN_FSDP=1

export PYTHONPATH=$PYTHONPATH:/mnt/shared-storage-user/llmrazor-share/data/suzhongling/environment/ib_wrapper/local/lib/python3.12/dist-packages/ib_wrapper-2.0.0-py3.12-linux-x86_64.egg/
export PYTHONPATH=$PYTHONPATH:/mnt/shared-storage-user/suzhongling/AdaptiveGEMM/
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD

# Check if custom communication is enabled
if [ "$XTUNER_ENABLE_CUSTOM_COMMUNICATION" = "1" ]; then
    echo "Copying custom FSDP communication files..."
    cp _fsdp_collectives_custom_comm.py /usr/local/lib/python3.12/dist-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py
    cp _fsdp_param_group.py /usr/local/lib/python3.12/dist-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py
fi

# torchrun --nproc-per-node=8  \
#     --master_port=6000 \
#     xtuner/v1/train/cli/sft_ib.py \
#     --config ci/scripts/ib_benchmark/customed_comm/ep1_fsdp_acc/exp1_2_ep1_fsdp_our_agrs_bench_acc.py

torchrun --nproc-per-node=8  \
    --master_addr=${MASTER_ADDR} \
    --master_port=6000 \
    --nnodes=${NODE_COUNT} \
    --node_rank=${NODE_RANK} \
    xtuner/v1/train/cli/sft_ib.py \
    --config ci/scripts/ib_benchmark/customed_comm/ep1_fsdp_acc/exp1_2_ep1_fsdp_our_agrs_bench_acc.py