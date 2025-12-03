set -ex
cd /mnt/shared-storage-user/caoweihan/projects/xtuner_251020

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

export NCCL_MAX_CTAS=20 # We need to control the max SM used by nccl
export DISTRIBUTED_COMMUNICATION_SM=24
export PYTHONPATH=$PYTHONPATH:/mnt/shared-storage-user/llmrazor-share/data/suzhongling/environment/ib_wrapper/local/lib/python3.12/dist-packages/ib_wrapper-2.0.0-py3.12-linux-x86_64.egg/
export PYTHONPATH=$PYTHONPATH:/mnt/shared-storage-user/suzhongling/AdaptiveGEMM/
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD


torchrun --nproc-per-node=8  \
    --master_addr=${MASTER_ADDR} \
    --master_port=6000 \
    --nnodes=${NODE_COUNT} \
    --node_rank=${NODE_RANK} \
    xtuner/v1/train/cli/sft.py \
    --config ci/scripts/ib_benchmark/exp2_2_ep8_grouped_router_nccl_agrs_bench_acc.py