set -ex
REPLICAS=32
# name=multinode$REPLICAS-dsv3-sft-mb1-32k-sp4-fp8
name=n$REPLICAS-dsv3-30l-ep8-g-router-32k-2-shared-szl-69
cd /mnt/shared-storage-user/suzhongling/xtuner_11_27/
rjob submit -e DISTRIBUTED_JOB=true \
    --image=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt26_20251113_7302ba5_grouped_router_topk1 \
    --host-network=true --name $name -P $REPLICAS --gpu 8 --cpu 120  --memory 1500000 --charged-group bw_gpu \
    --private-machine='group' --namespace ailab-pj \
    --gang-start=true \
    --mount=gpfs://gpfs1/caoweihan:/mnt/shared-storage-user/caoweihan \
    --mount=gpfs://gpfs1/llmrazor-share:/mnt/shared-storage-user/llmrazor-share \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --mount=gpfs://gpfs1/suzhongling:/mnt/shared-storage-user/suzhongling \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    -e SDC_TEST=true \
    --negative-tags node/gpu-lg-cmc-h-h200-0011.host.h.pjlab.org.cn \
    -- bash -ecx /mnt/shared-storage-user/suzhongling/xtuner_11_27/ci/dsv3_30l_ep8_customed/dsv3_30l_ep8_32k_2_grouped_router_256gpu.bash
