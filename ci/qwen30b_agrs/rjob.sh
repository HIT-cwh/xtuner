new_file=/mnt/shared-storage-user/suzhongling/xtuner_11_27/ci/qwen30b_agrs/rl_qwen30b_dapo_grouped_router.sh
cd /mnt/shared-storage-user/suzhongling/xtuner_11_27
chmod +x $new_file
set -ex
REPLICAS=16
name=n$REPLICAS-q30b-agrs-0
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
    --negative-tags node/gpu-lg-cmc-h-h200-0987.host.h.pjlab.org.cn \
    -- bash -ecx $new_file

