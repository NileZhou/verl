# 主机IP：TORCH_MASTER_ADDR
# 机器机器的id：TORCH_NODE_RANK （主机为0，别的机器依次为1，2，3）
set -x

# 设置公共环境变量
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export GLOO_SOCKET_IFNAME=bond0
export WANDB_BASE_URL=http://wandb.wml.weibo.com
export WANDB_API_KEY=local-7a836858e9c4c071a448f912d951314c20fe713f
export WANDB_DIR=/sina-ml-runtime
export VLLM_ATTENTION_BACKEND=XFORMERS
# 根据节点编号执行不同操作
if [ "$TORCH_NODE_RANK " -eq 0 ]; then
    # 主节点(node=0)操作
    echo "在主节点(Node 0)上启动Ray集群..."
    ray stop
    ray start --head --node-ip-address=$TORCH_MASTER_ADDR --port=6377 --dashboard-host=0.0.0.0 --dashboard-port=8265

    sleep 25
    # 启动训练脚本
    # data_dir=/njfs/train-nlp/huzheng/train_rlhf_on_cluster/data/open_reason_data
    # lx的数据有点问题，修复一下
    # data_dir=/njfs/train-nlp/lixue18/instruct_tuning_explore/repair_r1/for_code_study
    # 修复后的lx数据
    data_dir=/njfs/train-nlp/huzheng/train_rlhf_on_cluster/data/siyao_comment_data
    model_path=/njfs/train-nlp/huzheng/train_rlhf_on_cluster/train_models/siyao_train_qwen32/siyao_qwen32_model001
    cur_task=test_ds32b-siyao_commentdata-zjreward-32b
    save_model_checkpoint=/njfs/train-nlp/huzheng/train_rlhf_on_cluster/train_models/$cur_task

    echo "在主节点上启动训练..."
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$data_dir/train.parquet \
        data.val_files=$data_dir/test.parquet \
        data.train_batch_size=128 \
        data.max_prompt_length=1024 \
        data.max_response_length=4096 \
        actor_rollout_ref.model.path=$model_path \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        reward_model.reward_api=http://10.136.0.65:6009/get_reward3 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='ds_cloud' \
        trainer.experiment_name=$cur_task \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=200 \
        trainer.default_local_dir=$save_model_checkpoint \
        trainer.test_freq=20 \
        trainer.total_epochs=1 $@
    # 关闭node0上的ray
    ray stop

else
    # 其他节点操作
    echo "在从节点(Node $TORCH_NODE_RANK)上尝试连接到Ray集群..."
    # 使用循环尝试连接到主节点，直到成功
    max_attempts=30
    attempt=1
    connected=false

    # 确保本地没有运行的Ray进程
    ray stop

    while [ $attempt -le $max_attempts ] && [ "$connected" = false ]; do
        echo "尝试连接 ${TORCH_MASTER_ADDR}:6377 (尝试 $attempt/$max_attempts)..."

        # 尝试获取Ray集群状态
        if timeout 15 ray status --address="${TORCH_MASTER_ADDR}:6377" >/dev/null 2>&1; then
            echo "Ray集群状态检查成功，尝试连接..."

            # 尝试连接到Ray集群
            if ray start --address="${TORCH_MASTER_ADDR}:6377"; then
                echo "成功连接到Ray集群！"

                connected=true
            else
                echo "连接失败，稍后重试..."
            fi
        else
            echo "无法获取Ray集群状态，可能集群尚未就绪，等待20秒后再试..."
        fi

        if [ "$connected" = false ]; then
            sleep 20
            attempt=$((attempt + 1))
        fi
    done

    if [ "$connected" = false ]; then
        echo "无法连接到Ray集群，达到最大尝试次数($max_attempts)。退出。"
        exit 1
    fi

    # 从节点不需要启动训练脚本，只需等待任务分配
    echo "从节点(Node $TORCH_NODE_RANK)已连接到Ray集群，等待任务分配..."

    # 保持脚本运行状态并定期监控集群连接
    while true; do
        # 检查Ray集群状态，如果无法连接到主节点的Ray集群，则退出循环
        if ! timeout 10 ray status --address="${TORCH_MASTER_ADDR}:6377" >/dev/null 2>&1; then
            echo "无法连接到主节点Ray集群，可能训练已完成或集群已关闭，退出程序..."
            ray stop # 停止本地Ray进程
            break
        fi

        # 每分钟检查一次连接状态
        sleep 60
    done
fi