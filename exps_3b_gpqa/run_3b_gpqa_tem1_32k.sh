#!/usr/bin/env bash
set -euxo pipefail


export PATH="/root/miniconda3/bin/:$PATH"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-rllm-latest

export NCCL_DEBUG=INFO        # 设置NCCL日志级别 (可选值: VERSION, WARN, INFO, TRACE)

export WANDB_BASE_URL="http://wandb.wml.weibo.com"
export WANDB_API_KEY="local-a5e8cb31021c87c9aa5ed4e7d3cbb02665145208"

export TP_SOCKET_IFNAME=bond0 #在指定通信时走哪张网卡。
export GLOO_SOCKET_IFNAME=bond0

NODE_RANK=${node_rank:-0} #等价于“当前机器的节点编号，默认当成 0”

HOST_IP=$(python -c "import socket; import fcntl; import struct; print(socket.inet_ntoa(fcntl.ioctl(socket.socket(socket.AF_INET, socket.SOCK_DGRAM), 0x8915, struct.pack('256s', b'bond0'))[20:24]))")
RAY_PORT=6379
DASHBOARD_PORT=8299

# 共享文件方式，传递主节点IP以及设置任务结束标志
SHARED_IP_FILE="./base1.txt"
TASK_COMPLETE_FILE="./ray_task_complete1.txt"

project_name='exps-3b-gpqa'
exp_name='LR_3b_gpqa_32k_tem1'


WORKING_DIR=${WORKING_DIR:-"/njfs/train-aitech/projects/yingwei5/rllm-main-ifbench"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/verl/trainer/runtime_env.yaml"} #这个 job 用什么运行环境来执行。
NNODES=${NNODES:-4}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/njfs/train-aitech/projects/yingwei5/rllm-main-ifbench/exps_3b_gpqa"}
MODEL_PATH=${MODEL_PATH:-"/njfs/train-aitech/projects/yingwei5/rllm-main-ifbench/exps_3b_gpqa/ckpts/exps-3b-gpqa/LR_3b_gpqa_24k_tem1/global_step_1320/hf"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/train/mcqa-fi-13096.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/test/gpqa.parquet"}

clip_ratio_low=0.2
clip_ratio_high=0.2

use_kl_in_reward=False #kl-loss 都没有用

if [ "$NODE_RANK" -eq 0 ]; then
    # 主节点：启动Ray head并提交任务
    echo "Initializing Ray Head Node at $HOST_IP:$RAY_PORT"
    
    echo "$HOST_IP" > $SHARED_IP_FILE
    rm -f $TASK_COMPLETE_FILE
    
    ray start --head --port=$RAY_PORT --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT
    
    ## TODO: 提交任务，多几多卡代码
    ray job submit --runtime-env="${RUNTIME_ENV}" \
            --working-dir "${WORKING_DIR}" \
            --address="http://${HOST_IP}:${DASHBOARD_PORT}"  \
            -- python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            algorithm.use_kl_in_reward=${use_kl_in_reward} \
            algorithm.kl_ctrl.kl_coef=0.001 \
            data.train_files=${TRAIN_FILE} \
            data.val_files=${TEST_FILE} \
            data.train_batch_size=64 \
            data.val_batch_size=512 \
            data.max_prompt_length=2048 \
            data.max_response_length=32768 \
            reward_model.reward_manager=rllm \
            actor_rollout_ref.model.path=${MODEL_PATH} \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
            actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
            actor_rollout_ref.actor.ppo_mini_batch_size=64 \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
            actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.00 \
            actor_rollout_ref.actor.entropy_coeff=0.00 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
            actor_rollout_ref.rollout.n=16 \
            actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
            actor_rollout_ref.rollout.val_kwargs.n=16 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            actor_rollout_ref.ref.fsdp_config.param_offload=False \
            actor_rollout_ref.rollout.calculate_log_probs=True \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name=${project_name} \
            trainer.experiment_name=${exp_name} \
            trainer.val_before_train=True \
            +trainer.wandb_api_key="${WANDB_API_KEY}" \
            trainer.default_local_dir="${CKPTS_DIR}" \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=4 \
            trainer.resume_from_path="${CKPTS_DIR}" \
            trainer.save_freq=10 \
            trainer.test_freq=10 \
            trainer.default_hdfs_dir=null \
            trainer.total_epochs=40 "${@:1}" > ${exp_name}.log 2>&1
    
    # 等待任务完成，不适用于 ray job sumit的 no-wait
    touch $TASK_COMPLETE_FILE
    echo "Master Task Completed, Notify Sub-Nodes"
    
    sleep 30
    ray stop
else

    WAIT_COUNT=0
    while [ ! -f $SHARED_IP_FILE ]; do
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        echo "Waiting for Master Initialize for $WAIT_COUNT Seconds..."
    done

    # 读取主节点IP
    HEAD_IP=$(cat $SHARED_IP_FILE)
    
    echo "=> Connect to MasterIP: $HEAD_IP "
    ray start --address="$HEAD_IP:6379" 
    ray status
    
    # 等待主节点任务完成
    while [ ! -f $TASK_COMPLETE_FILE ]; do
        sleep 10
    done
    
    ray stop

    rm -f $TASK_COMPLETE_FILE
    rm -f $SHARED_IP_FILE
fi

