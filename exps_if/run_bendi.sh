#!/usr/bin/env bash
set -euo pipefail

#########################################
# 0. Conda
#########################################
# export PATH="/root/miniconda3/bin:$PATH"
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate verl-rllm-latest

#########################################
# 1. Env
#########################################
export NCCL_DEBUG=INFO

export WANDB_BASE_URL="http://wandb.wml.weibo.com"
export WANDB_API_KEY="local-a5e8cb31021c87c9aa5ed4e7d3cbb02665145208"

# 容器里不要强绑网卡
unset TP_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME

#########################################
# 2. Paths & Exp
#########################################
#########################################
# 2. Paths & Exp
#########################################
project_name="exps-instruct-3b-test"
exp_name="LR_ifbench_16k_tem1-test-merge-jiu"

WORKING_DIR=${WORKING_DIR:-"/njfs/train-aitech/projects/yingwei5/rllm-main-ifbench"}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/njfs/train-aitech/projects/yingwei5/rllm-main-ifbench/exps_if"}
MODEL_PATH=${MODEL_PATH:-"/njfs/train-aitech/projects/yingwei5/rllm-main/exps_instruct_3B/1_3_6_merged"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/train/merged_if_processed.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/test/ifeval.parquet"}


mkdir -p "${CKPTS_DIR}"
#########################################
# 3. PPO Hyperparams
#########################################
clip_ratio_low=0.2
clip_ratio_high=0.2
use_kl_in_reward=False

#########################################
# 4. Run (NO RAY)
#########################################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=16384 \
    reward_model.reward_manager=rllm \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.00 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.val_before_train=True \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.resume_from_path="${CKPTS_DIR}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=20 \
    "$@" 2>&1 | tee "${CKPTS_DIR}/train.log"


