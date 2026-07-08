set -x
ENGINE=${1:-vllm}
if [ $# -gt 0 ]; then
    shift
fi
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}

train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=${GROUP_SIZE:-8}

STUDENT_CKPT=${STUDENT_CKPT:-/root/hf/Qwen/Qwen2.5-1.5B-Instruct}
TEACHER_CKPT=${TEACHER_CKPT:-/root/hf/langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld}
OPD_METHOD=${OPD_METHOD:-pg}
OPD_TARGET=${OPD_TARGET:-sampled}
OPD_TOPK=${OPD_TOPK:-32}
RLSD_LAMBDA=${RLSD_LAMBDA:-0.5}
RLSD_LAMBDA_DECAY_STEPS=${RLSD_LAMBDA_DECAY_STEPS:-50}
RLSD_CLIP_EPSILON=${RLSD_CLIP_EPSILON:-0.2}
if [ $# -gt 0 ] && [ "$1" = "--topk" ]; then
    if [ $# -lt 2 ]; then
        echo "--topk requires a positive integer value."
        exit 1
    fi
    OPD_TOPK=$2
    shift 2
elif [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    OPD_TOPK=$1
    shift
fi
if ! [[ "$OPD_TOPK" =~ ^[0-9]+$ ]]; then
    echo "OPD_TOPK must be a positive integer, got: $OPD_TOPK"
    exit 1
fi
if [ "$OPD_TOPK" -le 0 ]; then
    echo "OPD_TOPK must be greater than 0, got: $OPD_TOPK"
    exit 1
fi
OPD_LOSS_MODE=${OPD_LOSS_MODE:-}
if [ -z "$OPD_LOSS_MODE" ]; then
    if [ "$OPD_METHOD" = "gkd" ] && [ "$OPD_TARGET" = "topk" ]; then
        OPD_LOSS_MODE=forward_kl_topk
    else
        OPD_LOSS_MODE=k3
    fi
fi
if { [ "$OPD_METHOD" = "pg" ] || [ "$OPD_METHOD" = "rlsd" ]; } && [ "$OPD_TARGET" != "sampled" ]; then
    echo "$OPD_METHOD uses sampled-token teacher logprobs; set OPD_TARGET=sampled or use OPD_METHOD=gkd for full/topk."
    exit 1
fi
if [ "$OPD_METHOD" = "pg" ] || [ "$OPD_METHOD" = "rlsd" ]; then
    OPD_USE_POLICY_GRADIENT=True
else
    OPD_USE_POLICY_GRADIENT=False
fi
PROJECT_NAME=${PROJECT_NAME:-verl_agent_alfworld}
if [ "$OPD_METHOD" = "rlsd" ]; then
    DEFAULT_EXPERIMENT_NAME=grpo_qwen2.5_1.5b_rlsd_l${RLSD_LAMBDA}_c${RLSD_CLIP_EPSILON}_d${RLSD_LAMBDA_DECAY_STEPS}
elif [ "$OPD_TARGET" = "topk" ]; then
    DEFAULT_EXPERIMENT_NAME=grpo_qwen2.5_1.5b_opd_${OPD_METHOD}_${OPD_TARGET}_k${OPD_TOPK}
else
    DEFAULT_EXPERIMENT_NAME=grpo_qwen2.5_1.5b_opd_${OPD_METHOD}_${OPD_TARGET}
fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$STUDENT_CKPT \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    distillation.enabled=True \
    distillation.teacher_model_path=$TEACHER_CKPT \
    distillation.method=$OPD_METHOD \
    distillation.target=$OPD_TARGET \
    distillation.topk=$OPD_TOPK \
    distillation.loss_mode=$OPD_LOSS_MODE \
    distillation.use_policy_gradient=$OPD_USE_POLICY_GRADIENT \
    distillation.rlsd_lambda=$RLSD_LAMBDA \
    distillation.rlsd_lambda_decay_steps=$RLSD_LAMBDA_DECAY_STEPS \
    distillation.rlsd_clip_epsilon=$RLSD_CLIP_EPSILON \
    distillation.use_task_rewards=True \
    distillation.distillation_loss_coef=1.0 \
    distillation.loss_max_clamp=10.0 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True "$@"
