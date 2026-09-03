set -x
ENGINE=${1:-vllm}

# ---------------------------------------------------------------------------
# Qwen3-8B GRPO on ALFWorld — H20-tuned variant.
#
# Same algorithm / data scale / KL monitoring as run_alfworld_qwen3_8b.sh. The
# knobs are tuned for 8×H20 (96GB HBM/card, weak BF16 compute). vllm (rollout)
# and the FSDP actor SHARE the same 8 GPUs (colocate), so the binding constraint
# is memory, not speed: vllm must not over-claim HBM or the actor can't fit.
# Changes vs the baseline script:
#   * enable_gradient_checkpointing: True -> False   (don't recompute activations)
#   * rollout.tensor_model_parallel_size: 4 -> 1      (8 vllm replicas, fastest rollout)
#   * ref.fsdp_config.param_offload: True -> False    (keep ref weights resident)
#   * *_micro_batch_size_per_gpu: 8 -> 8              (kept; OOM headroom)
#   * rollout.gpu_memory_utilization: 0.6 -> 0.45     (leave room for the actor)
#   * rollout.free_cache_engine: False -> True        (release KV cache before train)
# Everything else (adv_estimator, KL loss + KL-in-reward monitoring, data size,
# ppo_mini_batch_size, lr, penalties) is identical to the baseline.
# Memory budget per card (~96GB): vllm ~43GB (rollout) / actor ~40GB (train),
# freed between phases by free_cache_engine. If you still OOM, lower
# gpu_memory_utilization to 0.4, then micro_batch to 4; if you have headroom,
# raise gpu_memory_utilization back toward 0.5 for faster rollout.
# ---------------------------------------------------------------------------

# --- environment (matches the verified verl-agent ALFWorld recipe) ---
# Activate the dedicated conda env if not already active (non-interactive shells).
if [ -z "$VIRTUAL_ENV" ] && ! command -v python3 | grep -q "envs/alfworld-va"; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate /volume/posttrain/users/zhouyz/envs/alfworld-va 2>/dev/null || true
fi

export HF_HOME=/volume/posttrain/users/zhouyz/hf
export ALFWORLD_DATA=${ALFWORLD_DATA:-/root/.cache/alfworld}
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# TensorBoard event files land here; view with `tensorboard --logdir $TENSORBOARD_DIR`.
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-/volume/posttrain/users/zhouyz/verl_agent_tb/grpo_qwen3_8b_alfworld_h20}
mkdir -p "$TENSORBOARD_DIR"

num_cpus_per_env_worker=0.1 # CPU resource per environment worker. Lower to use fewer CPU resources.

train_data_size=16
val_data_size=128
group_size=8

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/volume/posttrain/users/zhouyz/data/verl-agent/text/train.parquet \
    data.val_files=/volume/posttrain/users/zhouyz/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_qwen3_8b_h20' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
