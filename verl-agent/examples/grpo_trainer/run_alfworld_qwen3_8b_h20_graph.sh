set -x
ENGINE=${1:-vllm}

# ---------------------------------------------------------------------------
# Qwen3-8B GRPO on ALFWorld — H20 variant B (keep CUDA graph).
#
# Same algorithm / data scale / KL monitoring as run_alfworld_qwen3_8b.sh.
# This is variant B: keep CUDA graph ON (enforce_eager=False) for faster
# rollout, and prevent OOM purely by shrinking vllm's memory claim instead of
# releasing the KV cache between phases.
#   * enforce_eager: kept False        (CUDA graph ON; needs free_cache_engine=False)
#   * free_cache_engine: True -> False (vllm KV cache stays resident, no release)
#   * gpu_memory_utilization: 0.45 -> 0.30 (leave room for the actor)
# Variant A (run_alfworld_qwen3_8b_h20.sh) instead keeps CUDA graph off and
# releases the KV cache (free_cache_engine=True) — the safer anti-OOM path.
# The two are mutually exclusive in verl (vllm_rollout_spmd.py asserts
# not (not enforce_eager and free_cache_engine)).
# With free_cache_engine=False, vllm and the FSDP actor hold memory
# SIMULTANEOUSLY, so the binding constraint is total HBM. Per card (~96GB):
# vllm ~29GB (0.30) + actor ~40-55GB (weights+grad+optimizer+activation) —
# headroom is thin; if this OOMs, lower gpu_memory_utilization to 0.25, then
# 0.20, or just switch to variant A. Everything else (adv_estimator, KL loss
# + KL-in-reward monitoring, data size, ppo_mini_batch_size, lr, penalties) is
# identical to the baseline.
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
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-/volume/posttrain/users/zhouyz/verl_agent_tb/grpo_qwen3_8b_alfworld_h20_graph}
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_qwen3_8b_h20_graph' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
