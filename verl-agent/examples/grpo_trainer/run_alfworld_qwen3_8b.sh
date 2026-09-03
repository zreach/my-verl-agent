set -x
ENGINE=${1:-vllm}

# ---------------------------------------------------------------------------
# Qwen3-8B GRPO training on ALFWorld with entropy + KL-to-reference monitoring.
#
# Monitoring (核心诉求):
#   * actor/entropy_loss        -> policy entropy each step (自动记录)
#   * actor/kl_loss + kl_coef   -> KL to the initial/reference policy (use_kl_loss=True)
#   * actor/reward_kl_penalty   -> KL-in-reward 视图 (use_kl_in_reward=True), 与 loss 侧 KL 交叉验证
#   These stream to BOTH the console and TensorBoard (see TENSORBOARD_DIR below), so
#   they become trackable curves over training steps.
#
# Model note: 本地 HF 无 Qwen3.5-8B; Qwen3.5-9B 是 qwen3_5 混合线性注意力+VL 架构,
#   当前 vllm 0.11.0 不支持。故使用已验证可跑的 Qwen3-8B (dense, Qwen3ForCausalLM)。
#
# Scale note: 默认小规模冒烟 (train_data_size=16 -> 1 step)。放大训练时提高
#   train_data_size / total_epochs, 并相应调整 ppo_mini_batch_size。
# ---------------------------------------------------------------------------

# --- environment (matches the verified verl-agent ALFWorld recipe) ---
# Activate the dedicated conda env if not already active (non-interactive shells).
if [ -z "$VIRTUAL_ENV" ] && ! command -v python3 | grep -q "envs/alfworld-va"; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate /volume/posttrain/users/zhouyz/envs/alfworld-va 2>/dev/null || true
fi

export HF_HOME=/volume/posttrain/users/zhouyz/hf
export ALFWORLD_DATA=${ALFWORLD_DATA:-/root/.cache/alfworld}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# TensorBoard event files land here; view with `tensorboard --logdir $TENSORBOARD_DIR`.
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-/volume/posttrain/users/zhouyz/verl_agent_tb/grpo_qwen3_8b_alfworld}
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
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='grpo_qwen3_8b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True $@
