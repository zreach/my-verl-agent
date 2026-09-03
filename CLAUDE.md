# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The repository root is a thin wrapper: the entire project lives in `verl-agent/`.
**Run all commands from `verl-agent/`, not the repo root** (`pip install -e .`, all
`python3 -m ...` invocations, and every script in `examples/` assume that as cwd; scripts
reference `repo_root/` meaning `verl-agent/`).

`verl-agent` is an extension of [veRL](https://github.com/volcengine/verl) for training LLM
agents with RL. Its own code (`agent_system/`, `gigpo/`) is layered onto a full, in-tree copy
of veRL (`verl/`); most files under `verl/` are upstream and should be treated as a vendored
dependency unless a change specifically targets the agent extensions.

## Common commands

```bash
# Install (from verl-agent/, in the verl-agent conda env)
pip install -e .

# Prepare rollout data. For gym-style envs this ONLY encodes modality + dataset size;
# the actual agent input comes from env.step(), not from the parquet rows.
python3 -m examples.data_preprocess.prepare --mode text  --train_data_size 256 --val_data_size 256
python3 -m examples.data_preprocess.prepare --mode visual --train_data_size 256 --val_data_size 256

# Train: pick an algorithm dir under examples/ and an env-specific script
bash examples/gigpo_trainer/run_alfworld.sh          # GiGPO on ALFWorld
bash examples/grpo_trainer/run_webshop.sh            # GRPO on WebShop
bash examples/grpo_trainer/run_alfworld_qwen3_8b.sh  # this branch's Qwen3-8B GRPO run

# Tests (pytest, upstream veRL suite)
pytest tests/                       # full suite
pytest tests/path/to/test_x.py::test_fn   # single test
```

Training scripts are plain bash that call `python3 -m verl.trainer.main_ppo` with a long list
of Hydra `key=value` overrides. To change a run, edit the script's overrides rather than the
YAML defaults.

## Per-environment setup caveats

Each environment needs its own install and often its **own conda env** to avoid version
conflicts (see `verl-agent/README.md` for exact steps):
- **WebShop** requires Python ≤ 3.10 and pins older `torch`/`vllm` — use a separate `verl-agent-webshop` env.
- **Search** needs a separate `retriever` conda env (faiss-gpu) running a local retrieval server
  (`bash examples/search/retriever/retrieval_launch.sh`) before training; it consumes ~6GB GPU per GPU.
- **ALFWorld** requires `alfworld-download -f` (data cached under `~/.cache/alfworld/`, overridable via `ALFWORLD_DATA`).
- **AppWorld** is experimental and runs its server in a separate `appworld` conda env.

## Architecture

The central design idea is **step-independent multi-turn rollout**: instead of concatenating
the full interaction history into the prompt (which blows up context on long-horizon tasks),
each step's LLM input is rebuilt from the current observation plus a short, customizable history
summary. This keeps context roughly constant across 30–50 step episodes.

**Training flow** (entry point `verl/trainer/main_ppo.py` → `verl/trainer/ppo/ray_trainer.py`):
1. `main_ppo.py` calls `agent_system.environments.make_envs(config)` to build train/val
   vectorized environments, and constructs a `TrajectoryCollector` + `EpisodeRewardManager`.
2. `ray_trainer.py` (a fork of veRL's PPO trainer) drives the loop. For agent runs it calls
   `traj_collector.multi_turn_loop(...)` to roll out full episodes, then computes advantages
   via `compute_advantage(...)`.
3. Rewards come from the environment (`env.step()`), not a reward model —
   `EpisodeRewardManager` (`agent_system/reward_manager/episode.py`) places per-episode rewards
   onto the response tokens.

**Key modules under `agent_system/`:**
- `multi_turn_rollout/rollout_loop.py` — `TrajectoryCollector`. `multi_turn_loop()` dispatches
  to `vanilla_multi_turn_loop` or `dynamic_multi_turn_loop` (dynamic sampling / DAPO). Handles
  per-step tokenization (`preprocess_single_sample`/`preprocess_batch`), the step→env→step loop,
  and gathering trajectories into a `DataProto` (`gather_rollout_data`).
- `environments/base.py` — `EnvironmentManagerBase`. Subclass contract: `reset()`/`step()`
  return an obs dict `{'text', 'image', 'anchor'}` (`anchor` is the history-free observation used
  only by GiGPO); `build_text_obs()` assembles the per-step prompt; `success_evaluator()` scores
  episodes (default reads `info['won']` of the last active step).
- `environments/env_manager.py` — `make_envs()` is the environment registry; it branches on
  `config.env.env_name` and wires each env package to its projection function and manager
  subclass. Each env has an `EnvironmentManager` subclass here (e.g. `AlfWorldEnvironmentManager`).
- `environments/env_package/<env>/` — gym-style, multi-process env implementations plus a
  `projection` function that maps generated text actions → environment actions (and reports
  action validity).
- `environments/prompts/<env>.py` — per-env prompt templates (minimal `<think>...</think>` +
  `<action>...</action>` format).
- `memory/memory.py` — `SimpleMemory`, the default history manager, invoked from
  `env_manager.build_text_obs()`. Extend this for custom history/summarization strategies.

**GiGPO** (`gigpo/core_gigpo.py`) is the project's own algorithm: a critic-free, two-level
grouping advantage estimator (episode-level like GRPO + step-level groups over repeated
"anchor" states). It is invoked from `ray_trainer.compute_advantage()` when
`algorithm.adv_estimator=gigpo`.

## Configuration

`verl/trainer/config/ppo_trainer.yaml` is the master Hydra config. Agent-specific sections
added by this project:
- `env:` — `env_name`, `max_steps`, `history_length`, `rollout.n` (group size for GRPO/GiGPO —
  all envs in a group share the same `reset()` initial state), and per-env sub-blocks
  (`alfworld`, `search`, `sokoban`, `webshop`).
- `algorithm.gigpo:` — `step_advantage_w`, `mode` (`mean_norm`/`mean_std_norm`), similarity-based
  anchor grouping.
- `actor_rollout_ref.actor.use_invalid_action_penalty` / `invalid_action_penalty_coef` — penalize
  actions the projection function marks invalid.
- `algorithm.filter_groups` — DAPO-style dynamic sampling.

Supported `adv_estimator` values: `gigpo`, `grpo`, `gspo`, `ppo`, `rloo`, `reinforce_plus_plus`,
etc. `examples/` has one directory per algorithm; each contains per-environment run scripts.

## Adding a new environment

1. Create a gym-style, multi-process env package under `agent_system/environments/env_package/<env>/`
   with a `build_<env>_envs` factory and a `<env>_projection` function.
2. Add prompt templates in `agent_system/environments/prompts/<env>.py`.
3. Add an `EnvironmentManager` subclass and a branch in `make_envs()` in
   `agent_system/environments/env_manager.py`.

Use the WebShop implementation as the reference pattern.
