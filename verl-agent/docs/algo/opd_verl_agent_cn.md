# verl-agent 当前 OPD 实现说明

## 1. OPD 是什么

OPD 全称是 **On-Policy Distillation**，即在线策略蒸馏。

普通蒸馏通常是 teacher 生成数据，然后 student 学 teacher 的输出。但这样会有一个问题：训练时 student 看到的是 teacher 走过的状态，真正推理时 student 看到的是自己走出来的状态，两者可能不一致。

OPD 的做法是：

```text
student 先自己 rollout
teacher 再对 student 走出来的轨迹打分
student 根据 teacher 的 token-level logprob 更新
```

也就是说，teacher 不负责生成轨迹，而是负责评价 student 当前真实访问到的状态和动作。

在 agent 任务里，这很自然：

```text
student 与环境交互
生成 action
环境返回 observation/reward
teacher 对 student 生成的 action token 计算 logprob
actor 更新时同时使用 RL reward 和 teacher 蒸馏信号
```

## 2. 当前 verl-agent 里的实现方式

当前实现是一个轻量版 OPD。

它没有完整搬新版 `verl` 的 teacher loop、top-k forward KL、多 teacher 体系，而是复用了 `verl-agent` 原有的 reference policy logprob 计算路径。

核心思想是：

```text
把 teacher 当成一个 frozen ref worker
只让它计算 teacher_log_probs
然后在 actor update_policy 中加入 distillation loss
```

也就是说，我们没有新写一套 teacher worker，而是复用了：

```python
ActorRolloutRefWorker
```

让它以：

```python
role="ref"
```

的方式启动。

## 3. 整体训练流程

当前 OPD 训练的一步大致如下：

```text
1. student 在 ALFWorld 环境中 rollout
2. 收集 student 生成的 responses/actions
3. actor 重新计算 old_log_probs
4. teacher worker 对同一批 responses 计算 teacher_log_probs
5. 环境 reward 计算 GRPO advantage
6. actor update 时计算原始 GRPO/PPO policy loss 和 OPD distillation loss
7. 最终 loss = RL loss + distillation_coef * OPD loss
```

简化成数据流：

```text
student rollout
    ↓
batch: prompts, responses, input_ids, attention_mask
    ↓
student compute_log_prob -> old_log_probs
    ↓
teacher compute_ref_log_prob -> teacher_log_probs
    ↓
compute_reward / compute_advantage
    ↓
actor update_policy
    ↓
GRPO loss + OPD loss
```

## 4. 新增配置

当前配置在：

```text
sequence/verl-agent/verl/trainer/config/ppo_trainer.yaml
```

新增了：

```yaml
distillation:
  enabled: False
  teacher_model_path: null
  loss_mode: k3
  use_policy_gradient: True
  use_task_rewards: True
  distillation_loss_coef: 1.0
  loss_max_clamp: 10.0
  clip_ratio: ${actor_rollout_ref.actor.clip_ratio}
  clip_ratio_low: ${actor_rollout_ref.actor.clip_ratio_low}
  clip_ratio_high: ${actor_rollout_ref.actor.clip_ratio_high}
```

各字段含义：

| 字段 | 含义 |
|---|---|
| `enabled` | 是否开启 OPD |
| `teacher_model_path` | teacher 模型路径，例如 8B ckpt |
| `loss_mode` | KL estimator，当前默认 `k3` |
| `use_policy_gradient` | 是否把 OPD 信号作为 policy-gradient reward |
| `use_task_rewards` | 是否保留原始环境 reward |
| `distillation_loss_coef` | OPD loss 权重 |
| `loss_max_clamp` | token-level distillation loss 截断 |
| `clip_ratio*` | OPD policy-gradient 更新使用的 PPO clip 参数 |

## 5. 当前启动脚本

当前 OPD 脚本是：

```text
sequence/verl-agent/examples/grpo_trainer/run_alfworld_1_opd.sh
```

里面默认：

```bash
STUDENT_CKPT=${STUDENT_CKPT:-/workspace/hf/Qwen3-1.7B}
TEACHER_CKPT=${TEACHER_CKPT:-/workspace/hf/Qwen3-8B}
PROJECT_NAME=${PROJECT_NAME:-verl_agent_alfworld}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_qwen3_1.7b_opd_from_8b}
```

也就是说：

```text
student = Qwen3-1.7B
teacher = Qwen3-8B
```

OPD 相关参数：

```bash
distillation.enabled=True
distillation.teacher_model_path=$TEACHER_CKPT
distillation.loss_mode=k3
distillation.use_policy_gradient=True
distillation.use_task_rewards=True
distillation.distillation_loss_coef=1.0
distillation.loss_max_clamp=10.0
```

默认 TensorBoard 目录：

```bash
tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}
```

也就是：

```text
tensorboard_log/verl_agent_alfworld/grpo_qwen3_1.7b_opd_from_8b
```

## 6. main_ppo.py 中做了什么

文件：

```text
sequence/verl-agent/verl/trainer/main_ppo.py
```

新增逻辑：

```python
if config.get("distillation", {}).get("enabled", False):
    if not config.distillation.get("teacher_model_path", None):
        raise ValueError("distillation.teacher_model_path must be set when distillation.enabled=True")
    role_worker_mapping[Role.TeacherPolicy] = ray.remote(ActorRolloutRefWorker)
    mapping[Role.TeacherPolicy] = global_pool_id
```

作用：

如果开启 OPD，就额外注册一个 teacher worker。

这个 worker 仍然使用：

```python
ActorRolloutRefWorker
```

但后面会以：

```python
role="ref"
```

启动。

这样做的好处是：

```text
不用写新的 teacher worker
复用 ref policy 的 logprob 计算逻辑
改动量最小
```

## 7. ray_trainer.py 中新增的 TeacherPolicy

文件：

```text
sequence/verl-agent/verl/trainer/ppo/ray_trainer.py
```

新增了一个角色：

```python
class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    TeacherPolicy = 7
```

初始化时记录：

```python
self.use_teacher_policy = Role.TeacherPolicy in role_worker_mapping
```

也就是说，如果 `main_ppo.py` 注册了 `TeacherPolicy`，trainer 就知道当前要跑 OPD。

## 8. teacher worker 如何加载 8B

在 `init_workers()` 中，新增了：

```python
if self.use_teacher_policy:
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherPolicy)
    teacher_config = deepcopy(self.config.actor_rollout_ref)
    OmegaConf.set_struct(teacher_config, False)
    teacher_config.model.path = self.config.distillation.teacher_model_path
    teacher_policy_cls = RayClassWithInitArgs(
        self.role_worker_mapping[Role.TeacherPolicy],
        config=teacher_config,
        role="ref",
    )
    self.resource_pool_to_cls[resource_pool]["teacher"] = teacher_policy_cls
```

这里关键点是：

```python
teacher_config = deepcopy(self.config.actor_rollout_ref)
```

先复制 student 的配置。

然后：

```python
teacher_config.model.path = self.config.distillation.teacher_model_path
```

把模型路径替换成 teacher 的路径，也就是 8B。

最后：

```python
role="ref"
```

让它以 frozen reference policy 的方式启动。

因此 teacher：

```text
不训练
不建 optimizer
只计算 logprob
```

## 9. teacher worker 何时初始化

仍然在 `ray_trainer.py`：

```python
if self.use_teacher_policy:
    self.teacher_policy_wg = all_wg["teacher"]
    self.teacher_policy_wg.init_model()
```

这一步会真正加载 teacher 模型。

如果 teacher path 不完整，就可能看到：

```text
Fetching 2 files: 0%
```

这通常说明模型目录缺文件，触发了 HuggingFace 下载。

## 10. actor 如何拿到 distillation 配置

文件：

```text
sequence/verl-agent/verl/workers/fsdp_workers.py
```

构建 actor 时加入：

```python
if self.config.get("distillation", {}).get("enabled", False):
    self.config.actor.distillation = self.config.distillation
```

原因是 `DataParallelPPOActor` 只拿到 actor 配置：

```python
self.config.actor
```

如果不把顶层 `distillation` 塞进去，actor update 时就看不到 OPD 配置。

## 11. 训练循环中如何计算 teacher_log_probs

文件：

```text
sequence/verl-agent/verl/trainer/ppo/ray_trainer.py
```

原本训练中会先计算 student 的 old logprob：

```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)
```

得到：

```text
old_log_probs
```

现在 OPD 开启时，额外执行：

```python
if self.use_teacher_policy:
    with _timer("teacher", timing_raw):
        teacher_log_prob = self.teacher_policy_wg.compute_ref_log_prob(batch)
        teacher_log_prob.batch["teacher_log_probs"] = teacher_log_prob.batch.pop("ref_log_prob")
        batch = batch.union(teacher_log_prob)
```

这里 teacher worker 调用的是已有的：

```python
compute_ref_log_prob(batch)
```

它原本返回：

```text
ref_log_prob
```

我们把它改名成：

```text
teacher_log_probs
```

避免和普通 KL reference policy 的 `ref_log_prob` 混淆。

最终 batch 里会有：

```text
old_log_probs
teacher_log_probs
advantages
responses
attention_mask
loss_mask
```

## 12. actor update 中如何加入 OPD

文件：

```text
sequence/verl-agent/verl/workers/actor/dp_actor.py
```

### 12.1 读取 teacher_log_probs

原来 actor update 只选：

```python
select_keys = [
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "old_log_probs",
    "advantages",
]
```

现在如果 OPD 开启：

```python
distillation_config = self.config.get("distillation", {})
distillation_enabled = distillation_config.get("enabled", False)

if distillation_enabled:
    select_keys.append("teacher_log_probs")
```

这样每个 micro batch 里都会有 teacher logprob。

## 13. 原始 GRPO/PPO loss 不变

actor update 中原本的 policy loss 仍然照常计算：

```python
entropy, log_prob = self._forward_micro_batch(
    micro_batch=data,
    temperature=temperature,
    calculate_entropy=calculate_entropy,
)

pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=response_mask,
    cliprange=clip_ratio,
    cliprange_low=clip_ratio_low,
    cliprange_high=clip_ratio_high,
    clip_ratio_c=clip_ratio_c,
    loss_agg_mode=loss_agg_mode,
)
```

这里：

```text
old_log_prob = 更新前 student logprob
log_prob = 当前 forward 得到的 student logprob
advantages = GRPO 根据环境 reward 算出来的 advantage
```

这部分就是原来的 RL loss。

## 14. OPD token-level loss 如何算

新增代码：

```python
teacher_log_probs = data["teacher_log_probs"]
distill_loss_mode = distillation_config.get("loss_mode", "k3")

distillation_losses = kl_penalty(
    logprob=log_prob,
    ref_logprob=teacher_log_probs,
    kl_penalty=distill_loss_mode,
)
```

也就是比较：

```text
student 当前 log_prob
teacher log_prob
```

得到每个 response token 上的 distillation loss。

当前默认：

```bash
distillation.loss_mode=k3
```

`k3` 是一种 reverse-KL estimator，通常比 `k1` 更稳定一些。

## 15. loss clamp

为了防止极端 token 造成训练不稳定，加入了：

```python
loss_max_clamp = distillation_config.get("loss_max_clamp", None)

if loss_max_clamp is not None:
    distillation_losses = distillation_losses.clamp(
        min=-loss_max_clamp,
        max=loss_max_clamp,
    )
```

当前脚本里是：

```bash
distillation.loss_max_clamp=10.0
```

## 16. PG OPD 模式

当前使用：

```bash
distillation.use_policy_gradient=True
```

这种模式下，不直接对 `distillation_losses` 做 supervised backprop，而是把它变成 policy-gradient advantage：

```python
advantages=-distillation_losses.detach()
```

代码：

```python
distill_pg_loss, distill_pg_clipfrac, distill_ppo_kl, distill_pg_clipfrac_lower = policy_loss_fn(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=-distillation_losses.detach(),
    response_mask=response_mask,
    cliprange=distillation_config.get("clip_ratio", clip_ratio),
    cliprange_low=distillation_config.get("clip_ratio_low", clip_ratio_low),
    cliprange_high=distillation_config.get("clip_ratio_high", clip_ratio_high),
    clip_ratio_c=clip_ratio_c,
    loss_agg_mode=loss_agg_mode,
)
```

直观解释：

```text
如果 teacher 认为 student 当前 token 好：
    distillation loss 小
    -distillation loss 相对大
    这个 token 被增强

如果 teacher 认为 student 当前 token 差：
    distillation loss 大
    -distillation loss 相对小
    这个 token 被压低
```

`.detach()` 很关键：

```python
-distillation_losses.detach()
```

它表示 teacher/student 的 logprob 差值只作为 reward，不让梯度穿过 reward 本身。

## 17. supervised OPD 模式

如果设：

```bash
distillation.use_policy_gradient=False
```

则走：

```python
distill_loss = agg_loss(
    loss_mat=distillation_losses,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
)
```

然后直接加入总 loss。

不过当前轻量实现只有 sampled-token teacher logprob，没有 teacher top-k 分布，因此更推荐 PG OPD。

## 18. 是否保留环境 reward

当前脚本：

```bash
distillation.use_task_rewards=True
```

所以最终 loss 是：

```text
GRPO/PPO loss + distillation_loss_coef * OPD loss
```

如果改成：

```bash
distillation.use_task_rewards=False
```

则会执行：

```python
policy_loss = torch.zeros_like(policy_loss)
```

最终只保留 OPD loss。

## 19. 最终 loss

最终合并：

```python
distill_coef = distillation_config.get("distillation_loss_coef", 1.0)
policy_loss = policy_loss + distill_coef * distill_loss
```

也就是：

```text
L = L_RL + lambda * L_OPD
```

当前：

```bash
lambda = 1.0
```

## 20. OPD 相关日志

新增的 TensorBoard 指标：

```text
actor/distillation_loss
actor/distillation_abs_loss
actor/distillation_loss_coef
actor/distillation_pg_clipfrac
actor/distillation_ppo_kl
actor/distillation_pg_clipfrac_lower
```

这些可以用来判断 OPD 是否有效。

目前实验中看到：

```text
actor/distillation_loss: 0.433 -> 0.275
```

说明 student 确实在向 teacher 靠近。

## 21. 当前实现和标准 verl OPD 的区别

标准 `verl` 里 OPD 是完整实现，配置类似：

```bash
distillation.enabled=True
distillation.n_gpus_per_node=...
distillation.nnodes=...
distillation.teacher_models.teacher_model.model_path=...
distillation.teacher_models.teacher_model.inference.name=vllm
distillation.distillation_loss.loss_mode=...
distillation.distillation_loss.use_policy_gradient=...
```

而我们当前 `verl-agent` 是轻量实现：

```bash
distillation.enabled=True
distillation.teacher_model_path=...
distillation.loss_mode=k3
distillation.use_policy_gradient=True
```

对比：

| 项目 | 标准 verl | 当前 verl-agent |
|---|---|---|
| teacher worker | 独立 teacher pool + inference server | 复用 ref worker |
| teacher 配置 | `teacher_models.teacher_model.model_path` | `teacher_model_path` |
| loss 配置 | `distillation_loss.*` | 扁平 `distillation.*` |
| top-k forward KL | 支持 | 不支持 |
| sampled-token PG OPD | 支持 | 支持 |
| 多 teacher | 支持 | 不支持 |
| 异步 teacher 计算 | 支持 | 不支持 |
| 改动量 | 大 | 小 |

## 22. 当前实验结果解读

当前 `/Users/bytedance/Desktop/code/va-tensorboard` 中有三个 run：

```text
1.7B
8B
opd
```

共同 step 附近对比：

| run | val/success_rate | val/test_score | episode/success_rate | episode/reward/mean |
|---|---:|---:|---:|---:|
| 1.7B | 0.203 | 0.627 | 0.109 | 1.094 |
| OPD | 0.336 | 1.088 | 0.391 | 3.906 |
| 8B | 0.547 | 2.522 | 0.641 | 6.406 |

说明：

```text
OPD 明显优于 1.7B baseline
但还没有达到 8B teacher 水平
```

OPD best：

```text
val/success_rate best: 0.398
val/text/test_score best: 1.393
episode/success_rate best: 0.555
episode/reward/mean best: 5.547
```

## 23. 当前实验的风险信号

OPD 的 KL 比较高：

```text
actor/kl_loss:
1.7B final ~= 0.183
8B final ~= 0.199
OPD final ~= 0.713
```

OPD 的 entropy 很低：

```text
actor/entropy_loss:
1.7B final ~= 0.194
8B final ~= 0.225
OPD final ~= 0.032
```

这说明 teacher 信号比较强，student policy 变得很确定。

好处是收敛快，坏处是可能探索不足或过早收敛。

## 24. 调参建议

如果后续 OPD 的 validation 继续涨，可以保持当前配置。

如果出现 val 掉点或 entropy 过低，可以尝试：

```bash
distillation.distillation_loss_coef=0.5
```

或者：

```bash
distillation.distillation_loss_coef=0.3
```

也可以提高 entropy：

```bash
actor_rollout_ref.actor.entropy_coeff=0.002
```

或：

```bash
actor_rollout_ref.actor.entropy_coeff=0.003
```

重点观察：

```text
val/success_rate
val/text/test_score
actor/kl_loss
actor/entropy_loss
actor/distillation_loss
episode/success_rate
```

## 25. 一句话总结

当前 `verl-agent` 的 OPD 实现是：

```text
让 1.7B student 先在 ALFWorld 中 on-policy rollout，
再让 frozen 8B teacher 对 student 生成的 response token 计算 logprob，
然后在 GRPO actor update 中加入基于 teacher_log_probs 的 sampled-token PG OPD loss。
```

它是一个轻量实现，已经能有效提升 1.7B，但 teacher 信号偏强，需要继续关注 KL 和 entropy。
