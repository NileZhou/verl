# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**verl** (Volcano Engine Reinforcement Learning) is a distributed RL framework for training LLMs. Version 0.8.0 supports PPO, GRPO, DAPO, and various algorithm variants via a **recipe** system.

Key design principle: core framework in `verl/`, algorithm variants in `recipe/`. Recipes extend the framework without modifying core code (with minimal exceptions for hooks).

## Repository Structure

```
verl/                           # Core framework
├── trainer/
│   ├── ppo/
│   │   ├── core_algos.py       # Policy loss registry, advantage estimators, agg_loss()
│   │   ├── ray_trainer.py      # RayPPOTrainer — base training loop
│   │   ├── reward.py           # load_reward_manager()
│   │   └── metric_utils.py
│   ├── config/
│   │   ├── ppo_trainer.yaml    # Base training config (all recipes inherit this)
│   │   └── legacy_reward_impl.yaml
│   └── main_ppo.py             # run_ppo(), TaskRunner base class
├── workers/
│   ├── config/                 # Strict dataclass configs
│   │   ├── actor.py            # ActorConfig, FSDPActorConfig, PolicyLossConfig
│   │   ├── engine.py           # FSDPEngineConfig (dtype lives HERE)
│   │   ├── rollout.py          # RolloutConfig, SamplingConfig
│   │   ├── model.py            # HFModelConfig
│   │   └── reward.py           # RewardConfig
│   ├── utils/losses.py         # ppo_loss() — loss entry point
│   ├── engine_workers.py       # Engine worker (init_model, train_batch)
│   └── fsdp_workers.py         # FSDP distributed worker
├── experimental/
│   ├── reward_loop/            # Async reward computation
│   │   ├── reward_loop.py      # RewardLoopWorker
│   │   └── reward_manager/     # dapo.py, naive.py, registry.py, ...
│   └── agent_loop/             # Rollout + reward orchestration
├── utils/
│   ├── config.py               # omega_conf_to_dataclass(), validate_config()
│   ├── dataset/rl_dataset.py   # RLHFDataset
│   └── torch_functional.py     # masked_mean, etc.
├── base_config.py              # BaseConfig (frozen dataclass + dict-like interface)
└── single_controller/          # Ray-based distributed controller

recipe/                         # Algorithm variants (30+ recipes)
├── dapo/                       # DAPO — filter_groups reference implementation
├── archer/                     # Archer — ASPO + overlong filter + custom reward
├── gvpo/, prime/, spo/ ...     # Other algorithm variants
└── ...

rllm/                           # Custom reward functions
└── rewards/
    ├── rl_reward.py            # rllm_reward_fn() — unified entry point
    ├── code_reward.py          # Code evaluation (LiveCodeBench, etc.)
    └── math_reward.py          # Math evaluation
```

## Config System (CRITICAL)

verl 0.8.0 uses **strict dataclass validation** via Hydra. Config values go through `omega_conf_to_dataclass()` → `hydra.utils.instantiate()`, which calls the dataclass `__init__`. **Unknown fields cause `TypeError`**.

### Three-layer config hierarchy
```
ppo_trainer.yaml (base)
  → recipe/xxx/config/xxx_trainer.yaml (recipe override via Hydra defaults)
    → CLI overrides in run scripts
```

### Key config paths and their dataclasses
```
actor_rollout_ref.actor.*         → ActorConfig / FSDPActorConfig
actor_rollout_ref.actor.policy_loss.* → PolicyLossConfig
actor_rollout_ref.actor.fsdp_config.* → FSDPEngineConfig  ← dtype is HERE
actor_rollout_ref.actor.optim.*   → OptimizerConfig
actor_rollout_ref.ref.*           → FSDPActorConfig (same class as actor)
actor_rollout_ref.rollout.*       → RolloutConfig
actor_rollout_ref.rollout.val_kwargs.* → SamplingConfig
actor_rollout_ref.model.*         → HFModelConfig
algorithm.*                       → AlgoConfig (via yaml, less strict)
reward.*                          → reward yaml (open dict for reward_kwargs)
trainer.*                         → ppo_trainer.yaml trainer section (open)
data.*                            → data yaml (open)
```

### Adding new config fields
If a field goes through dataclass validation (actor, ref, rollout, model), it MUST be declared in the corresponding dataclass. Use `grep -A 30 "class XxxConfig" verl/workers/config/xxx.py` to check existing fields.

Fields under `trainer.*`, `data.*`, `reward.reward_kwargs.*` are accessed via `.get()` and don't need dataclass declaration.

## Policy Loss Registry

```python
# verl/trainer/ppo/core_algos.py
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask,
                                 loss_agg_mode, config, rollout_is_weights):
    ...
    return pg_loss, metrics_dict

# Dispatch: config.actor_rollout_ref.actor.policy_loss.loss_mode → registry lookup
```

To add a new loss: create a function with `@register_policy_loss("name")`, import the module before training starts (typically in recipe's `__init__.py` or trainer import).

`agg_loss()` accepts `**kwargs` to absorb extra keys from `config.global_batch_info`.

## Reward System

### DapoRewardManager call signature
```python
result = compute_score(data_source=..., solution_str=...,  # NOT llm_solution
                       ground_truth=..., extra_info=...)
```

### Expected return values
- Scalar: `return 1.0`
- Dict: `return {"score": 1.0, ...}`  ← MUST have "score" key

### overlong_buffer_cfg constraint
If `overlong_buffer_cfg` is provided (even with `enable=False`), `max_resp_len` MUST also be set:
```yaml
reward.reward_kwargs.max_resp_len=16384
reward.reward_kwargs.overlong_buffer_cfg.enable=False
```

## Recipe Pattern

### Creating a new recipe
```
recipe/myalgo/
├── __init__.py
├── myalgo_core_algos.py       # Register custom loss: @register_policy_loss("myalgo")
├── myalgo_ray_trainer.py      # class RayMyAlgoTrainer(RayDAPOTrainer)
├── main_myalgo.py             # TaskRunner subclass + @hydra.main
├── config/
│   └── myalgo_trainer.yaml    # defaults: [ppo_trainer, _self_]
├── runtime_env.yaml           # Ray runtime env (working_dir, excludes, env_vars)
└── run_myalgo.sh              # Training script
```

### Trainer inheritance chain
```
RayPPOTrainer → RayDAPOTrainer → Your recipe trainer
```
- `RayPPOTrainer`: core loop (rollout → reward → advantage → update)
- `RayDAPOTrainer`: adds filter_groups, entropy/KL metrics
- Your trainer: custom modifications (e.g., overlong filter)

### Entry point pattern
```python
class MyTaskRunner(TaskRunner):
    def run(self, config):
        from .myalgo_ray_trainer import RayMyAlgoTrainer
        trainer = RayMyAlgoTrainer(config=config, ...)
        trainer.init_workers()
        trainer.fit()

@hydra.main(config_path="config", config_name="myalgo_trainer", version_base=None)
def main(config):
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(MyTaskRunner))
```

## Common Commands

### Training
```bash
# Run a recipe (example: Archer baseline)
python -m recipe.archer.main_archer \
    data.train_files=... \
    actor_rollout_ref.model.path=... \
    ...

# Multi-node via Ray
ray job submit --runtime-env=runtime_env.yaml \
    --address=http://$HEAD_IP:8265 \
    -- python -m recipe.archer.main_archer ...
```

### Testing
```bash
python -m pytest tests/ -x
```

### Checking config dataclasses
```bash
# See what fields a config accepts
grep -A 30 "class PolicyLossConfig" verl/workers/config/actor.py
grep -A 30 "class FSDPEngineConfig" verl/workers/config/engine.py
grep -A 15 "class SamplingConfig" verl/workers/config/rollout.py
grep -A 30 "class ActorConfig" verl/workers/config/actor.py
```

### Checking registered losses
```bash
grep -r "@register_policy_loss" verl/ recipe/
```

### Exporting W&B metrics (internal Weibo W&B)
Use `scripts/export_wandb_metrics.py` to export run history and key metrics into CSV files for offline analysis.

```bash
# Basic usage (uses WANDB_API_KEY from env)
WANDB_API_KEY=... python scripts/export_wandb_metrics.py \
  --outdir ./wandb_exports_rl_compare

# Override entity/project/base url if needed
WANDB_API_KEY=... python scripts/export_wandb_metrics.py \
  --base-url http://wandb.wml.weibo.com \
  --entity zy9 \
  --project s_3b_old_baseline_stage1 \
  --outdir ./wandb_exports_rl_compare

# Export specific runs (repeat --run)
WANDB_API_KEY=... python scripts/export_wandb_metrics.py \
  --project s_3b_old_baseline_stage1 \
  --run s_run_3b_rl_code24k_0.125-0.875_offpolicy_no-uly_c3 \
  --run run_my_dapo_offpolicy_mini_c6_latest \
  --outdir ./wandb_exports_subset
```

Notes:
- The script queries GraphQL directly (`/graphql`) instead of `wandb.Api()`, which avoids client/server schema mismatch issues on the internal deployment.
- Authentication uses `WANDB_API_KEY` and sends `Authorization: Basic base64("api:<key>")`.
- If no `--run` is provided, it exports the default RL comparison run set defined in the script.

Main outputs:
- `<run>.history.full.csv`: full history rows for one run.
- `<run>.history.focused.csv`: selected key metrics for one run.
- `all_runs.focused.csv`: merged focused metrics across all exported runs.
- `all_runs.summary.csv`: run-level summary metrics (including final `val-core/livecodebench/acc/best@4/mean` if available).
- `export_config.json`: export configuration snapshot (entity, project, run list, focused metrics).

Common troubleshooting:
- Missing key: set `WANDB_API_KEY` or pass `--api-key`.
- Run not found: verify `--entity/--project` and run display names.
- Empty metrics for a column: that run may not log the metric (normal for custom metrics across different recipes).

## Key Algorithms

- **GRPO**: Group-relative advantage, no critic needed. `algorithm.adv_estimator=grpo`
- **PPO/GAE**: Generalized Advantage Estimation with critic. `algorithm.adv_estimator=gae`
- **filter_groups**: Rejection sampling — filters prompt groups with zero reward variance (all correct or all incorrect). `algorithm.filter_groups.enable=True`
- **ASPO (Archer)**: Asymmetric PPO clipping based on per-token entropy. Low-entropy tokens get tight clip, high-entropy get wide clip. `actor_rollout_ref.actor.policy_loss.loss_mode=aspo`

## Common Pitfalls

1. **Config field not in dataclass**: Any field under `actor.*`, `ref.*`, `rollout.*`, `model.*` must be declared in the corresponding Python dataclass. Use `++` prefix only for truly new keys that won't be validated.

2. **dtype path**: It's `actor_rollout_ref.actor.fsdp_config.dtype`, NOT `actor_rollout_ref.actor.dtype`.

3. **Reward function return format**: DAPO reward manager expects `{"score": float}` dict or plain scalar. Other dict keys (e.g., `{"is_correct": True}`) will cause `KeyError: 'score'`.

4. **Reward function parameter name**: DAPO calls with `solution_str=`, not `llm_solution=`.

5. **agg_loss and global_batch_info**: Any extra keys added to `config.global_batch_info` will be passed to `agg_loss(**config.global_batch_info)`. The `**kwargs` in `agg_loss` absorbs them safely.

6. **Ray multi-node**: Always use `ray job submit` WITHOUT `--no-wait` (blocking mode), and clean up stale IP files before starting.

7. **BaseConfig frozen fields**: Fields set once on `BaseConfig` subclasses cannot be changed. Use `_mutable_fields` or store mutable data in dict-type fields like `global_batch_info`.

8. **SamplingConfig has no response_length**: Validation-time generation length is controlled elsewhere, not via `val_kwargs.response_length`.

## Archer Recipe (recipe/archer/)

Custom features on top of DAPO:
- **ASPO loss** (`archer_core_algos.py`): Per-token entropy-based asymmetric clip ratios
- **Overlong filter** (`archer_ray_trainer.py`): Removes responses hitting max_response_length
- **Custom reward** (`rllm/rewards/rl_reward.py`): Handles `data_source="code"` and various code benchmarks

Config toggles:
```yaml
actor_rollout_ref.actor.policy_loss.loss_mode: aspo|vanilla
actor_rollout_ref.actor.calculate_entropy: true|false
trainer.enable_overlong_filter: true|false
algorithm.filter_groups.enable: true|false
```

## Important Paths

- Base training config: `verl/trainer/config/ppo_trainer.yaml`
- Policy loss implementations: `verl/trainer/ppo/core_algos.py`
- Loss entry point: `verl/workers/utils/losses.py`
- Config dataclasses: `verl/workers/config/`
- DAPO reward manager: `verl/experimental/reward_loop/reward_manager/dapo.py`
- Archer recipe: `recipe/archer/`
- Custom rewards: `rllm/rewards/`
