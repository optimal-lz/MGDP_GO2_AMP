# MGDP Go2 AMP Implementation Plan

> **Status (2026-10-14):** Historical implementation plan. The local implementation portions of Tasks 1-12 and the post-implementation remediation are complete; training-host validation remains deferred as recorded below. This file preserves how the implementation was carried out; the normative behavior is defined by the final design spec, and the delivered state is recorded in `docs/MGDP_GO2_AMP_V1_INITIAL_BASELINE.md`.

> **2026-10-16 correction notice:** Whole-project audit found runtime defects in full-config JSON normalization, push timing, GRU training ownership, reset/resume observation coherence, and WM evaluation aggregation. Those corrections are specified and tracked in `docs/superpowers/plans/2026-10-16-mgdp-go2-amp-runtime-corrections.md`; for these areas the corrective plan supersedes this historical plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the single-stage `go2_amp` task that preserves MGDP perception/PPO, adds the frozen WMP-style AMP regularizer, and provides reproducible AMP/noAMP training, restore, visualization, and comparison.

**Architecture:** Keep Isaac Gym, Legged Gym, MGDP World Model, PPO, and Actor-Critic as the foundation. Isolate the formal task in `Go2AmpCfg` plus a dedicated `Go2AmpPolicyRunner`; reuse PPO/WM primitives while replacing stage/ramp/anchor AMP with one unconditional discriminator and replay. Focused checkpoint and evaluation modules protect the original `random_dog_stage1/2` tasks.

**Tech Stack:** Python 3.8, Isaac Gym, PyTorch, Legged Gym, repository MGDP/RSL-RL-derived PPO, NumPy, TensorBoard, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-mgdp-go2-amp-final-design.md`

## Global Constraints

- Work only in this repository; preserve `random_dog_stage1/2`.
- Official project task is exactly `go2_amp`; do not create `amp_v2`.
- Actor uses deployment observations plus proprio token and detached noisy-depth WM token; Critic is exactly 239D with no WM token.
- AMP state is clean 30D ordered `q12, base_lin_vel3, base_ang_vel3, qdot12`; D consumes the 60D consecutive pair.
- Default `lambda_amp=0.01`; no ramp, stage gate, conditioned/multiple D, or RSI.
- noAMP constructs no D, replay, normalizer, or AMP RNG.
- Update order is rollout/online WM, four extra WM updates, GAE, PPO, then D×20.
- Commands, ten terrains, rewards, reset, termination, noise, randomization, and PPO values match the spec exactly.
- Formal default is 4096 envs × 24 steps; no automatic env fallback, microbatch, or gradient accumulation.
- Remain Python 3.8 compatible and do not modify expert files or the Go2 URDF.
- Generated caches, outputs, events, media, CSVs, and checkpoints stay ignored.
- Each task follows TDD and ends in one reviewable commit.

## Review Focus

- Terminal locomotion with reset command resampling keeps the pre-reset terminal state, while a live command change is excluded (Task 4).
- A crash between policy and WM writes never advances `last.json` or loads a mixed pair (Task 8).
- Twenty terrain columns encode 10/10/10/10/15/10/10/10/10/5 percent and level 19 gives `d=1` (Task 3).
- AMP initialization/sampling does not consume environment/policy RNG; noAMP constructs no AMP objects (Tasks 6-7).
- Actor rejects privileged fields and WM gradients while Critic remains exactly 239D (Task 5).

---

## File Map

- Rewrite `legged_gym/legged_gym/envs/go2_amp/{config,terrain,env}.py`; create `commands.py`.
- Extend `legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py` and explicit-column handling in `utils/terrain.py`.
- Modify `rl/MGDP/modules/actor_critic.py` and `algorithms/ppo.py`.
- Rewrite `rl/MGDP/amp/{discriminator,motions,replay,session}.py`; retain/test `state.py` and `transition.py`.
- Create `rl/MGDP/runners/{go2_amp_runner,go2_amp_logging,go2_amp_checkpoint}.py`.
- Create `legged_gym/utils/go2_amp_run.py` and official train/vis/comparison scripts.
- Delete the two project stage training scripts; rewrite obsolete stage/ramp/anchor tests.

### Task 1: Single task configuration and registry

**Files:**
- Rewrite: `legged_gym/legged_gym/envs/go2_amp/config.py:1-100`
- Modify: `legged_gym/legged_gym/envs/__init__.py:18-25`
- Delete: `legged_gym/scripts/train_go2_amp_stage1.py`
- Delete: `legged_gym/scripts/train_go2_amp_stage2.py`
- Rewrite: `tests/test_task_config.py`

**Interfaces:**
- Produces `Go2AmpCfg`, `Go2AmpTrainCfg`, `AMP_EXPERT_GROUPS`, `AMP_EXPERT_GROUP_WEIGHTS`, registry key `go2_amp`.
- Inherits MGDP defaults from `RandomCfgStage1/RandomCfgPPOStage1`, then explicitly overrides every frozen value.

- [ ] **Step 1: Write failing AST configuration tests**

Assert only `go2_amp` is newly registered, original MGDP tasks remain, active experts are the approved eight files, weights are `{"forward":0.8,"turn":0.2}`, and all frozen control/PPO/camera/domain-randomization/reward values equal the spec.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_task_config.py -v`
Expected: FAIL because the single classes/registration do not exist.

- [ ] **Step 3: Implement configuration and registration**

Set 4096 envs, 239 privileged observations, frozen depth/WM/PPO/control/reward/randomization fields, `runner_class_name="Go2AmpPolicyRunner"`, mode `amp`, coefficient 0.01, D updates 20, replay rollouts 2, save interval 1000. Remove stage/ramp/anchor/stratified fields and old registrations.

- [ ] **Step 4: Delete old entrypoints**

Delete the two stage-specific scripts; do not delete the original MGDP visualization/reference scripts.

- [ ] **Step 5: Verify the task contract**

Run: `python -m pytest tests/test_task_config.py -v`
Expected: PASS; original MGDP tasks remain.

- [ ] **Step 6: Commit**

~~~bash
git add legged_gym/legged_gym/envs/go2_amp/config.py legged_gym/legged_gym/envs/__init__.py tests/test_task_config.py
git rm legged_gym/scripts/train_go2_amp_stage1.py legged_gym/scripts/train_go2_amp_stage2.py
git commit -m "refactor: define the single Go2 AMP task"
~~~

### Task 2: Command gating and expert sampling

**Files:**
- Create: `legged_gym/legged_gym/envs/go2_amp/commands.py`
- Modify: `legged_gym/rl/MGDP/amp/motions.py:13-86`
- Rewrite: `tests/test_motion_data.py`
- Create: `tests/test_go2_amp_commands.py`

**Interfaces:**
- Produces `apply_command_deadband(commands)`, `classify_commands(commands)`, `amp_transition_mask(locomotion_before, epoch_before, epoch_after, dones)`, and `MotionDataset.sample_group(count, group, device)`.
- Consumes expert constants from Task 1.

- [ ] **Step 1: Write failing command tests**

Assert `vx<=0.2` and `abs(yaw)<=0.2` map to zero, idle is exactly both zero, structured terrain yaw is zero, and transition eligibility is `locomotion & (same_epoch | done)`.

- [ ] **Step 2: Write failing dataset tests**

Assert only eight files, `FrameDuration==0.02`, within-clip transitions, 80/20 group rate, within-group `MotionWeight`, group-specific sampling, and filename-bearing validation errors.

- [ ] **Step 3: Verify failure**

Run: `python -m pytest tests/test_go2_amp_commands.py tests/test_motion_data.py -v`
Expected: FAIL on missing helpers and old expert groups.

- [ ] **Step 4: Implement helpers and online sampling**

Use a caller-owned NumPy RNG; reject missing, one-frame, non-finite, wrong-width, or wrong-dt clips.

- [ ] **Step 5: Verify the implementation**

Run: same pytest command. Expected: PASS.

- [ ] **Step 6: Commit**

~~~bash
git add legged_gym/legged_gym/envs/go2_amp/commands.py legged_gym/rl/MGDP/amp/motions.py tests/test_go2_amp_commands.py tests/test_motion_data.py
git commit -m "feat: gate commands and expert sampling"
~~~

### Task 3: Ten-terrain geometry and curriculum

**Files:**
- Rewrite: `legged_gym/legged_gym/envs/go2_amp/terrain.py:1-47`
- Modify: `legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py:1-57`
- Modify: `legged_gym/legged_gym/utils/terrain.py:199-228`
- Rewrite: `tests/test_amp_terrain.py`

**Interfaces:**
- Produces `TERRAIN_NAMES`, `TERRAIN_CLASS_IDS`, `TERRAIN_PROPORTIONS`, `TERRAIN_COLUMNS`, `difficulty_for_level(level, num_rows=20)`, `next_curriculum_levels(...)`.
- Consumes forward-command mask from Task 2.

- [ ] **Step 1: Write failing terrain contract tests**

Assert exact names/ids, proportions sum to one, 20 columns count `[2,2,2,2,3,2,2,2,2,1]`, and difficulty maps 0→0 and 19→1.

- [ ] **Step 2: Write failing builder/curriculum tests**

Using fake terrain utilities, assert all names call intended primitives with monotonic geometry. Assert only forward envs update; success increments, early failure decrements, idle/turn remain, and top-level success uses a supplied lower random level.

- [ ] **Step 3: Verify failure**

Run: `python -m pytest tests/test_amp_terrain.py -v`
Expected: FAIL because only six stage columns exist.

- [ ] **Step 4: Implement named builders and pure curriculum**

Build flat, rough flat, smooth/rough slope, stairs, obstacles, hurdle, gap, pit, narrow passage; assign stable ids and safe spawn platforms. For explicit columns use `row/(num_rows-1)` without changing legacy randomized behavior.

- [ ] **Step 5: Verify the implementation**

Run: same pytest command. Expected: PASS.

- [ ] **Step 6: Commit**

~~~bash
git add legged_gym/legged_gym/envs/go2_amp/terrain.py legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py legged_gym/legged_gym/utils/terrain.py tests/test_amp_terrain.py
git commit -m "feat: add the unified ten-terrain curriculum"
~~~

### Task 4: Formal Go2 AMP environment

**Files:**
- Rewrite: `legged_gym/legged_gym/envs/go2_amp/env.py:1-101`
- Modify: `tests/test_amp_transition.py`
- Rewrite: `tests/test_amp_reward.py`
- Create: `tests/test_go2_amp_env_contract.py`

**Interfaces:**
- Produces `get_amp_observations()`, `get_locomotion_mask()`, `compose_total_reward(amp_contribution)`, terminal AMP extras, command epochs, aggregate episode metrics.
- Consumes Tasks 2-3 helpers.

- [ ] **Step 1: Write failing observation/command/termination tests**

Assert clean 30D order, command epoch increments, idle exclusion, structured yaw zero, and only base contact >1N plus timeout truncation—no height/angle/velocity cutoff.

- [ ] **Step 2: Write failing reward tests**

Assert `clamp(task+regularization+safety,0)+amp+termination`; exact air-time cap/gate; idle-only `dof_error`; conditional `feet_edge`, `cheat`, `stuck`; stuck never terminates.

- [ ] **Step 3: Write failing reset/transition tests**

Assert q multiplier `[0.9,1.1]`, zero qdot/root velocity, local height +0.45, spawn rules, history reset, pre-reset capture. Pin terminal changed-epoch inclusion and live changed-epoch exclusion.

- [ ] **Step 4: Verify failure**

Run: `python -m pytest tests/test_amp_transition.py tests/test_amp_reward.py tests/test_go2_amp_env_contract.py -v`.

- [ ] **Step 5: Rewrite the subclass narrowly**

Override command sampling, reward composition, termination, reset placement/history, curriculum, AMP capture, and conditional rewards only; do not add RSI.

- [ ] **Step 6: Verify the implementation**

Run: same command. Expected: PASS.

- [ ] **Step 7: Commit**

~~~bash
git add legged_gym/legged_gym/envs/go2_amp/env.py tests/test_amp_transition.py tests/test_amp_reward.py tests/test_go2_amp_env_contract.py
git commit -m "feat: implement the formal Go2 AMP environment"
~~~

### Task 5: Actor/Critic/WM information boundary

**Files:**
- Modify: `legged_gym/rl/MGDP/modules/actor_critic.py:8-136`
- Modify: `legged_gym/legged_gym/envs/random_dog/random_dog.py:232-611`
- Create: `tests/test_go2_amp_observation_boundary.py`

**Interfaces:**
- Produces `build_actor_input(obs_dict)` with `45+7+64`, `build_critic_input(obs_dict)` with `45+3+4+187=239`.
- Consumes base obs, proprio history, detached `image_buf`, and 194D privileged tensor.

- [ ] **Step 1: Write failing dimension/leakage tests**

Changing privileged values cannot alter actor input; changing WM token cannot alter critic input; critic width is 239.

- [ ] **Step 2: Write failing autograd tests**

Actor backprop does not reach the WM token source, and PPO optimizer contains no environment WM parameters.

- [ ] **Step 3: Verify failure**

Run: `python -m pytest tests/test_go2_amp_observation_boundary.py -v`.

- [ ] **Step 4: Extract builders and detach WM output**

Keep MGDP proprio supervision, but never concatenate target privileged tensors into Actor; WM optimization remains in environment methods.

- [ ] **Step 5: Verify the implementation**

Run: same command plus `tests/test_device_args.py`. Expected: PASS.

- [ ] **Step 6: Commit**

~~~bash
git add legged_gym/rl/MGDP/modules/actor_critic.py legged_gym/legged_gym/envs/random_dog/random_dog.py tests/test_go2_amp_observation_boundary.py
git commit -m "refactor: enforce deployable actor observations"
~~~

### Task 6: Single-discriminator AMP

**Files:**
- Rewrite: `legged_gym/rl/MGDP/amp/discriminator.py:1-24`
- Modify: `legged_gym/rl/MGDP/amp/replay.py:1-54`
- Rewrite: `legged_gym/rl/MGDP/amp/session.py:1-445`
- Rewrite: `tests/test_amp_checkpoint.py`
- Modify: `tests/test_amp_state.py`

**Interfaces:**
- Produces `AMPSession(dataset, device, config, amp_seed)`, `reward(state,next_state,eligible,enabled)`, `record(state,next_state,mask)`, `update()`, complete state round-trip.
- Reward returns AMP contribution only; zero outside mask or when disabled.

- [ ] **Step 1: Write failing architecture/reward tests**

Assert D `60→1024 ReLU→512 ReLU→1`, exact raw reward, idle zero, and rollout 0 disabled without D/normalizer updates.

- [ ] **Step 2: Write failing update/replay tests**

Assert capacity `2*num_envs*num_steps`, one replay, D×20, LS-GAN +1/-1, expert-only GP=10, and normalizer updates only during D update from eligible policy plus experts.

- [ ] **Step 3: Write failing RNG/restore tests**

AMP construction/sampling leaves global NumPy/Torch RNG unchanged. Round-trip D/optimizer/normalizer/replay/private RNG/update count; reject mismatches before mutation.

- [ ] **Step 4: Verify failure**

Run: `python -m pytest tests/test_amp_state.py tests/test_amp_checkpoint.py -v`.

- [ ] **Step 5: Implement and remove obsolete behavior**

Remove stages, ramps, gates, delayed D start, stratified replay, policy-only restore. Initialize D under forked RNG; dataset/replay use private generators.

- [ ] **Step 6: Verify the implementation**

Run: same command. Expected: PASS.

- [ ] **Step 7: Commit**

~~~bash
git add legged_gym/rl/MGDP/amp/discriminator.py legged_gym/rl/MGDP/amp/replay.py legged_gym/rl/MGDP/amp/session.py tests/test_amp_state.py tests/test_amp_checkpoint.py
git commit -m "refactor: implement single-discriminator AMP"
~~~

### Task 7: Dedicated runner and lightweight TensorBoard

**Files:**
- Create: `legged_gym/rl/MGDP/runners/go2_amp_logging.py`
- Create: `legged_gym/rl/MGDP/runners/go2_amp_runner.py`
- Modify: `legged_gym/rl/MGDP/runners/__init__.py`
- Modify: `legged_gym/legged_gym/utils/task_registry.py:83-160`
- Modify: `legged_gym/rl/MGDP/algorithms/ppo.py:109-190`
- Rewrite: `tests/test_amp_rollout.py`
- Create: `tests/test_go2_amp_logging.py`

**Interfaces:**
- Produces standard runner constructor, `learn`, `state_dict/load_state_dict`, `get_inference_policy`; `IterationMetrics.write`.
- Registry uses configured `runner_class_name`, falling back to legacy `MGDPPolicyRunner`.

- [ ] **Step 1: Write failing fake-component order tests**

Assert online WM during rollout, extra WM×4, GAE, PPO, D×20. Rollout 0 passes `enabled=False`; rollout 1 onward true.

- [ ] **Step 2: Write failing mode/transition tests**

noAMP constructs no AMP object and only logs contribution zero. AMP records eligible masks and uses terminal replacement.

- [ ] **Step 3: Write failing logging tests**

Core tags once/iteration; terrain/group/timing every 10; no step/image/histogram writes; reject NaN; resume steps monotonic; noAMP has no D train tags.

- [ ] **Step 4: Verify failure**

Run: `python -m pytest tests/test_amp_rollout.py tests/test_go2_amp_logging.py -v`.

- [ ] **Step 5: Implement runner/logger and PPO aggregate metrics**

Reuse PPO storage and WM methods. Preserve PPO legacy return while exposing `last_update_metrics`. Aggregate existing tensors with one bounded CPU conversion and no extra forward.

- [ ] **Step 6: Verify the implementation**

Run: same command plus `tests/test_task_config.py`. Expected: PASS including legacy fallback.

- [ ] **Step 7: Commit**

~~~bash
git add legged_gym/rl/MGDP/runners/go2_amp_logging.py legged_gym/rl/MGDP/runners/go2_amp_runner.py legged_gym/rl/MGDP/runners/__init__.py legged_gym/legged_gym/utils/task_registry.py legged_gym/rl/MGDP/algorithms/ppo.py tests/test_amp_rollout.py tests/test_go2_amp_logging.py
git commit -m "feat: add the single-stage Go2 AMP runner"
~~~

### Task 8: Atomic paired checkpoints

**Files:**
- Create: `legged_gym/rl/MGDP/runners/go2_amp_checkpoint.py`
- Modify: `legged_gym/rl/MGDP/runners/go2_amp_runner.py`
- Create: `tests/test_go2_amp_checkpoint_pair.py`

**Interfaces:**
- Produces `CheckpointCompatibility`, `save_pair(iteration,policy_state,wm_state,compatibility)`, `load_pair(manifest,expected)`, `resolve_last()`.
- Publication order: temp policy/WM, hash, atomic rename, atomic manifest, then atomic `last.json`.

- [ ] **Step 1: Write failing naming/hash tests**

Assert mode-specific policy name, WM/manifest names, SHA-256, same iteration, pointer-only `last.json`.

- [ ] **Step 2: Write failing interruption/mismatch tests**

Injected mid-save failure leaves old `last.json`. Reject missing file, hash, mode/coef/seed/env/key-config mismatches before runner mutation.

- [ ] **Step 3: Write failing continuity test**

Round-trip PPO, WM, AMP when present, terrain levels, iteration, optimizers, stats, replay, all RNG. noAMP omits AMP keys.

- [ ] **Step 4: Verify failure**

Run: `python -m pytest tests/test_go2_amp_checkpoint_pair.py -v`.

- [ ] **Step 5: Implement manager and runner integration**

Canonical JSON config hash; `os.replace`; save at 1000 multiples and final target; no `last.pt`, best files, or policy-only warm start.

- [ ] **Step 6: Verify the implementation**

Run: same command. Expected: PASS.

- [ ] **Step 7: Commit**

~~~bash
git add legged_gym/rl/MGDP/runners/go2_amp_checkpoint.py legged_gym/rl/MGDP/runners/go2_amp_runner.py tests/test_go2_amp_checkpoint_pair.py
git commit -m "feat: pair policy and world-model checkpoints"
~~~

### Task 9: Unified training CLI and metadata

**Files:**
- Create: `legged_gym/legged_gym/utils/go2_amp_run.py`
- Create: `legged_gym/scripts/train_go2_amp.py`
- Modify: `legged_gym/legged_gym/utils/__init__.py`
- Create: `tests/test_train_go2_amp_cli.py`

**Interfaces:**
- Produces `format_run_name`, `remaining_iterations`, `prepare_run_directory`, `write_run_metadata`, official CLI.
- Consumes registry, runner, checkpoint manager.

- [ ] **Step 1: Write failing naming/target tests**

Default names are `amp_coef0p01_seed1` and `no_amp_seed1`; non-default env count appends `_env64`, avoiding smoke/formal collision. Target 2000 from 1000 gives 1000; target<=current raises.

- [ ] **Step 2: Write failing CLI tests**

Require mode/target; defaults seed1/env4096; optional resume and AMP-only coefficient. Reject coefficient in noAMP. Do not expose task/algo/policy-only; standard Isaac Gym device/headless flags remain.

- [ ] **Step 3: Write failing directory/metadata tests**

No zero-start overwrite; explicit resume required. `config.json` and `run.json` contain effective config, command, commit, runtime versions and experts.

- [ ] **Step 4: Verify failure**

Run: `python -m pytest tests/test_train_go2_amp_cli.py -v`.

- [ ] **Step 5: Implement thin entrypoint**

Force task/runner internally, validate resume before mutable state, and call `learn(target-current)`.

- [ ] **Step 6: Verify the implementation**

Run: same command. Expected: PASS.

- [ ] **Step 7: Commit**

~~~bash
git add legged_gym/legged_gym/utils/go2_amp_run.py legged_gym/legged_gym/utils/__init__.py legged_gym/scripts/train_go2_amp.py tests/test_train_go2_amp_cli.py
git commit -m "feat: add the unified Go2 AMP training entrypoint"
~~~

### Task 10: Viewer, WM snapshots, optional recording

**Files:**
- Create: `legged_gym/scripts/vis_go2_amp.py`
- Create: `tests/test_vis_go2_amp.py`

**Interfaces:**
- Produces paired-manifest Viewer CLI with terrain/level/vx/yaw/env/follow-camera, `--record`, output/duration/fps, optional numbered snapshot.

- [ ] **Step 1: Write failing behavior tests**

Default is Viewer, deterministic frozen networks, fixed seed/few envs, noise/randomization/push off. FFmpeg probe occurs only for `--record`.

- [ ] **Step 2: Write failing path tests**

Video defaults to `<run>/eval/videos`; WM latest paths exactly match spec; numbered snapshots are opt-in.

- [ ] **Step 3: Verify failure**

Run: `python -m pytest tests/test_vis_go2_amp.py -v`.

- [ ] **Step 4: Implement Viewer and hooks**

Reuse live MGDP panels; load strict pair; apply requested case; print command/actual velocity, terrain/level, AMP reward, failure on termination.

- [ ] **Step 5: Implement optional capture**

Probe/import FFmpeg only inside the `--record` branch and write only beneath the selected run directory.

- [ ] **Step 6: Verify the implementation**

Run: same command. Expected: PASS.

- [ ] **Step 7: Commit**

~~~bash
git add legged_gym/scripts/vis_go2_amp.py tests/test_vis_go2_amp.py
git commit -m "feat: add Go2 AMP visualization tools"
~~~

### Task 11: Deterministic AMP/noAMP comparison

**Files:**
- Create: `legged_gym/scripts/evaluate_go2_amp_comparison.py`
- Create: `tests/test_go2_amp_comparison.py`

**Interfaces:**
- Produces `build_cases()`, `evaluate_pair(..., episodes_per_case=20)`, two CSV and four PNG writers.
- Both policies use the AMP checkpoint’s frozen D+normalizer for style score.

- [ ] **Step 1: Write failing case tests**

Assert 60 forward cases (10 terrains×3 levels×2 vx) plus 12 pure-turn cases (flat/rough×3 levels×yaw ±0.5), 72 total, 20 episodes each.

- [ ] **Step 2: Write failing fairness/style tests**

Identical case order/seeds/physics; training/noise/randomization/push off; frozen shared D; no optimizer/replay writes.

- [ ] **Step 3: Write failing schema tests**

Only `summary.csv`, `cases.csv`, `completion.png`, `tracking.png`, `smoothness.png`, `style_score.png`. Include all spec metrics/case keys; no Markdown/video/pass-fail.

- [ ] **Step 4: Verify failure**

Run: `python -m pytest tests/test_go2_amp_comparison.py -v`.

- [ ] **Step 5: Implement matched headless evaluation**

Reject incompatible manifests before environments; deterministic actions and matched episode seeds.

- [ ] **Step 6: Verify the implementation**

Run: same command. Expected: PASS.

- [ ] **Step 7: Commit**

~~~bash
git add legged_gym/scripts/evaluate_go2_amp_comparison.py tests/test_go2_amp_comparison.py
git commit -m "feat: compare AMP and noAMP policies"
~~~

### Task 12: Documentation and full verification

**Files:**
- Modify: `README.md`
- Rewrite/remove: remaining tests asserting stage/ramp/anchor/gate/double replay/policy-only warm start.
- Verify: all Tasks 1-11 files.

**Interfaces:**
- Produces reproducible smoke/formal/resume/viewer/record/comparison commands.

- [ ] **Step 1: Update README**

Document 64-env paired smoke, 4096 AMP smoke, absolute targets 1000/2000/5000, manifests, Viewer, optional recording, WM PNG and comparison outputs. Note existing expert video is prior evidence and is not rerun.

- [ ] **Step 2: Run CPU/static suite**

Run: `python -m pytest tests -v`
Expected: PASS, with only explicitly GPU/Isaac-marked SKIP; no obsolete design assertions.

- [ ] **Step 3: Run syntax/repository checks**

Run: `python -m compileall -q legged_gym/legged_gym legged_gym/rl legged_gym/scripts tests`
Run: `git diff --check`
Expected: exit 0; caches ignored.

- [ ] **Step 4: Run RTX 4090 functionality smoke**

~~~bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 64 --target_iteration 10 --headless
python legged_gym/scripts/train_go2_amp.py --mode no_amp --seed 1 --num_envs 64 --target_iteration 10 --headless
~~~

Expected: finite WM/PPO/D; noAMP has no D/replay tags; event files and complete final pairs exist.

- [ ] **Step 5: Run full-load AMP smoke**

~~~bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 4096 --target_iteration 10 --headless
~~~

Expected: completes on RTX 4090 24GB without fallback; on OOM stop and diagnose measured allocation before design changes.

- [ ] **Step 6: Verify paired resume**

Resume the 4096 run to absolute target 20; confirm monotonic TensorBoard steps and checkpoint/RNG/terrain state continuity.

- [ ] **Step 7: Verify Viewer**

View one flat and one structured case with live panels and terminal print; optional recording remains manual.

- [ ] **Step 8: Commit documentation**

~~~bash
git add README.md tests
git commit -m "docs: document the formal Go2 AMP workflow"
~~~

- [ ] **Step 9: Final review**

Map every approved spec section to implementation/tests/manifests. Confirm `git status --short` has no generated or unexpected files.

## Post-implementation review and remediation

The plan above remains a historical execution record. The following later findings refine the delivered V1 behavior without rewriting that history:

- `8104e96` — enforced the formal randomization allowlist; separated action latency from motor strength/offset; connected camera-mount jitter; made Viewer/comparison stochasticity fully disabled; restored PPO adaptive learning rate and derived terrain state; completed the frozen TensorBoard schema; aligned the formal CLI to `--amp_coef`.
- `8d653e2` — explicitly disabled inherited `terrain_adaptive_reward` for `go2_amp`, preventing legacy MGDP terrain ids from scaling the new task's `collision`, `lin_vel_z`, and `orientation` terms.
- Local verification at this baseline: 93 tests passed, 10 PyTorch/Isaac training-host tests skipped as declared; `compileall` and `git diff --check` passed.
- Deferred verification: 64-env AMP/noAMP smoke, paired resume, Viewer checks, and the 4096-env RTX 4090 load test still require the Linux CUDA/Isaac Gym host.
- The `random_dog_stage1/2` registrations remain temporarily present. They are not a correctness or compatibility target for this new project and may be removed later if they interfere with GPU training. The required MGDP perception, World Model, history encoder, PPO, and shared environment primitives remain dependencies of `go2_amp`.

When this historical plan conflicts with later wording, precedence is: final design specification for intended behavior, V1 initial baseline for delivered/verified state, then this implementation plan for execution history.
