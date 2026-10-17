# MGDP Go2 AMP Runtime Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the formal Go2 AMP entrypoint, timing, temporal perception, reset/resume, evaluation, and logging defects found in the 2026-10-16 whole-project audit without restoring legacy MGDP behavior.

**Architecture:** Keep the existing single-stage MGDP CNN/map World Model, proprio encoder, asymmetric PPO, and single AMP discriminator. Normalize the complete run configuration once before hashing or persistence; drive push from parsed policy-step intervals; move the 64D GRU from the environment WM into Actor/PPO with contiguous per-environment recurrent minibatches; and give training, resume, Viewer, and comparison one coherent fresh-observation/reset contract.

**Tech Stack:** Python 3.8, Isaac Gym, PyTorch, Legged Gym, repository MGDP/RSL-RL-derived PPO, NumPy, unittest, TensorBoard.

**Spec:** `docs/superpowers/specs/2026-10-12-mgdp-go2-amp-final-design.md`

## Global Constraints

- Work only in `D:\New Beginning Help\LZAMP\MGDP_GO2_AMP`; do not create a parallel project or `amp_v2`.
- Preserve the MGDP CNN/map encoder-decoder, contrastive loss, proprio history encoder, PPO base, and original reference task registrations needed by `go2_amp`.
- New-project correctness takes priority over behavioral compatibility with old MGDP tasks.
- Actor receives only deployment information; Critic remains exactly 239D; AMP remains clean 30D state/60D transition and independent of Actor/Critic inputs.
- CNN token is detached before policy memory. Policy GRU is PPO-owned and never appears in the WM optimizer or WM checkpoint.
- Depth remains interval 5. Policy GRU updates only on a true depth refresh and resets per done environment.
- Resume begins new episodes on restored terrain levels; it does not claim exact mid-episode simulator continuation.
- Formal randomization remains friction, action latency, and push only. Push period is exactly 15 seconds at policy dt 0.02 seconds.
- Remain Python 3.8 compatible; do not modify expert files or the Go2 URDF.
- Every behavior change follows RED→GREEN TDD. Documentation-only edits do not require a synthetic prose test.

## Review Focus

- Full inherited configs containing NumPy arrays, NumPy scalars, tuples, nested lists, and dicts must hash, write, and compare identically after JSON round trip.
- The first push must occur at policy step 750, not step 1 or 15; no division-by-zero or fractional interval can silently enter the callback.
- Recurrent PPO must preserve time order, reset hidden state after dones, hold hidden state on cached-depth steps, and propagate PPO gradients into GRU but never into the WM CNN token source.
- A reset between camera refreshes must never expose the preceding episode's visual token or derived Actor state; resume/evaluation must not act before loaded-WM visual refresh.
- WM comparison metrics must be per environment and count refresh frames only; TensorBoard must aggregate all actual WM updates in an iteration.

---

### Task 1: Make the formal entrypoint serializable and timing/noise exact

**Files:**
- Modify: `legged_gym/legged_gym/utils/go2_amp_run.py`
- Modify: `legged_gym/scripts/train_go2_amp.py`
- Modify: `legged_gym/legged_gym/envs/baseline/randomization.py`
- Modify: `legged_gym/legged_gym/envs/baseline/legged_robot_terrains.py`
- Modify: `legged_gym/legged_gym/envs/go2_amp/config.py`
- Modify: `tests/test_train_go2_amp_cli.py`
- Modify: `tests/test_go2_amp_randomization.py`
- Modify: `tests/test_task_config.py`

**Interfaces:**
- Produces `normalize_effective_config(value) -> JSON-native value` in `go2_amp_run.py`.
- Produces `push_is_due(common_step_counter, push_interval_steps) -> bool` in `randomization.py`.
- `train_go2_amp.py` supplies the same normalized environment/training dictionaries to metadata and checkpoint compatibility.

- [ ] **Step 1: Write failing configuration round-trip tests**

Add tests that pass nested NumPy arrays/scalars and tuples through `normalize_effective_config`, assert literal list/Python-scalar output, write metadata twice, and confirm the second call accepts the JSON-loaded configuration unchanged.

- [ ] **Step 2: Run the configuration tests and verify RED**

Run: `python -m unittest tests.test_train_go2_amp_cli -v`

Expected: FAIL because `normalize_effective_config` does not exist and raw NumPy arrays are not JSON serializable.

- [ ] **Step 3: Implement one canonical configuration normalization boundary**

Normalize dict keys/values recursively, convert tuple/list to list, NumPy arrays to nested lists, and NumPy scalars to Python scalars. Call it immediately after `class_to_dict` and before constructing `runner_config`, hashing, metadata writing, or resume comparison.

- [ ] **Step 4: Write failing push-period and clean-height tests**

Assert `push_is_due(1, 750)` and `push_is_due(15, 750)` are false, `push_is_due(750, 750)` and `push_is_due(1500, 750)` are true, and `Go2AmpCfg.noise.add_height_noise` is explicitly false.

- [ ] **Step 5: Run timing/config tests and verify RED**

Run: `python -m unittest tests.test_go2_amp_randomization tests.test_task_config -v`

Expected: FAIL because the callback uses `push_interval_s` as a step count and Go2 does not override inherited height noise.

- [ ] **Step 6: Use parsed push steps and disable height noise**

Validate the interval is a positive integer-like value, use `cfg.domain_rand.push_interval` in the active callback, and set `Go2AmpCfg.noise.add_height_noise=False` plus `add_privileged_noise=False` explicitly.

- [ ] **Step 7: Verify Task 1**

Run: `python -m unittest tests.test_train_go2_amp_cli tests.test_go2_amp_randomization tests.test_task_config -v`

Expected: PASS with canonical config round trip, first push at step 750, and clean height supervision.

### Task 2: Move temporal visual memory into PPO Actor

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2_amp/config.py`
- Modify: `legged_gym/legged_gym/envs/random_dog/random_dog.py`
- Modify: `legged_gym/rl/MGDP/modules/actor_critic.py`
- Modify: `legged_gym/rl/MGDP/algorithms/ppo.py`
- Modify: `legged_gym/rl/MGDP/storage/rollout_storage.py`
- Modify: `legged_gym/rl/MGDP/runners/policy_runner.py`
- Modify: `legged_gym/rl/MGDP/runners/go2_amp_runner.py`
- Modify: `tests/test_go2_amp_observation_boundary.py`
- Modify: `tests/test_amp_rollout.py`
- Modify: `tests/test_go2_amp_checkpoint_pair.py`

**Interfaces:**
- Environment observation dictionary adds `image_update_mask: bool[N,1]`; `image_buf` becomes the detached 64D CNN token, not an environment-GRU output.
- `ActorCritic.get_hidden_states() -> Tensor[N,64]` returns the pre-observation visual hidden state; `reset(dones)` clears selected rows.
- `RolloutStorage.recurrent_mini_batch_generator(...)` yields complete `[T,B,...]` environment sequences, the hidden state at `T=0`, depth refresh masks, and episode-start masks.
- Actor policy checkpoint owns GRU parameters; WM checkpoint contains image/map encoder-decoder only.

- [ ] **Step 1: Write failing Actor temporal-memory tests**

Using real PyTorch tensors, assert: CNN source gets no PPO gradient; GRU parameters do get PPO/actor gradient; hidden changes when refresh mask is true; hidden is identical when false; `reset(dones)` clears only selected rows; sequence mode zeros hidden after the preceding done.

- [ ] **Step 2: Run observation tests and verify RED**

Run: `python -m unittest tests.test_go2_amp_observation_boundary -v`

Expected: FAIL because Actor has no visual GRU or refresh-mask contract.

- [ ] **Step 3: Implement Actor-owned gated 64D GRU**

Use a single-layer `GRUCell(64,64)` or equivalent explicit time loop. In rollout mode maintain detached runtime hidden state; in update mode consume explicit initial hidden and `[T,B,1]` refresh/episode-start masks without mutating rollout state. Concatenate the resulting 64D temporal token with the existing 45D base observation and 7D proprio token.

- [ ] **Step 4: Write failing recurrent storage/PPO tests**

Assert minibatches partition environments rather than shuffled transitions, preserve all 24 ordered steps, construct episode starts from preceding dones, return the stored rollout-start hidden state, and expose refresh masks. Assert the recurrent generator name called by PPO exists.

- [ ] **Step 5: Run rollout tests and verify RED**

Run: `python -m unittest tests.test_amp_rollout tests.test_go2_amp_observation_boundary -v`

Expected: FAIL because storage currently has only the misspelled incomplete recurrent generator and does not store refresh masks.

- [ ] **Step 6: Implement contiguous recurrent PPO updates**

Store pre-action hidden and refresh mask per transition. For each epoch, shuffle environment indices only, select complete `T×B` sequences, and derive episode-start masks from prior dones. Compute PPO and auxiliary losses across every valid sequence element while backpropagating through the Actor GRU.

- [ ] **Step 7: Remove GRU from WM ownership**

Set the formal environment camera memory off and policy visual memory on. Publish raw detached CNN tokens plus refresh masks. Remove environment memory parameters from the formal WM optimizer/state/validation path; retain legacy classes only for reference tasks that still enable them.

- [ ] **Step 8: Verify Task 2**

Run: `python -m unittest tests.test_go2_amp_observation_boundary tests.test_amp_rollout tests.test_go2_amp_checkpoint_pair -v`

Expected: PASS; Actor GRU has PPO gradients, WM boundary remains detached, recurrent minibatches retain time order, and GRU is policy-checkpoint-owned.

### Task 3: Make reset and paired resume start from coherent observations

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2_amp/env.py`
- Modify: `legged_gym/legged_gym/envs/random_dog/random_dog.py`
- Modify: `legged_gym/legged_gym/envs/baseline/legged_robot_terrains.py`
- Modify: `legged_gym/rl/MGDP/runners/go2_amp_runner.py`
- Modify: `tests/test_go2_amp_env_contract.py`
- Modify: `tests/test_go2_amp_checkpoint_pair.py`

**Interfaces:**
- `Go2AmpRandomDog.refresh_reset_state(env_ids)` recomputes root-derived Actor fields, refreshes selected terrain heights, clears stale foot-height privilege, cached visual token, refresh mask, and histories.
- `Go2AmpRandomDog.restart_after_checkpoint_load()` starts all environments as new episodes at restored origins without applying curriculum, clears Actor runtime memory, and forces one inference-only visual refresh with loaded WM weights.
- `Randomdog.refresh_visual_observation()` performs camera/WM inference without an optimizer step and marks a true image refresh.

- [ ] **Step 1: Write failing reset-cache tests**

Test the pure/stateful parts with lightweight tensor fixtures: selected rows of cached visual token and GRU/history state clear; root velocity produces zero base velocity after reset; selected height refresh does not use terminal positions; untouched environments remain unchanged.

- [ ] **Step 2: Run environment contract tests and verify RED**

Run: `python -m unittest tests.test_go2_amp_env_contract -v`

Expected: FAIL because reset clears raw buffers but leaves cached visual and derived observations.

- [ ] **Step 3: Implement fresh reset state without extra per-reset camera rendering**

On ordinary asynchronous episode reset, clear the selected visual token and hold GRU at zero until the next scheduled global depth refresh. Recompute base quaternion/velocity/gravity and selected height samples from reset root state; clear stale foot-height privilege. Do not trigger an all-environment camera render whenever one environment resets.

- [ ] **Step 4: Write failing resume-restart tests**

Assert load order is policy + WM + terrain/RNG restore, then new-episode reset at restored origins, Actor hidden reset, and an inference-only forced visual refresh. Assert no WM optimizer step is performed by the refresh.

- [ ] **Step 5: Run checkpoint tests and verify RED**

Run: `python -m unittest tests.test_go2_amp_checkpoint_pair -v`

Expected: FAIL because restore currently changes terrain metadata only and retains pre-load observations.

- [ ] **Step 6: Implement explicit new-episode resume semantics**

After both checkpoint halves are loaded, suppress curriculum for one all-environment reset, preserve restored terrain levels/origins, clear histories and Actor hidden, execute the normal zero-action reset step, then force loaded-WM inference. Restore the saved RNG before this restart; document that the restart intentionally consumes it.

- [ ] **Step 7: Verify Task 3**

Run: `python -m unittest tests.test_go2_amp_env_contract tests.test_go2_amp_checkpoint_pair -v`

Expected: PASS with no terminal cache leakage and coherent resume origins/observations.

### Task 4: Make Viewer/comparison and WM metrics use the same state contract

**Files:**
- Modify: `legged_gym/legged_gym/envs/random_dog/random_dog.py`
- Modify: `legged_gym/rl/MGDP/runners/go2_amp_runner.py`
- Modify: `legged_gym/scripts/vis_go2_amp.py`
- Modify: `legged_gym/scripts/evaluate_go2_amp_comparison.py`
- Modify: `tests/test_vis_go2_amp.py`
- Modify: `tests/test_go2_amp_comparison.py`
- Modify: `tests/test_go2_amp_logging.py`

**Interfaces:**
- Environment publishes `wm_metric_mask: bool[N]`, `visual_mse_per_env: float[N]`, and `height_mse_per_env: float[N]` for the latest refresh.
- Environment exposes reset/get methods for iteration WM metric sums and update count.
- Viewer/comparison reset policy GRU after each `dones` and force one visual refresh after setting the requested case.

- [ ] **Step 1: Write failing per-environment refresh-only metric tests**

Use literal two-environment, five-step fixtures where only one step is a refresh. Assert each environment's mean uses its own loss and denominator 1, not five and not a shared batch mean.

- [ ] **Step 2: Run comparison/logging tests and verify RED**

Run: `python -m unittest tests.test_go2_amp_comparison tests.test_go2_amp_logging -v`

Expected: FAIL because evaluation currently adds one batch scalar on every step and runner logs only the last WM update.

- [ ] **Step 3: Publish per-environment WM losses and aggregate update metrics**

Compute reconstruction MSE with reduction over non-batch dimensions. Mark only actual sensor refresh outputs valid. Accumulate detached scalar WM losses and count every online/extra update; reset once at iteration start and log means after all four extra updates.

- [ ] **Step 4: Write failing Viewer/comparison first-action tests**

Assert scripts call the shared reset/forced-refresh path after checkpoint plus case setup, and call `actor_critic.reset(dones)` after every step.

- [ ] **Step 5: Run visualization tests and verify RED**

Run: `python -m unittest tests.test_vis_go2_amp tests.test_go2_amp_comparison -v`

Expected: FAIL because loaded policies can act on pre-load/previous-case tokens and recurrent hidden is not reset on done.

- [ ] **Step 6: Apply the shared evaluation reset contract**

After setting terrain/level/command, reset all case environments, refresh root-derived state, force one loaded-WM inference frame, and reset policy hidden before the first action. Reset hidden with each returned done mask. Divide WM sums by refresh counts only.

- [ ] **Step 7: Verify Task 4**

Run: `python -m unittest tests.test_vis_go2_amp tests.test_go2_amp_comparison tests.test_go2_amp_logging -v`

Expected: PASS with clean first actions, recurrent done handling, per-environment comparison WM errors, and iteration-mean TensorBoard WM losses.

### Task 5: Record the corrected V1 state and verify the whole project

**Files:**
- Modify: `docs/MGDP_GO2_AMP_V1_INITIAL_BASELINE.md`
- Modify: `docs/superpowers/plans/2026-10-12-mgdp-go2-amp-implementation.md`
- Modify: `README.md`
- Modify: this plan with completion status if execution tooling records it here or in its ledger

**Interfaces:**
- Documents distinguish the 2026-10-14 initial baseline from the 2026-10-16 corrective implementation.
- GPU-only claims remain explicitly unverified until cloud smoke is executed.

- [ ] **Step 1: Update the baseline and historical-plan pointers**

Record each corrected defect, new checkpoint incompatibility caused by policy-side GRU ownership, exact local verification command/count, and remaining Linux RTX 4090 smoke steps. Mark the 2026-10-12 plan as superseded for these correction areas and link this plan.

- [ ] **Step 2: Update user-facing training notes**

Ensure README states that checkpoints created before the corrective version are not formal-resume compatible, resume starts new episodes at restored curriculum levels, and short smoke precedes 4096-env training.

- [ ] **Step 3: Run the complete local suite**

Run: `python -m unittest discover -s tests -p 'test_*.py' -v`

Expected: all runnable tests pass; training-host skips are enumerated rather than described as passes.

- [ ] **Step 4: Run static compilation and repository checks**

Run: `python -m compileall -q legged_gym/legged_gym legged_gym/rl legged_gym/scripts tests`

Expected: exit 0.

Run: `git diff --check`

Expected: exit 0.

- [ ] **Step 5: Review the final diff against the corrected spec**

Check configuration normalization, push units, GRU ownership/gradients, observation dimensions, reset/resume order, evaluation refresh counts, checkpoint fields, and documentation line by line. Do not claim Isaac Gym runtime success from static/local tests.

