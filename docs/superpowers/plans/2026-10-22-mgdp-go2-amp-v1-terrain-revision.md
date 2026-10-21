# MGDP GO2 AMP V1 Terrain Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the invalid V1 terrain implementation with the approved seven-type, ten-level directional terrain curriculum, while preserving the training algorithm and isolating every corrected run from the old output namespace.

**Architecture:** Keep the formal terrain contract and evaluation helpers in `envs/go2_amp/terrain.py`, move the new height-field construction into a focused NumPy-based generator module, and let the shared `Terrain` class consume per-tile spawn metadata without changing legacy unnamed terrain generation. Adapt the environment, viewers, paired evaluator, and training path at their existing narrow integration points; AMP, WM, PPO, observations, randomization, checkpoint pairing, and reward weights remain unchanged.

**Tech Stack:** Python 3.8, NumPy/SciPy height fields, PyTorch, Isaac Gym Preview 4, Warp sensor, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-22-mgdp-go2-amp-v1-terrain-revision.md`

## Global Constraints

- Formal terrain set and weights are exactly: `flat` 15%, `rough_ground` 15%, `slope_course` 20%, `stairs_course` 20%, `low_obstacle_course` 15%, `gap_course` 10%, `narrow_passage` 5%.
- Every tile remains `8 m x 8 m`; curriculum rows are exactly 10 (`level=0..9`, `difficulty=level/9`).
- Do not change AMP, WM, PPO, actor/critic observations, randomization ranges, reward weights, paired checkpoint structure, manifest format, or the formal training sequence.
- Never apply a generic center-clearing pass after terrain generation; each generator owns its spawn, course, recovery, and exit geometry.
- New formal runs live only under `legged_gym/outputs/go2_amp_v1/`; never overwrite, migrate, or resume `legged_gym/outputs/go2_amp/` runs.
- Preserve old runs as pre-revision engineering evidence only. Do not delete artifacts in this implementation.
- Leave `docs/MGDP_GO2_AMP_V1_INITIAL_BASELINE.md` unchanged until corrected cloud validation passes and the user separately approves an update.
- Work around unrelated local modifications; do not stage `docs/MGDP_GO2_AMP_V1_INITIAL_BASELINE.md` or `docs/figures/`.

## Review Focus

- Meter-to-height-field quantization at 0.1 m horizontal and 0.005 m vertical scale must preserve section endpoints and never write outside an 8 m tile; Task 2 adds boundary and section-slice tests.
- The first/last samples of ramps, stairs, and rough transitions must not reintroduce a seam or zero-height pit at the top platform; Task 2 tests continuity at every named boundary.
- Seeded rough terrain must reproduce exactly for the same seed yet differ for another seed; Task 2 pins both cases.
- Environment counts not divisible by 20 physical terrain columns must still map every environment to a valid column; Task 1 adds a clamping/distribution test for `_get_env_origins`.
- A copied or explicitly supplied pre-revision run must fail before training state is loaded; Task 3 tests old-root rejection and the terrain revision field in effective config.

---

### Task 1: Freeze the seven-terrain contract and ten-level environment behavior

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2_amp/terrain.py`
- Modify: `legged_gym/legged_gym/envs/go2_amp/config.py`
- Modify: `legged_gym/legged_gym/envs/go2_amp/env.py`
- Modify: `tests/test_amp_terrain.py`
- Modify: `tests/test_go2_amp_env_contract.py`
- Modify: `tests/test_go2_amp_commands.py`
- Modify: `tests/test_task_config.py`

**Interfaces:**
- Produces: `TERRAIN_NAMES`, `TERRAIN_CLASS_IDS`, `TERRAIN_PROPORTIONS`, `TERRAIN_COLUMNS`, `TERRAIN_OVERVIEW_LEVELS=(0, 3, 6, 9)`, and `difficulty_for_level(level, num_rows=10)`.
- Produces: `Go2AmpCfg.terrain.revision = "v1_terrain_r1"`, `num_rows=10`, and the existing 20-column exact-weight grid.
- Consumes: existing `apply_terrain_command_policy(...)` and `next_curriculum_levels(...)` interfaces unchanged.

- [ ] **Step 1: Rewrite the contract tests to fail on the old ten-terrain design**

  Assert the exact seven names/IDs, proportions `(0.15, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05)`, 20-column counts `(3, 3, 4, 4, 3, 2, 1)`, preview levels `(0, 3, 6, 9)`, 28 overview cases, level bounds `0..9`, and curriculum top-level recycle with `max_level=9`.

- [ ] **Step 2: Add failing environment/config assertions**

  Pin turning commands to `flat` and `rough_ground`; structured commands to `vx>0.2` and `yaw=0`; `feet_edge` to `gap_course & (level>=2)` only; heading deviation to all five structured classes; `lin_vel_z=-1.0`; `num_rows=10`; and `terrain.revision="v1_terrain_r1"`. Add a pure helper or AST-extractable test showing terrain-column assignment stays in `[0, 19]` for non-divisible environment counts.

- [ ] **Step 3: Run the focused tests and confirm they fail for the retired names/levels**

  Run: `python -m unittest tests.test_amp_terrain tests.test_go2_amp_env_contract tests.test_go2_amp_commands tests.test_task_config -q`

  Expected: FAIL on the old terrain tuple, 20 rows, old reward IDs/threshold, or old column calculation.

- [ ] **Step 4: Implement the formal contract and environment adaptations**

  Replace only the terrain-dependent constants and conditions. Preserve the existing command deadband, curriculum success/failure algorithm, reward scales, termination rules, and logging. Make `_get_env_origins` assign valid columns for arbitrary positive `num_envs` without changing the 4096-env distribution intent.

- [ ] **Step 5: Run the focused tests**

  Run: `python -m unittest tests.test_amp_terrain tests.test_go2_amp_env_contract tests.test_go2_amp_commands tests.test_task_config -q`

  Expected: PASS.

- [ ] **Step 6: Commit the contract change**

  ```bash
  git add legged_gym/legged_gym/envs/go2_amp/terrain.py legged_gym/legged_gym/envs/go2_amp/config.py legged_gym/legged_gym/envs/go2_amp/env.py tests/test_amp_terrain.py tests/test_go2_amp_env_contract.py tests/test_go2_amp_commands.py tests/test_task_config.py
  git commit -m "feat: revise Go2 AMP terrain contract"
  ```

### Task 2: Build deterministic directional height fields and front spawn origins

**Files:**
- Create: `legged_gym/legged_gym/utils/new_terrains/go2_amp_terrain.py`
- Modify: `legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py`
- Modify: `legged_gym/legged_gym/utils/terrain.py`
- Modify: `legged_gym/legged_gym/envs/go2_amp/env.py`
- Rewrite: `tests/test_amp_terrain.py`

**Interfaces:**
- Produces: `build_go2_amp_terrain(terrain, terrain_name: str, difficulty: float, seed: int) -> None`; it fills `height_field_raw`, sets `terrain.idx`, and sets local `terrain.spawn_origin`.
- Produces: `trimesh_terrain_by_name(..., seed=None)` as the compatibility wrapper used by `Terrain.make_terrain_by_name(...)`.
- Produces: named terrain tiles with local spawn `(0.65, 4.0, z)` except centered `flat`; `Terrain.add_terrain_to_map(...)` translates this metadata to world coordinates while legacy terrain paths retain their current center-origin behavior.
- Consumes: the Task 1 terrain names/IDs and `cfg.terrain.revision`.

- [ ] **Step 1: Add generator tests using an 80 x 80 sample tile**

  Test exact geometry at `difficulty=0` and `1`: slope ratios `0.05/0.30`; seven 0.30 m stairs per side with heights `0.05/0.14`; obstacles at x `2/4/6 m`, width `0.30 m`, heights `0.04/0.16 m`; gaps near x `2.6/5.4 m`, widths `0.10/0.35 m`, depth `1.0 m`; and narrow widths `1.50/1.00 m` with 0.35 m walls and the `1.3+0.7+4.0+0.7+1.3 m` layout.

- [ ] **Step 2: Add continuity, boundary, and roughness tests**

  Assert every write stays inside the tile; the slope/stairs top platform inherits the ascent height and their exit returns to zero; rough ground has 1.3 m zero-height ends, smooth transitions, approximately zero mean, amplitude at most `0.005+0.045*d` (within one vertical quantum), identical arrays for equal seeds, and unequal arrays for different seeds. Assert no generator invokes or contains the retired generic `clear_spawn_platform` behavior.

- [ ] **Step 3: Add spawn integration tests**

  Verify named tiles carry correct local spawn height, `Terrain.add_terrain_to_map(...)` uses the local origin, structured X jitter is `±0.10 m` with zero Y jitter, rough-ground spawn remains inside its front platform, and flat retains small XY jitter. Keep tests isolated from Isaac Gym by extracting the narrow methods or stubbing `SubTerrain` as current tests do.

- [ ] **Step 4: Run tests and confirm failure**

  Run: `python -m unittest tests.test_amp_terrain tests.test_go2_amp_env_contract -q`

  Expected: FAIL because the focused generator and spawn metadata do not exist.

- [ ] **Step 5: Implement the seven generators and seed plumbing**

  Use integer sample boundaries derived from physical meters, clamp difficulty to `[0,1]`, and use a local NumPy RNG. Set `cfg.terrain.generation_seed=cfg.seed` before the base environment constructs `Terrain`; derive a stable tile seed from the run seed, row, and column. Keep legacy `trimesh_terrain(...)` and unnamed terrain generation untouched.

- [ ] **Step 6: Implement origin translation and reset jitter**

  Use `terrain.spawn_origin` only when present; otherwise retain the shared Terrain class's existing center-origin calculation. Do not clear or flatten any post-generation region.

- [ ] **Step 7: Run focused and legacy terrain tests**

  Run: `python -m unittest tests.test_amp_terrain tests.test_go2_amp_env_contract -q`

  Expected: PASS, including legacy dispatch coverage.

- [ ] **Step 8: Commit the geometry change**

  ```bash
  git add legged_gym/legged_gym/utils/new_terrains/go2_amp_terrain.py legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py legged_gym/legged_gym/utils/terrain.py legged_gym/legged_gym/envs/go2_amp/env.py tests/test_amp_terrain.py tests/test_go2_amp_env_contract.py
  git commit -m "feat: add directional Go2 AMP terrain courses"
  ```

### Task 3: Isolate corrected runs and reject pre-revision resume paths

**Files:**
- Modify: `legged_gym/scripts/train_go2_amp.py`
- Modify: `tests/test_train_go2_amp_cli.py`
- Modify: `tests/test_go2_amp_checkpoint_pair.py`

**Interfaces:**
- Produces: formal output root `legged_gym/outputs/go2_amp_v1/` with existing run-name formatting beneath it.
- Produces: `_resolve_resume_manifest(resume_from, run_dir)` that accepts only the derived run directory in the new namespace.
- Consumes: `Go2AmpCfg.terrain.revision` through the existing normalized effective config and checkpoint config hash.

- [ ] **Step 1: Add failing namespace and resume tests**

  Assert the script contains one formal root named `go2_amp_v1`; fresh AMP/noAMP paths differ from every `outputs/go2_amp/...` path; an old-root directory or manifest raises before `runner.load_pair`; a valid same-run resume remains accepted; and serialized environment config contains terrain revision `v1_terrain_r1`.

- [ ] **Step 2: Run the focused tests and confirm failure on `outputs/go2_amp`**

  Run: `python -m unittest tests.test_train_go2_amp_cli tests.test_go2_amp_checkpoint_pair -q`

  Expected: FAIL on the old output root or missing revision marker.

- [ ] **Step 3: Switch the official output root and retain strict path checks**

  Change only the root namespace. Do not copy, rename, delete, or inspect old run binaries; keep `format_run_name`, paired files, manifests, TensorBoard layout, and absolute-target iteration semantics unchanged.

- [ ] **Step 4: Run the focused tests**

  Run: `python -m unittest tests.test_train_go2_amp_cli tests.test_go2_amp_checkpoint_pair -q`

  Expected: PASS.

- [ ] **Step 5: Commit the run isolation change**

  ```bash
  git add legged_gym/scripts/train_go2_amp.py tests/test_train_go2_amp_cli.py tests/test_go2_amp_checkpoint_pair.py
  git commit -m "fix: isolate corrected terrain runs"
  ```

### Task 4: Update the overview and single-case Viewer contracts

**Files:**
- Modify: `legged_gym/scripts/view_go2_amp_terrains.py`
- Modify: `legged_gym/scripts/vis_go2_amp.py`
- Modify: `tests/test_view_go2_amp_terrains.py`
- Modify: `tests/test_vis_go2_amp.py`

**Interfaces:**
- Consumes: Task 1's 28-item `terrain_overview_layout()` and Task 2 spawn origins.
- Produces: a checkpoint-free 7-row x 4-level overview with printed names, levels, weights, and env IDs; camera is set once before the loop and is not continuously overwritten.
- Produces: existing single-case Viewer CLI with new terrain choices and the same deterministic checkpoint load behavior.

- [ ] **Step 1: Add failing overview and case-validation tests**

  Assert exactly 28 display environments, levels `0/3/6/9`, seven printed rows, no retired names, a centered full-grid initial camera, and no `set_camera` call inside the simulation loop. Assert `vis_go2_amp.py` allows yaw only for `flat`/`rough_ground`, validates level `0..9`, and accepts all seven new names through `TERRAIN_NAMES`.

- [ ] **Step 2: Run Viewer tests and confirm failure**

  Run: `python -m unittest tests.test_view_go2_amp_terrains tests.test_vis_go2_amp -q`

  Expected: FAIL on the old 30-tile/ten-name assumptions.

- [ ] **Step 3: Adapt the two viewers without adding new UI machinery**

  Preserve the one-time overview console legend and ordinary Isaac Viewer mouse controls. Do not add scene labels, custom overlays, or a narrow-passage-specific evaluator.

- [ ] **Step 4: Run Viewer tests**

  Run: `python -m unittest tests.test_view_go2_amp_terrains tests.test_vis_go2_amp -q`

  Expected: PASS.

- [ ] **Step 5: Commit the Viewer change**

  ```bash
  git add legged_gym/scripts/view_go2_amp_terrains.py legged_gym/scripts/vis_go2_amp.py tests/test_view_go2_amp_terrains.py tests/test_vis_go2_amp.py
  git commit -m "feat: update revised terrain viewers"
  ```

### Task 5: Revise matched AMP/noAMP cases and add vertical-motion evidence

**Files:**
- Modify: `legged_gym/scripts/evaluate_go2_amp_comparison.py`
- Modify: `tests/test_go2_amp_comparison.py`

**Interfaces:**
- Consumes: the seven formal names and evaluation levels `(0,3,6,9)`.
- Produces: the existing 72-case matrix: `7 terrains x 4 levels x 2 forward speeds = 56` plus `2 turning terrains x 4 levels x 2 yaw rates = 16`.
- Produces: `base_lin_vel_z_sq` in `METRIC_FIELDS`, `cases.csv`, and `mean_base_lin_vel_z_sq` in `summary.csv`.

- [ ] **Step 1: Add failing case-matrix and metric tests**

  Assert all seven terrains and only those terrains, levels `{0,3,6,9}`, forward speeds `{0.4,0.8}`, turns only on `flat`/`rough_ground`, identical schedules for both policies, and CSV/summary inclusion of vertical velocity squared.

- [ ] **Step 2: Add a source/runtime test for terminal-step vertical velocity**

  Require accumulation of `base_lin_vel[:,2].square()` under the active mask and replacement from `terminal_diagnostics["base_lin_vel"][:,2]` for terminal steps, matching the existing vx/yaw terminal handling.

- [ ] **Step 3: Run the paired-evaluation tests and confirm failure**

  Run: `python -m unittest tests.test_go2_amp_comparison -q`

  Expected: FAIL on old terrain names/levels and the missing metric.

- [ ] **Step 4: Implement the revised matrix and metric**

  Keep manifest validation, shared AMP discriminator scoring, deterministic seeds, inference-only modules, per-environment WM metrics, CSV names, and PNG set unchanged.

- [ ] **Step 5: Run the paired-evaluation tests**

  Run: `python -m unittest tests.test_go2_amp_comparison -q`

  Expected: PASS.

- [ ] **Step 6: Commit the evaluator change**

  ```bash
  git add legged_gym/scripts/evaluate_go2_amp_comparison.py tests/test_go2_amp_comparison.py
  git commit -m "feat: revise matched terrain evaluation"
  ```

### Task 6: Document the corrected workflow and pre-revision artifact status

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme_go2_amp.py`

**Interfaces:**
- Consumes: the new terrain contract, output root, Viewer commands, and paired evaluator from Tasks 1-5.
- Produces: operator commands that can only create/resume corrected runs in `outputs/go2_amp_v1/` and explicit labeling of `outputs/go2_amp/` as pre-revision engineering evidence.

- [ ] **Step 1: Add failing documentation contract tests**

  Require all seven terrain names, `0..9`, overview levels `0/3/6/9`, the new output root in fresh/resume/Viewer/evaluation/TensorBoard examples, `base_lin_vel_z_sq`, and the phrase that old runs must not be resumed or used as corrected V1 performance. Reject active command examples pointing at `outputs/go2_amp/`.

- [ ] **Step 2: Run the README test and confirm failure**

  Run: `python -m unittest tests.test_readme_go2_amp -q`

  Expected: FAIL because README still documents the old terrain set and namespace.

- [ ] **Step 3: Update README without claiming cloud validation**

  Preserve the validated environment-installation and checkpoint explanations. Mark prior 4096-env, memory, resume, and numerical-stability findings as pre-revision engineering evidence; do not carry their reward, terrain level, or locomotion comparisons forward as corrected results.

- [ ] **Step 4: Run the README test**

  Run: `python -m unittest tests.test_readme_go2_amp -q`

  Expected: PASS.

- [ ] **Step 5: Commit the documentation change**

  ```bash
  git add README.md tests/test_readme_go2_amp.py
  git commit -m "docs: describe corrected terrain workflow"
  ```

### Task 7: Verify local compatibility and prepare cloud acceptance commands

**Files:**
- Modify only if needed for a discovered regression: files already owned by Tasks 1-6
- Do not modify: `docs/MGDP_GO2_AMP_V1_INITIAL_BASELINE.md`

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: a clean implementation diff and a cloud checklist; it does not produce or delete run artifacts locally.

- [ ] **Step 1: Run the complete pure-Python suite**

  Run: `python -m unittest discover -s tests -q`

  Expected: all available tests PASS; only dependency-gated Isaac/PyTorch tests may skip in the local environment.

- [ ] **Step 2: Run static checks**

  Run: `python -m compileall -q legged_gym/legged_gym legged_gym/rl legged_gym/scripts tests`

  Run: `git diff --check`

  Expected: both exit 0.

- [ ] **Step 3: Review the final diff for frozen-system regressions**

  Confirm no changes to AMP/WM/PPO math, observation dimensions, randomization ranges, checkpoint pair format, manifest format, or reward coefficients. Confirm old-output deletion/migration code is absent and the unrelated baseline/figures changes are not staged.

- [ ] **Step 4: Prepare (do not execute locally) the ordered cloud gate**

  Document commands for: full tests; 28-tile overview; representative single-case Viewer including narrow level 9; fresh 64-env AMP/noAMP short runs in the new namespace; strict 10-to-20 resume for both; 4096-env AMP load short run; checkpoint/TensorBoard/NaN/VRAM checks; then fresh paired 1000/2000/5000 training and matched evaluation. Every command must use `outputs/go2_amp_v1/`.

- [ ] **Step 5: Commit any verification-only corrections separately**

  ```bash
  git add <only-files-corrected-during-verification>
  git commit -m "test: complete terrain revision verification"
  ```
