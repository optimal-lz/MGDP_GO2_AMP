# MGDP Go2 AMP

本项目是在 Legged Gym、RSL-RL/MGDP 现有框架上进行的进阶修改：保留 MGDP 的前下视深度感知、对比学习 World Model 和 PPO 运动控制架构，引入 WMP 风格的单判别器 AMP，把已有 Go2 专家动作作为自然步态先验。正式任务只有一个：`go2_amp`。

当前唯一有效的总体规格是 [`docs/superpowers/specs/2026-10-12-mgdp-go2-amp-final-design.md`](docs/superpowers/specs/2026-10-12-mgdp-go2-amp-final-design.md)。更早的 Stage、anchor、AMP ramp 等文档只保留为历史记录，不是正式训练协议。

## 正式设计摘要

- Actor 只使用部署时可获得的带噪本体观测、命令、本体历史 token，以及由带噪深度产生并 `detach` 的 MGDP WM token。
- Critic 使用 Actor 的 45D 基础本体观测，加真实基座线速度、足端高度和局部仿真高度采样；不重复接入 WM token。
- AMP 与 Actor、Critic、WM 分支独立。单一判别器比较策略与专家的干净 30D 状态转移，只通过 AMP reward 影响 PPO return。
- idle 命令精确为 `vx=0, yaw=0`，只使用 task、regularization 和 safety reward，也不写 AMP policy replay。其余均为 locomotion 命令并启用 AMP。
- 专家库只启用前进和左右转向八个文件，前进/转向组权重为 80%/20%。站立、后退、横向及未启用动作仍保留在数据目录，但不进入判别器分布。
- `feet_air_time` 作为弱运动支架保留；`dof_error`、`feet_edge`、`cheat`、`stuck` 作为必要控制/安全约束保留。手工 trot、bound、pace 和 stand-still 风格项关闭。
- 单阶段从十类低难度地形联合训练，不设独立 WM 预训练、task-only 阶段、AMP 系数渐变、地形锚点或难度解锁。
- 每轮顺序固定为：rollout 内 WM 在线更新 → rollout 后 4 次 WM 更新 → GAE → PPO → 20 次判别器更新。第 0 个 rollout 的 AMP contribution 为 0，从下一轮起使用固定系数。
- AMP 和 noAMP 对照除 AMP 模块及奖励外完全一致。noAMP 不构造判别器、专家数据集或 AMP replay。

正式十类地形为：`flat`、`rough_flat`、`smooth_slope`、`rough_slope`、`stairs`、`discrete_obstacles`、`hurdle`、`gap`、`pit`、`narrow_passage`。低矮钻越类地形不在范围内。平地与粗糙平地允许前进和转向，其他结构化地形只训练前进穿越。

## 运行环境

训练需要 Linux、NVIDIA GPU、CUDA、PyTorch、Isaac Gym 和 Warp。参考 MGDP 的 Python 3.8 / PyTorch 1.10 + CUDA 11.3 环境起步，并根据云服务器驱动验证具体组合。Isaac Gym 需从 NVIDIA 包的 `python` 目录单独安装，本仓库不分发它。

在项目根目录安装：

```bash
pip install -r legged_gym/requirement-gpu.txt
pip install -e ./legged_gym
pip install -e ./warp_sensor
export PYTHONPATH="$PWD/legged_gym:$PYTHONPATH"
```

先运行测试：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Windows 当前工作区没有安装 PyTorch、Isaac Gym 和匹配的 CUDA 栈，因此这里只能完成纯 Python、静态与源码编译验证；GPU 训练和 Viewer 验证必须在 RTX 4090 云服务器上执行。

## 统一训练入口

正式入口只有：

```text
legged_gym/scripts/train_go2_amp.py
```

`--mode` 与 `--target_iteration` 必填；默认 `seed=1`、`num_envs=4096`，AMP 默认系数为 `0.01`。`target_iteration` 是绝对目标迭代数，不是“再训练多少轮”，续训时无需人工做减法。

### 64 环境配对集成短跑

先分别确认 AMP/noAMP 能启动、训练、保存严格配对 checkpoint：

```bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 64 --target_iteration 10 --headless
python legged_gym/scripts/train_go2_amp.py --mode no_amp --seed 1 --num_envs 64 --target_iteration 10 --headless
```

对应目录是：

```text
legged_gym/outputs/go2_amp/amp_coef0p01_seed1_env64/
legged_gym/outputs/go2_amp/no_amp_seed1_env64/
```

再分别从刚保存的 manifest 续训到绝对目标 20，验证优化器、WM、AMP 与 RNG 状态均能恢复：

```bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 64 --target_iteration 20 --resume_from legged_gym/outputs/go2_amp/amp_coef0p01_seed1_env64 --headless
python legged_gym/scripts/train_go2_amp.py --mode no_amp --seed 1 --num_envs 64 --target_iteration 20 --resume_from legged_gym/outputs/go2_amp/no_amp_seed1_env64 --headless
```

64 环境只验证集成，不代表 4090 上的正式吞吐、显存占用或控制效果。

### 4096 环境 AMP 全负载短跑

正式 AMP 目录从 10 iterations 起步：

```bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 4096 --target_iteration 10 --headless
```

检查无 OOM、NaN/Inf，WM/PPO/D 指标均产生，且 `checkpoints/` 内生成完整 policy/WM/manifest 三件套。这里不预先增加显存保护分支；若 RTX 4090 24 GB 实测不足，再基于数据调整。

随后在同一正式目录续训到绝对目标 20，检查 TensorBoard step 单调，以及 checkpoint、RNG 和地形等级连续：

```bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 4096 --target_iteration 20 --resume_from legged_gym/outputs/go2_amp/amp_coef0p01_seed1 --headless
```

### 正式 AMP/noAMP 对照

AMP 的 4096 环境短跑已经建立正式目录，因此继续续训到 1000：

```bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 4096 --target_iteration 1000 --resume_from legged_gym/outputs/go2_amp/amp_coef0p01_seed1 --headless
```

noAMP 从零开始训练到同一绝对目标：

```bash
python legged_gym/scripts/train_go2_amp.py --mode no_amp --seed 1 --num_envs 4096 --target_iteration 1000 --headless
```

审阅 1000 轮结果后，分别续训到 2000：

```bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 4096 --target_iteration 2000 --resume_from legged_gym/outputs/go2_amp/amp_coef0p01_seed1 --headless
python legged_gym/scripts/train_go2_amp.py --mode no_amp --seed 1 --num_envs 4096 --target_iteration 2000 --resume_from legged_gym/outputs/go2_amp/no_amp_seed1 --headless
```

通过中期检查后续训到 5000：

```bash
python legged_gym/scripts/train_go2_amp.py --mode amp --seed 1 --num_envs 4096 --target_iteration 5000 --resume_from legged_gym/outputs/go2_amp/amp_coef0p01_seed1 --headless
python legged_gym/scripts/train_go2_amp.py --mode no_amp --seed 1 --num_envs 4096 --target_iteration 5000 --resume_from legged_gym/outputs/go2_amp/no_amp_seed1 --headless
```

若要新建不同 AMP 系数实验，可在首次启动时传入例如 `--amp_coef 0.005`，将得到独立目录 `amp_coef0p005_seed1`。续训必须保持 mode、系数、seed、环境数与完整配置一致；noAMP 不接受 AMP 系数参数。

## 实验目录与严格续训

每个实验目录结构为：

```text
legged_gym/outputs/go2_amp/<run_name>/
├── config.json
├── run.json
├── checkpoints/
│   ├── policy_amp_1000.pt        # noAMP 时为 policy_no_amp_1000.pt
│   ├── wm_1000.pt
│   ├── checkpoint_1000.json
│   └── last.json
├── tensorboard/
└── eval/
    ├── wm/
    ├── comparison/
    └── videos/
```

`checkpoint_1000.json` 记录 mode、AMP 系数、seed、环境数、配置哈希，以及 policy/WM 文件各自的 SHA-256。`last.json` 只包含最新 manifest 指针。加载时会在改变模型前验证兼容性、文件存在性、哈希与迭代号，禁止 policy-only 或错配 WM 的正式续训。

policy checkpoint 包含 PPO 网络与优化器、地形等级、Python/NumPy/Torch CPU/CUDA RNG，以及 AMP 模式下的判别器、优化器、normalizer、replay 和采样 RNG。WM checkpoint 独立保存网络、优化器与统计量，便于解释性分析，但只能与同 manifest 的 policy 配对。

## TensorBoard

```bash
tensorboard --logdir legged_gym/outputs/go2_amp
```

每轮只记录精简核心曲线：task reward、AMP contribution、total reward、PPO 更新指标以及采样/学习耗时；每 10 轮补充 terrain mean level 与 action std。AMP 模式记录判别器指标，noAMP 不产生判别器训练曲线且 AMP contribution 必须恒为 0。TensorBoard 用于诊断，不改变训练流程。

## Isaac Gym Viewer 与 WM 可解释性

Viewer 严格加载一对 policy/WM checkpoint，关闭课程、随机化和训练更新，并允许指定单一地形、等级与命令。平地转向示例：

```bash
python legged_gym/scripts/vis_go2_amp.py --run_dir legged_gym/outputs/go2_amp/amp_coef0p01_seed1 --manifest checkpoint_1000.json --terrain flat --level 5 --vx 0.4 --yaw 0.6 --follow_camera
```

楼梯前进示例：

```bash
python legged_gym/scripts/vis_go2_amp.py --run_dir legged_gym/outputs/go2_amp/amp_coef0p01_seed1 --manifest checkpoint_1000.json --terrain stairs --level 9 --vx 0.5 --yaw 0 --follow_camera
```

结构化地形要求 `yaw=0`。省略 `--manifest` 时使用 `last.json`。Viewer 会显示机器人运动，并在 episode 结束时输出命令、实际速度、AMP reward 和失败原因。

WM 对比图写入：

```text
eval/wm/checkpoint_1000/<terrain>_level_<level>/depth_latest.png
eval/wm/checkpoint_1000/<terrain>_level_<level>/height_latest.png
```

需要保留时间序列图片时添加 `--numbered_snapshots --snapshot_interval 100`；默认只覆盖 `depth_latest.png` 与 `height_latest.png`，避免产生大量冗余文件。

视频录制是可选工具，不是默认验收步骤。安装 FFmpeg 且使用 Isaac Gym Viewer 时可自行决定录制：

```bash
python legged_gym/scripts/vis_go2_amp.py --run_dir legged_gym/outputs/go2_amp/amp_coef0p01_seed1 --terrain stairs --level 9 --vx 0.5 --yaw 0 --record --duration 20 --fps 30 --record_output stairs_level9.mp4
```

视频只允许写入所选 run 的 `eval/videos/` 下。

已有专家兼容性证据 `D:\New Beginning Help\算力自由使用\Video_260929142335.mp4` 已包含当前正式启用的前进和左右转向动作在本项目 Go2 URDF 上的回放，因此不重复做专家库离线回放验收。该视频是本机外部证据，不属于仓库产物。

## AMP/noAMP 固定对照评估

在相同迭代、seed 和配置下执行：

```bash
python legged_gym/scripts/evaluate_go2_amp_comparison.py \
  --amp_manifest legged_gym/outputs/go2_amp/amp_coef0p01_seed1/checkpoints/checkpoint_1000.json \
  --no_amp_manifest legged_gym/outputs/go2_amp/no_amp_seed1/checkpoints/checkpoint_1000.json \
  --episodes_per_case 20
```

评估固定覆盖 72 个 case：十类地形的前进命令，以及 `flat`/`rough_flat` 的转向命令；两个策略使用相同 case、episode seed 和冻结环境。noAMP 策略也使用 AMP run 的同一个冻结判别器与 normalizer 离线打分，使 style score 可直接比较，评估过程不更新任何网络或 replay。

默认只生成以下文件，不生成 Markdown 报告：

```text
eval/comparison/cases.csv
eval/comparison/summary.csv
eval/comparison/completion.png
eval/comparison/tracking.png
eval/comparison/smoothness.png
eval/comparison/style_score.png
```

CSV 保留逐 case 数据与汇总，PNG 用于快速对照完成率、速度追踪、平滑/力矩和 AMP style score。后续根据这些结果与 TensorBoard 曲线共同分析，再决定是否改系数或训练设置。

## 验收边界

本地测试和 64 环境短跑只证明代码集成；4096×10 只证明单卡全负载能够运行。运动控制任务的最终判断必须基于 1000/2000/5000 迭代的配对实验、十类地形 Viewer 观察、WM 图和固定对照 CSV/PNG，不能用某一次短跑或单个 reward 曲线代替。

云服务器首次执行时依次确认：

1. 64 环境 AMP/noAMP 均能保存、严格加载并继续训练。
2. 4096 环境 AMP 运行 10 轮无 OOM、NaN/Inf，policy/WM/D 与 TensorBoard 指标正常。
3. 用同一正式目录续训，checkpoint 迭代单调且 `last.json` 正确指向最新 manifest。
4. Viewer 在 flat、stairs 及其他代表地形上冻结运行，WM 图片持续更新。
5. AMP/noAMP 同迭代对照输出两份 CSV 和四张 PNG，并由人工结合运动效果审阅。

参见 [`SOURCES.md`](SOURCES.md) 了解参考项目与数据来源。
