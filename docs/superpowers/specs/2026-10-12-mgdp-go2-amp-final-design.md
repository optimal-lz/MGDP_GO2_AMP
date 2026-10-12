# MGDP Go2 AMP 最终设计规格

日期：2026-10-12
状态：已冻结，等待实施计划
适用仓库：`D:\New Beginning Help\LZAMP\MGDP_GO2_AMP`

## 1. 文档地位与目标

本文是当前项目唯一有效的总体设计规格。以下旧规格仅作历史记录，不得覆盖本文：

- `docs/superpowers/specs/2026-09-21-mgdp-go2-amp-design.md`
- `docs/superpowers/specs/2026-10-09-mgdp-go2-amp-training-redesign.md`

验收目标是：Go2 使用 MGDP 式视觉感知，在 MGDP Stage1 代表地形与 WMP 主要地形的合理并集上追踪 WMP 量级速度命令，同时由已有专家动作约束出自然步态。项目只服务于感知运动控制、地形穿越与自然步态，不因照搬参考项目而增加无关任务。

系统由两个互补部分组成：

1. 保留 MGDP 感知运动控制：前下视深度、本体历史编码、对比学习与高度预测 World Model、PPO。
2. 采用 WMP 思路正确实现 AMP：PPO 策略是生成器；单一判别器比较策略与专家的同构状态转移；输出仅形成风格奖励。AMP 是学习式正则化，不代替任务奖励。

正式方案是单阶段联合训练。旧 Stage1/Stage2 切换、AMP ramp、地形锚点、难度解锁和双策略回放均不属于最终设计。

## 2. 总体数据流和信息隔离

```text
带噪深度 -> MGDP World Model -> detach WM token --\
带噪本体历史 -> proprio token ---------------------+-> PPO Actor -> action
部署可用当前本体观测与命令 -------------------------/

Actor 基础观测 + 干净特权信息 -----------------------> PPO Critic

干净 30D 策略状态转移 --\
干净 30D 专家状态转移 ---+-> 单一 AMP 判别器 -> AMP reward -> PPO return
```

- PPO 不得反向传播到 WM；Actor 的 WM token 必须 `detach`。
- 判别器输出不进入 Actor、Critic 或 WM，只通过 reward 影响 PPO return。
- WM 输入带噪、AMP 状态干净，二者用途不同且支路独立。
- 专家数据不初始化环境，也不直接生成策略动作。

## 3. 专家库

### 3.1 启用范围

只启用八个文件：

- 前进：`go2_forward.txt`、`go2_forward_fast.txt`
- 左转：`go2_turn_left.txt`、`go2_turn_left_fast.txt`、`go2_turn_left_faster.txt`
- 右转：`go2_turn_right.txt`、`go2_turn_right_fast.txt`、`go2_turn_right_faster.txt`

保留但不启用：`go2_stance.txt`、`go2_forward_faster.txt`、全部后退和横向文件。站立样本不进入判别器；idle 只需稳定，不需要模仿“专家站立”。后退和横向数据保留文件，但不让无关状态转移进入当前专家分布。

### 3.2 采样

- 先按组抽样：前进 80%，转向 20%；组内沿用现有 `MotionWeight`，未配置时等概率。
- 在线采样，不预加载全部 transition。
- 专家帧间隔 `0.02 s`，与 50 Hz policy transition 一一对应。
- `D:\New Beginning Help\算力自由使用\Video_260929142335.mp4` 已证明专家动作与当前 Go2 URDF 兼容，不重复专家回放验证。

## 4. 命令与 idle gating

- `vx ∈ [0, 0.8] m/s`，`vy = 0`，`yaw ∈ [-1, 1] rad/s`。
- 采样死区：`vx <= 0.2 -> 0`；`abs(yaw) <= 0.2 -> 0`。
- 死区后精确分类：
  - idle：`vx == 0 and yaw == 0`。只用 task、regularization、safety reward；不计算 AMP reward，不写 policy replay，不更新 AMP normalizer。
  - locomotion：其余命令。使用 task、regularization、safety 与 AMP reward。
- 不增加第二套“近零速阈值”。每 10 秒重采样命令，不设命令课程。
- `flat`、`rough_flat` 允许前进、原地转向、前进加转向；其余八类结构化地形只采样前进并强制 `yaw=0`。

## 5. AMP

### 5.1 判别器

全系统只有一个无条件判别器、一个 policy replay、一个 AMP normalizer，不按命令、专家组或地形拆分。

干净 30D AMP 状态的顺序固定为：

```text
[12D joint position,
  3D base-frame linear velocity,
  3D base-frame angular velocity,
 12D joint velocity]
```

判别器输入为 `(s_t, s_t+1)` 拼接的 60D transition，网络为 `60 -> 1024 ReLU -> 512 ReLU -> 1`。使用 LS-GAN：专家目标 `+1`，策略目标 `-1`；只对专家施加系数 10、目标梯度 0 的 gradient penalty。

### 5.2 奖励

```text
r_amp_raw = max(0, 1 - 0.25 * (D(s_t, s_t+1) - 1)^2)
r_base    = clamp(r_task + r_regularization + r_safety, min=0)
r_total   = r_base + lambda_amp * r_amp_raw + r_termination
```

默认冻结 `lambda_amp=0.01`，唯一配置源应为 `legged_gym/legged_gym/envs/go2_amp/config.py` 中的 `amp.reward_coef`。CLI 的 `--amp_coef` 只用于新 AMP 实验；续训不匹配必须拒绝。noAMP 必须用 `--mode no_amp`，不能用系数 0 冒充。训练中不做系数 ramp 或中途修改。

### 5.3 replay 与时序

- 容量 `2 * num_envs * num_steps`，默认 4096×24 时为 196608。
- 第一个 rollout 收集有效 locomotion transition，但因 D 未训练，其 AMP contribution 为 0；rollout 后开始 D 更新。
- 每轮 rollout 后进行 20 次 D 更新。
- rollout 内冻结 D 与 AMP normalizer；rollout 后只用有效 locomotion policy transition 和启用专家更新。
- 排除 idle transition 和跨命令重采样边界的 transition。
- 终止 transition 使用 reset 前终止状态，绝不使用 reset 后状态。
- AMP clean state 单独构造，不能复用 Actor/WM 带噪输入。

## 6. MGDP World Model 与单阶段训练

保留 MGDP 的对比学习与视觉高度预测 WM，不替换成 WMP/Dreamer 动态模型。

- 深度刷新间隔 5 个策略步；每次刷新执行一次在线 WM 更新。
- 每个 rollout 后额外执行 4 次 WM 更新。
- Actor 使用带噪深度产生且已 `detach` 的 WM token。
- WM 以带噪深度为输入、以干净深度/高度为监督。

每轮固定顺序：

```text
rollout 中深度刷新与在线 WM 更新
-> Actor action
-> 冻结 D 计算 AMP reward 并存储有效 transition
-> rollout 后 4 次额外 WM 更新
-> GAE
-> PPO
-> D × 20
```

从第 0 轮开始单阶段联合训练：不做 task-only、AMP ramp、flat-only、独立 WM 预训练或冻结/解冻阶段；十类地形从低难度同时出现。rollout 0 只有 task/regularization/safety，结束后训练 D；rollout 1 起使用固定 AMP 系数。WM 单独保存以便解释，但必须由 manifest 与同迭代 policy 严格配对。

## 7. Actor、Critic、D 的观测边界

Actor 只读部署可用信息：

- 带噪角速度、重力投影
- 速度命令
- 带噪关节位置、关节速度、上一动作
- 4 步本体历史 token
- 带噪深度产生、已 `detach` 的 MGDP WM token

Actor 不得读取真实基座线速度、干净高度图、接触力、terrain id/level 或其他仿真状态。

Critic 特权观测严格为 239D：

- Actor 的 45D 基础本体观测，不含 WM token
- 干净真实基座线速度 3D
- 足端高度 4D
- 局部仿真高度采样 187D

Critic 不接入 WM token、D 输出、摩擦、延迟、接触、terrain id/level 或额外 WMP 特权量。D 只读取第 5 节的干净 30D transition，与 Actor/Critic 观测完全独立。

## 8. 控制与 PPO 基线

- 位置 PD：`q_target = q_default + 0.5 * action`；`Kp=30`，`Kd=0.8`。
- sim dt `0.005 s`，decimation 4，policy 50 Hz。
- Actor/Critic MLP `[512,256,128]`，ELU，初始 std 1。
- proprio history 4，encoder `[256,128,7]`。
- rollout 24。
- PPO：epochs 5、minibatches 4、adaptive LR `1e-3`、clip `0.2`、entropy `0.01`、gamma `0.99`、GAE lambda `0.95`、desired KL `0.01`、grad norm `1`。

这些参数是正式基线，只有实验给出证据后才通过新设计修改。

## 9. 地形与课程

正式任务恰好十类：

| terrain | proportion |
| --- | ---: |
| `flat` | 10% |
| `rough_flat` | 10% |
| `smooth_slope` | 10% |
| `rough_slope` | 10% |
| `stairs` | 15% |
| `discrete_obstacles` | 10% |
| `hurdle` | 10% |
| `gap` | 10% |
| `pit` | 10% |
| `narrow_passage` | 5% |

不加入低矮通道、匍匐或强迫下蹲地形。narrow passage 限为 5%；gap/pit 使用条件安全奖励；其余地形符合正常足式运动目标。

统一几何难度：`num_rows=20`，`d=level/19`；各类地形仅把同一 d 映射到自身几何参数。十类从训练开始都存在，环境从低难度开始；每个 env 的 terrain type 固定，只改变 row/level。

课程只在 `vx>0.2` 的前进命令上更新，依据目标前进方向投影进度：成功升级，早期失败/倒地降级。idle 和纯转向不改变 level。超过最高难度后沿用 MGDP 行为，随机返回较低 level。无 anchor env、阶段解锁或独立地形训练阶段。

## 10. 奖励

| term | coefficient | condition |
| --- | ---: | --- |
| `tracking_lin_vel` | +1.5 | 相关命令 |
| `tracking_ang_vel` | +0.5 | yaw tracking |
| `feet_air_time` | +0.5 | 仅 `vx>0.2`，单次上限 0.75；idle/纯转向关闭 |
| `lin_vel_z` | -1 | regularization |
| `ang_vel_xy` | -0.05 | regularization |
| `orientation` | -0.1 | regularization |
| `torques` | -1e-4 | regularization |
| `dof_acc` | -2.5e-7 | regularization |
| `action_rate` | -0.03 | smoothness |
| `collision` | -1 | non-foot collision |
| `feet_stumble` | -0.1 | stumble |
| `dof_pos_limits` | -10 | safety |
| `torque_limits` | -0.01 | safety |
| `dof_error` | -0.04 | idle only |
| `feet_edge` | -1 | gap/pit 且 `level>3` |
| `cheat` | -1 | 结构化直行地形且 `abs(heading_error)>1` |
| `stuck` | -1 | `vx>0.2` 且实际基座前向进度 `<0.1`；不终止 |

不加入 trot/bound/pace 显式步态模板。启用 `only_positive_rewards=true`，但只对 task+regularization+safety 合计截断；AMP 在其后加入，termination reward 最后加入。

## 11. 终止与 reset

唯一失败终止为 base contact force `>1 N`。timeout 是正常 truncation 并 bootstrap；`termination_reward=0`。头、大腿、小腿接触只罚不终止。首版不增加头部、垂直速度、倾角、速度或机身高度终止。

reset：

- 无 RSI/reference-state initialization；专家状态绝不初始化环境。
- `q=q_default * Uniform(0.9,1.1)`；qdot/root velocity 为 0。
- upright，出生高度为局部地面 `+0.45 m`。
- flat/rough_flat 出生 xy 均为 `±0.2 m`；结构化地形安全起点为 `x±0.2 m, y=0`。
- 重置 proprio/depth/WM history，并把 AMP previous state 设为 reset 后初始状态。
- 绝不产生 terminal-to-reset AMP transition。

## 12. 随机化与噪声

Domain randomization 从训练开始启用：

- friction `[0.5,1.25]`
- action latency `[0,0.02] s`
- 每 15 s push，最大 xy velocity 1

首版不随机 restitution、base/link mass、COM、PD gains、motor strength、motor/joint offsets；无随机化课程或地形例外。

Proprio noise：dof_pos 0.01、dof_vel 1.5、lin_vel 0.1（只给需要的训练支路，不能泄露真实线速度给 Actor）、ang_vel 0.2、projected gravity 0.05。

Depth：16×16、history 2、interval 5、near 0、far 5 并归一化；Gaussian 0.03 m、dropout 0.10；camera translation jitter ±0.02 m、pitch ±2°。不对 privileged/height supervision 加噪，不做 noise curriculum、额外 frame drop 或额外 latency。WM 用带噪输入和干净监督；AMP 用干净状态。

## 13. 训练规模与对照

- 正式使用 4096 env，保持 MGDP/WMP 量级 batch 与地形多样性；每轮 4096×24=98304 transitions。
- 不预加 microbatch、gradient accumulation 或自动降 env 的显存保护；真实 OOM 后再诊断。
- AMP/noAMP 从头独立训练，默认 seed 1；除 AMP 路径外配置相同。
- 环境/策略与 AMP 专家/D 使用独立 RNG streams，避免 AMP 抽样扰动环境/策略随机序列。
- 旧 64-env 实验仅为历史，不能代表新架构。

顺序：

1. AMP/noAMP 各做 64 env × 10 iterations 功能短跑。
2. AMP 做 4096 env × 10 iterations 满负载短跑。
3. 正式先到 absolute target iteration 1000。
4. 根据证据人工决定是否分别续训至 2000、5000；两组保持相同 target。
5. 每 1000 iterations 及最终 iteration 保存。

不自动早停、续训、准入或调参。可先跑 AMP 再跑 noAMP，但比较预算必须相同。

## 14. 正式任务和统一入口

最终只有项目任务 `go2_amp`，配置类为 `Go2AmpCfg`、`Go2AmpTrainCfg`。MGDP 原始 `random_dog_stage1/2` 保持不动。项目旧 `go2_amp_stage1/2` 注册、脚本、stage switching、AMP ramp、anchor、easy/hard gate、stratified/double replay 应由正式实现移除。不要另建 `amp_v2` 平行实现。

正式脚本：

- `legged_gym/scripts/train_go2_amp.py`
- `legged_gym/scripts/vis_go2_amp.py`
- `legged_gym/scripts/evaluate_go2_amp_comparison.py`

训练 CLI：

- `--mode amp|no_amp`
- `--seed`，默认 1
- `--num_envs`，默认 4096
- `--target_iteration`，绝对目标
- `--resume_from`
- `--amp_coef`，只供 AMP 新实验

不公开 `--task`、`--algo`、`--amp_policy_only`。wrapper 读取当前 iteration，运行 `target-current`；target 不大于 current 时报错。noAMP 完全不构造/训练 D，不创建 replay，不算 AMP reward。

## 15. 输出、保存与恢复

根目录：`legged_gym/outputs/go2_amp/`。运行目录如 `amp_coef0p01_seed1/`、`no_amp_seed1/`。同名目录不能从零覆盖；续训必须显式 `--resume_from`。

```text
config.json
run.json
checkpoints/
  policy_amp_1000.pt       # noAMP 为 policy_no_amp_1000.pt
  wm_1000.pt
  checkpoint_1000.json
  last.json
tensorboard/
eval/
  wm/
  comparison/
  videos/
```

`config.json` 保存完整生效配置；`run.json` 保存命令、Git commit、Python/PyTorch/CUDA/Isaac Gym 版本、实际专家文件。

policy checkpoint 包含 Actor/Critic、PPO optimizer、terrain levels、iteration、RNG；AMP 模式另含 D、D optimizer、AMP normalizer/replay。WM checkpoint 含 WM 网络、optimizer、depth stats、iteration。manifest 记录文件名、配置哈希、mode、coef、seed、env count、iteration 与一致性检查；`last.json` 只指向最近完整 pair，不复制大型 `last.pt`。

恢复必须严格配对 policy、WM、manifest。mode/coef/seed/env count/关键配置不匹配则拒绝。不支持旧 checkpoint 正式全量续训，也不把 policy-only warm start 纳入正式流程。

## 16. 可视化与对照评估

### 16.1 Viewer 和可选视频

`vis_go2_amp.py` 加载配对 checkpoint，支持 terrain、level、vx、yaw、env id、follow camera。默认直接打开 Isaac Gym Viewer：

- deterministic Actor，所有网络冻结
- noise/domain randomization/push 关闭
- 固定 seed、少量 env
- live depth 与 WM height panels
- episode 结束打印 command/actual velocity、terrain/level、AMP reward、failure reason

仅传 `--record` 时检查 FFmpeg，并接受 output/duration/fps；默认写入 `<run>/eval/videos/`。训练和普通 Viewer 不自动录制。

WM 解释图保留，默认覆盖 latest：

```text
<run>/eval/wm/<checkpoint>/<terrain>_level_<n>/depth_latest.png
<run>/eval/wm/<checkpoint>/<terrain>_level_<n>/height_latest.png
```

可选 numbered snapshot，但非默认。

### 16.2 AMP/noAMP comparison

`evaluate_go2_amp_comparison.py` 是训练后的独立 headless 工具，不是训练 hook。它读取两组 paired manifests，使用相同 terrain/level/command/seed/episodes/physics。

默认矩阵：十类 terrain；levels 0/9/19；vx 0.4/0.8；flat/rough_flat 另含 turn cases；每 case 20 episodes。

指标：completion/progress、vx/yaw RMSE、base-contact failure、action rate、torque、episode length、WM errors、AMP style score。两种 policy 的 style score 都使用 AMP checkpoint 中同一个冻结 D+normalizer，仅作诊断。

只输出：

- `summary.csv`
- `cases.csv`
- `completion.png`
- `tracking.png`
- `smoothness.png`
- `style_score.png`

不生成 Markdown 报告、视频或自动 pass/fail/调参。结果由用户和助手后续共同分析。

## 17. 轻量 TensorBoard

不记 per-step、histogram、images/videos、大 tensor 或完整 reward breakdown，也不为日志增加 forward/simulation。

每 iteration 记录约 25 个聚合 scalar：

- task、regularization、AMP contribution、total reward
- vx/yaw error、episode length、base-contact failure、mean terrain level
- AMP D policy/expert、D loss、GP、valid transition count、replay fill；noAMP 只记 AMP contribution=0，不创建 D train tags
- WM total/depth/height/contrastive losses
- PPO surrogate/value/entropy/KL/action std
- iteration time

每 10 iterations 记录：per-terrain mean level/completion/base-contact/stuck，forward/turn expert D outputs，simulation/WM/PPO/D timing。复用已有 tensor，每 iteration 最多一次必要 GPU-to-CPU 同步并缓冲写入。

短跑验证 event 存在、resume iteration 单调、值有限、mode tags 正确。TensorBoard 不设性能阈值，不触发停止、准入或参数变化。

## 18. 验证与测试

CPU/static tests 覆盖：

- Actor/Critic/AMP 维度与信息边界
- 30D 顺序、60D transition
- idle gating 与命令边界
- AMP reward 公式与 clamp 顺序
- LS-GAN/GP
- 专家范围、80/20 抽样与数据有效性
- 十地形创建、比例、课程
- terminal/reset transition
- checkpoint pair、manifest、absolute target iteration
- mode/coef/seed/env/config mismatch
- 统一 CLI

断言旧 Stage1/Stage2、ramp、anchor、double replay 的测试必须删除或重写。

GPU smoke 验证有限数值、固定更新顺序、noAMP 不创建 AMP 路径、pair save/restore 后 iteration/terrain/optimizer/normalizer/replay/RNG 连续，以及 4096 env 在 RTX 4090 24 GB 上可运行；若 OOM 再按实测峰值诊断。

性能不写成单元测试阈值。训练效果由 Viewer、可选视频、WM PNG、comparison CSV/PNG、轻量 TensorBoard 综合判断；不重复专家回放。

## 19. 实现边界

实现优先保证：

1. 指定地形上的感知运动控制与命令追踪。
2. AMP reward 来自同构专家/策略 transition，并真实进入 PPO return。
3. Actor 可部署、Critic 特权有限、D 完全独立。
4. AMP/noAMP 可复现、可续训、可解释。
5. 入口与产物简洁，不保留项目专用的平行旧流程。

首版明确不做：低矮匍匐地形、站立/后退/横向专家、多个/条件 D、RSI、显式步态模板、分阶段训练、自动调参/准入、训练时自动视频/报告、无关随机化与额外终止条件。

本文确认后，下一步只编写逐文件实施计划；实施计划获确认前不修改训练代码。
