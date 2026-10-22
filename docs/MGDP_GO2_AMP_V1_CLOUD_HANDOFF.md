# MGDP GO2 AMP V1 云端工作交接

更新时间：2026-10-22

交接分支：`Test`

修订实现基线：`dba7bbc5e10fa21f883d9a765baa705096851fe7`

## 1. 文档用途

本文是新对话接手云服务器测试、正式训练和结果分析时使用的短交接索引。
它不重复设计规格、训练命令或历史实验明细，也不把尚未完成的云端验证写成结论。

单次实验的参数、日志、TensorBoard、CSV、PNG 和 Viewer 观察继续记录在对应
run 产物中。经过确认的阶段结论只有在用户同意后才更新到 V1 基线文档。

## 2. 仓库与工作区

- GitHub：`https://github.com/optimal-lz/MGDP_GO2_AMP`
- 正式分支：`Test`
- 本地项目：`D:\New Beginning Help\LZAMP\MGDP_GO2_AMP`
- 云端项目：`/root/gpufree-data/MGDP_GO2_AMP`
- 修订实现基线：`dba7bbc`

新对话开始时先执行只读检查：

```bash
git status --short --branch
git rev-parse --short HEAD
git remote -v
```

云服务器应先执行：

```bash
git pull --ff-only origin Test
git rev-parse --short HEAD
git merge-base --is-ancestor dba7bbc HEAD
```

最后一条命令必须返回 0，证明当前 HEAD 包含修订实现基线；交接文档本身提交后，
HEAD 可以是 `dba7bbc` 的后继提交。若不包含该基线，先处理代码版本，不启动训练。

本地交接时仍有用户自己的未提交内容，必须保留，不得擅自覆盖、清理或加入其他
提交：

- `docs/MGDP_GO2_AMP_V1_INITIAL_BASELINE.md`
- `docs/figures/`

## 3. 必读文档与覆盖关系

按以下顺序建立当前工作上下文：

1. 本文；
2. `docs/superpowers/specs/2026-10-22-mgdp-go2-amp-v1-terrain-revision.md`；
3. `docs/superpowers/specs/2026-10-12-mgdp-go2-amp-final-design.md`；
4. `docs/MGDP_GO2_AMP_V1_INITIAL_BASELINE.md`；
5. `README.md`。

10 月 22 日地形修订只替代旧最终设计中的正式地形集合、地形几何、课程相关
command/reward 条件、Viewer/验收矩阵和正式输出根目录。以下内容继续沿用 10 月
12 日冻结设计：

- Actor、Critic、AMP 判别器与 World Model 的观测边界；
- AMP、WM、PPO 数学与更新顺序；
- 奖励主体及冻结系数；
- 随机化和噪声范围；
- paired policy/WM checkpoint、manifest、哈希和严格续训协议；
- AMP/noAMP 正式配对训练流程。

旧最终设计和当前 V1 基线中仍有“十类地形、20 个 level、
`outputs/go2_amp/`”等修订前文字。涉及地形和运行目录时，以 10 月 22 日修订及
当前 README 为准。不要把这些旧文字重新实现到代码中。

## 4. 当前正式地形与输出边界

修订后的正式地形只有七类：

- `flat`：15%；
- `rough_ground`：15%；
- `slope_course`：20%；
- `stairs_course`：20%；
- `low_obstacle_course`：15%；
- `gap_course`：10%；
- `narrow_passage`：5%。

课程等级固定为 level `0..9`。`flat` 和 `rough_ground` 允许前进及转向；其他
五类结构化地形只训练前进，固定 `yaw=0`。结构化地形从前置平地出生；坡面和
楼梯的顶部平台与相邻最高表面等高，不允许出现修订前的中央坑。

所有修订后实验统一写入：

```text
legged_gym/outputs/go2_amp_v1/
```

旧目录：

```text
legged_gym/outputs/go2_amp/
```

只保留为旧地形实现下的工程诊断证据。旧 run 不得续训、迁移、覆盖或作为新地形
的性能结论。修订后 AMP/noAMP 必须从新目录全新开始。

## 5. 当前代码与本地验证状态

修订实现基线已经包含：

- 七类定向地形生成器和 10 级课程；
- 正式出生点、课程、命令和地形条件奖励适配；
- 新输出根目录隔离；
- 28 地块训练地形总览 Viewer；
- 单 case 策略 Viewer；
- 新地形下的 72-case AMP/noAMP 固定配对评估；
- 对应回归、几何、CLI、Viewer、resume 和 checkpoint 测试。

在 Windows 工作区、修订实现基线上最后确认：

```text
python -m unittest discover -s tests -q
Ran 147 tests: OK (skipped=13)

python -m compileall -q legged_gym/legged_gym legged_gym/rl legged_gym/scripts tests
compileall_exit=0

git diff --check
diff_check_exit=0
```

13 项跳过项明确依赖训练机上的 PyTorch/Isaac Gym 张量环境。本地通过不能替代
云端 GPU、Isaac Gym、Viewer、4096-env 显存或运动效果验证。

## 6. 最近确认的云服务器环境

以下是地形修订前最后确认可工作的运行环境，用于复用，不授权重装或升级：

- 单卡 NVIDIA GeForce RTX 4090 24 GB；
- Conda 环境：`lzamp`；
- Python `3.8.20`；
- PyTorch `2.4.1+cu121`，CUDA 可用；
- Isaac Gym `1.0rc4`，位于仓库的 `isaacgym/python`；
- `legged_gym`、`warp_sensor` 和 `isaacgym` 使用 editable install；
- `pip check` 曾报告 `No broken requirements found`。

优先复用该环境。不要因为 Gym、`torch.meshgrid` 或受信任 checkpoint 的
`torch.load(weights_only=False)` warning 擅自升级 PyTorch、CUDA、Isaac Gym
或更换 Gymnasium。先判断 warning 是否实际导致失败。

## 7. 旧云端结果的有效边界

旧地形版本曾完成 64-env 集成和严格续训、4096-env 满负载，以及 AMP/noAMP
1000/2000/5000 iteration 配对训练。这些结果仍可证明当时以下工程链路可运行：

- RTX 4090 上的 Isaac Gym/PyTorch/Warp 基础环境；
- 4096-env PhysX 容量和约 19.4 GiB 峰值显存量级；
- paired checkpoint、manifest、TensorBoard 和严格续训机制；
- 长时间训练未出现 OOM、CUDA error 或 NaN/Inf。

但旧 run 使用了已判定错误的地形，因此不能证明修订后地形的训练效果，也不能
用于新正式训练续接。V1 基线中相关数字必须按“修订前工程证据”理解。

## 8. 修订后云端执行顺序

必须按顺序推进，并在每一步根据证据区分环境问题、代码问题和训练效果问题：

1. 拉取 `Test` 并确认 HEAD 包含实现基线 `dba7bbc`；
2. 在 `lzamp` 环境运行完整测试，确认 13 项训练机测试实际执行而不是 skip；
3. 运行 28 地块地形总览，检查七类地形、level 0/3/6/9、出生区、坡顶平台、
   楼梯平台、沟槽和通道是否与设计一致；
4. 运行代表性单 case Viewer，至少覆盖 `flat`、`stairs_course`、
   `gap_course` 和 `narrow_passage` level 9；
5. 在 `outputs/go2_amp_v1/` 从零运行 64-env AMP/noAMP × 10；
6. 两组分别严格续训 10→20，检查 iteration、optimizer、WM、AMP、RNG、课程
   和 TensorBoard step 连续；
7. 从零运行 4096-env AMP × 10，再严格续训 10→20，检查 PhysX、OOM、
   CUDA、NaN/Inf、吞吐和显存；
8. 通过阶段验收后，运行 AMP/noAMP 1000、2000、5000 absolute target 配对训练；
9. 使用同迭代 checkpoint 运行 Viewer、WM 图片和固定 72-case comparison；
10. 综合 TensorBoard、checkpoint、CSV/PNG、显存和实际运动形成结果分析。

每一个阶段必须使用 README 中的当前命令和新输出目录。不要从本文复制或推断
过时命令。

## 9. 证据分类与变更规则

收到命令输出、错误日志、TensorBoard、CSV、PNG 或 Viewer 观察后，先分类：

- 环境问题：依赖、动态库、CUDA、显存、Viewer/显示、文件路径或权限；
- 代码问题：可复现异常、合同违反、checkpoint/manifest 不一致或数值错误；
- 训练效果问题：代码正常运行，但跟踪、稳定性、地形通过率或运动风格不足。

只有云服务器实测证据明确指向设计问题时，才分析并提出修正。未经确认不得重新
设计 AMP、奖励、观测边界、地形集合、随机化范围或训练流程。

特别保留以下观察边界：

- `lin_vel_z=-1.0` 继续冻结；配对评估记录 `base_lin_vel_z_sq`。只有修订后
  Viewer 和指标共同证明楼梯、低障碍或 gap 的必要垂向运动受到压制，才提出
  具名地形调整；
- narrow passage level 9 只需总览和普通单 case Viewer，不建立专项评分器；
- 64-env 只证明集成，4096×10 只证明单卡满负载可运行；
- 训练 reward 或单轮汇总不能替代固定配对评估和 Viewer 运动判断。

## 10. 新对话的首轮动作

新对话接手后应先完成以下只读动作并报告，不立即改代码：

1. 按第 3 节顺序读取文档；
2. 检查 Git 分支、HEAD、远端和未提交文件；
3. 确认正式任务仍只有 `go2_amp`，正式训练入口仍为
   `legged_gym/scripts/train_go2_amp.py`；
4. 确认新 run 根目录为 `legged_gym/outputs/go2_amp_v1/`；
5. 根据用户随后提供的第一份云端证据继续第 8 节流程。

不要仅凭交接文档宣称云端验证已经完成，也不要未经用户同意更新
`docs/MGDP_GO2_AMP_V1_INITIAL_BASELINE.md`。
