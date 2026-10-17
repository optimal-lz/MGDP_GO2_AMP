# MGDP Go2 AMP V1 初版基线

日期：2026-10-14  
代码基线：Git `8d653e2`  
状态：2026-10-17 运行时修正版已完成本地静态/CPU 回归；Linux RTX 4090 + Isaac Gym smoke 待执行

## 1. 文档用途与优先级

本文记录新项目 V1 初版实际交付了什么、已经验证了什么，以及训练机上还要验证什么。它不是另一份设计规格。

文档优先级如下：

1. `docs/superpowers/specs/2026-10-12-mgdp-go2-amp-final-design.md`：规定系统应有的行为。
2. 本文：记录 V1 代码基线的实际状态与验证边界。
3. `docs/superpowers/plans/2026-10-16-mgdp-go2-amp-runtime-corrections.md`：记录初版审查后的修正实施。
4. `docs/superpowers/plans/2026-10-12-mgdp-go2-amp-implementation.md`：保留历史实施过程。

若三者出现冲突，先按最终设计判断正确行为，再以本文确认当前代码是否已经落地；实施计划不覆盖后来确认的设计或审查修正。

## 2. V1 正式范围

正式任务和训练入口只有 `go2_amp` 与 `legged_gym/scripts/train_go2_amp.py`。目标是在十类冻结地形上完成 Go2 感知运动控制、速度命令追踪和自然步态学习。

V1 保留并实际依赖的 MGDP 实现包括：

- 前下视带噪深度输入及 Warp camera 数据流；
- 本体历史编码 token；
- 对比学习、深度重建和视觉高度预测 World Model；
- Map encoder/decoder，以及由 PPO 训练的 Actor 侧 64D GRU memory；
- Actor-Critic、PPO、rollout storage 和 Legged Gym/Isaac Gym 环境基础。

这些感知与优化组件是新项目架构的一部分，不因放弃旧 MGDP 任务结果兼容而删除。旧 `random_dog_stage1/2` 注册目前仍存在，但不是 V1 的正式入口、验收对象或兼容目标；只有在确认不被 `go2_amp` 依赖后才能清理。

## 3. 已落地的正式行为

- Actor 只使用部署可用基础观测、本体历史 token 和从带噪深度得到且已 `detach` 的 WM token。
- WM 只负责 CNN/map 感知与监督；Actor 侧 64D GRU 由 PPO 所有，只在每 5 步真实深度刷新时更新，done 时按环境清零。
- Critic 使用 45D 基础观测与 194D 干净特权后缀，不读取 WM token。
- AMP 使用单一无条件判别器和干净 30D 连续状态转移；只启用前进、左右转向八个专家文件。
- idle transition 不计算 AMP reward、不进入 policy replay、不更新 AMP normalizer；locomotion 使用固定 `lambda_amp=0.01`。
- task/regularization/safety 先合计并做正值截断，AMP contribution 随后加入，termination reward 最后加入。
- `terrain_adaptive_reward=false`；不继承旧 MGDP 按旧地形编号缩放奖励的行为。
- 十类地形从 level 0 同时训练，仅前进命令更新课程；结构化地形只使用前进命令。
- 训练随机化只允许 friction、action latency、push。action latency 与 motor strength/offset 完全解耦；未批准的随机化在构造环境时拒绝。
- WM、PPO、AMP 采用冻结的单阶段更新顺序；noAMP 完全不构造 AMP 训练路径。
- policy 与 WM 分文件保存并由 manifest 严格配对；恢复覆盖 PPO 当前学习率、地形派生状态、AMP 私有状态和 RNG。
- Viewer 默认使用 Isaac Gym 实时可视化；视频录制可选。AMP/noAMP 对照只产生约定的 CSV 与 PNG。
- TensorBoard 使用冻结的轻量 core/detail scalar 集，不增加额外网络 forward。
- 15 秒 push 在策略步长 0.02 秒下严格对应 750 步；Critic 高度特权与 WM 高度监督显式不加噪。

## 4. 实施后审查修正

### `8104e96` — 运行时与冻结设计对齐

- 显式关闭 link mass、restitution、COM、gain、motor 等非批准随机化，并增加动态允许列表校验。
- 去除 action latency 对 motor strength/offset 的隐式副作用。
- 将 camera translation/pitch jitter 接入 Warp camera；评估时关闭 proprio、depth、mount 和 domain stochasticity。
- 补齐 paired resume 的 PPO adaptive learning rate、terrain origin/class 派生状态。
- 补齐 task/regularization、tracking、terrain、WM、PPO 和 AMP TensorBoard 指标。
- 正式 CLI 统一为 `--amp_coef`。

### `8d653e2` — 移除旧地形奖励语义

- 在 `Go2AmpCfg.rewards` 中显式设置 `terrain_adaptive_reward=false`。
- `collision`、`lin_vel_z`、`orientation` 不再被旧 MGDP 的 `env_class == 5/6` 规则缩放。
- Go2 明确定义的 `feet_edge`、`cheat`、`stuck` 等条件奖励保持不变。

### 2026-10-17 — 运行时一致性修正版

- `1c01ed8`：完整配置先归一化为 JSON-native 数据再用于 metadata、hash 和 resume 比较；push 改用解析后的 750 策略步间隔；关闭高度与特权高度噪声。
- `254592e`：环境 WM 不再拥有无梯度 GRU；原始 64D CNN token 在 WM 边界 `detach`，Actor GRU 进入 PPO optimizer，并使用保持完整时间顺序的 recurrent rollout。
- `91bdd83`：异步 reset 清除终止帧视觉缓存并重算基座/高度派生观测；paired resume 在 policy 与 WM 都加载后，从恢复的地形层级开始新 episode，并做一次 inference-only 视觉刷新。
- `0d126ec`：Viewer/对照首动作前统一强制视觉刷新并处理 GRU done；对照 WM 指标按环境、仅按刷新帧统计；TensorBoard 汇总一轮内全部 WM 更新。

该修正版改变了 policy 网络参数集合、WM checkpoint 内容和完整配置哈希。`254592e` 之前产生的 checkpoint 不属于正式可续训格式，不能混入修正后的 run；应从修正版新建实验目录开始训练。

## 5. 已完成验证

在当前 Windows 工作区完成：

- `python -m unittest discover -s tests -p 'test_*.py' -v`：109 项中 96 项通过，13 项按声明跳过；
- `python -m compileall -q legged_gym/legged_gym legged_gym/rl legged_gym/scripts tests`：通过；
- `git diff --check`：通过；
- 工作区在 V1 修正提交后保持干净。

跳过项依赖 PyTorch/Isaac Gym 训练环境，因此上述结果不构成运动效果、显存占用或 4096 环境吞吐量证明。

## 6. 训练机待验证项

按 README 的正式命令在 Linux RTX 4090 24 GB 环境依次验证：

1. 64-env AMP 与 noAMP 短跑，确认数值有限、更新顺序和 mode tags 正确。
2. 确认本地跳过的 13 项 PyTorch 张量测试在训练环境实际执行通过，特别是 GRU 梯度、刷新门控和连续 recurrent minibatch。
3. paired checkpoint 保存与续训，确认 iteration、PPO/WM optimizer、terrain、AMP replay/normalizer 与 RNG 连续；续训从恢复层级上的新 episode 开始，不宣称物理中途续接。
4. Isaac Gym Viewer 的 flat 和结构化地形案例，以及 WM depth/height 面板。
5. 4096-env AMP 正式负载；只有实测 OOM 后才调整环境数或训练参数。
6. 首批 AMP/noAMP 正式结果的 TensorBoard、comparison CSV/PNG 与人工运动效果分析。

## 7. 后续变更规则

- 修改正式行为时先更新最终设计规格，再修改代码和测试。
- 单纯实验超参数变化记录在对应 run 的 `config.json`、`run.json` 和实验分析中，不为每次实验复制一份设计规格。
- V1 的训练机验证结果可继续追加到本文；若算法、奖励结构、观测边界或正式地形集合发生变化，则创建新的版本基线文档。
- 清理旧 MGDP 代码前必须从 `go2_amp` 正式入口追踪依赖，确认不会删除 Randomdog 感知数据流、WM、历史编码、PPO 或必要环境基础。
