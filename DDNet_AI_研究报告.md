# DDNet AI 自动通关 —— 技术调研报告

> 调研日期：2026-05-24

---

## 目录

1. [DDNet 游戏概述](#1-ddnet-游戏概述)
2. [DDNet 的核心挑战](#2-ddnet-的核心挑战)
3. [现有 DDNet AI/Bot 相关工作](#3-现有-ddnet-aibot-相关工作)
4. [可用的基础设施与工具](#4-可用的基础设施与工具)
5. [推荐技术路线](#5-推荐技术路线)
6. [训练环境架构设计](#6-训练环境架构设计)
7. [算法选择与对比](#7-算法选择与对比)
8. [实施路线图](#8-实施路线图)
9. [风险与开放问题](#9-风险与开放问题)
10. [参考文献与资源](#10-参考文献与资源)

---

## 1. DDNet 游戏概述

**DDNet（DDraceNetwork）** 是一款基于 Teeworlds 的开源多人合作平台跳跃游戏。玩家操控名为 "Tee" 的角色，通过**钩索（Hook）**和各种武器（锤子、榴弹枪、激光等）配合物理引擎，通过充满障碍的关卡。

### 核心机制

| 机制 | 描述 |
|------|------|
| **物理引擎** | 确定性物理，客户端/服务端共享 `CCharacterCore`，50 tick/s |
| **移动** | 地面/空中移动，摩擦力、加速度系统 |
| **跳跃** | 地面跳 + 二段跳（空中），可配置多于 2 次跳跃 |
| **钩索** | 直线射出，可钩墙、钩其他玩家，拉动角色，无时间限制 |
| **武器** | 锤子（近战/击飞）、手枪、霰弹枪、榴弹枪、激光枪 |
| **冻结** | 被冻结后无法移动，需队友解冻 |
| **飞行技巧** | Hammerfly、Hookfly、Rocketfly、Speedfly 等需要多人配合或精确操作的技术 |

### DDNet 关卡的特点

- 地图由**瓦片（Tiles）** 组成，包括：可钩墙、不可钩墙、钩穿透墙、冻结区、传送区、加速带、即死区、检查点、起点/终点等
- 关卡通常**线性推进**（起点 → 多个检查点 → 终点）
- 许多关卡可以使用**单人**完成，但部分需要**双人合作**（Dummy 模式）
- 地图难度跨度极大：从 Novice（新手）到 Brutal（残酷）

---

## 2. DDNet 的核心挑战

使用 AI/RL 让 DDNet 自动通关面临以下核心挑战：

### 2.1 精准的连续控制
- DDNet 的物理引擎要求**逐 tick（50Hz）的精确输入**，包括移动方向、跳跃时机、钩索角度、武器使用时机
- 许多技巧（如 Hammerfly）需要在**几个 tick 级别的窗口**内协调多个按键
- 动作空间是**连续+离散混合**的：钩索角度是连续的 0-360°，移动方向是离散的（左/右），跳跃/武器是离散的（按下/释放）

### 2.2 长长的 horizon 与稀疏奖励
- 一个典型关卡可能需要**几分钟到几十分钟**的连续操作
- 奖励信号非常稀疏：只有通过检查点或通关时才有明确的进度信号
- 中间状态缺乏自然的密集奖励（"你离终点还有多远"不易量化）

### 2.3 部分关卡需要双人合作
- 某些地图强制要求两个 Tee 协作（如 Hammerfly、Hookfly 等双人飞行技巧）
- 这需要训练**多智能体（Multi-Agent）** 系统，复杂度显著增加

### 2.4 多样化的关卡设计
- DDNet 有**数千张地图**，每张地图的障碍配置、物理参数（Tune Zones）可能不同
- 理想的 AI 需要在**未见过的地图上泛化**，而不是仅记住特定关卡的按键序列

### 2.5 缺乏现成的 RL 环境
- DDNet 本身**不是为 AI 训练设计的**，没有 Python Gym 接口
- 需要在 DDNet 物理引擎之上构建自定义训练环境

---

## 3. 现有 DDNet AI/Bot 相关工作

### 3.1 Paszczak 的神经网络+遗传算法 Bot（2016-2017）

**来源**：[Teeworlds 论坛](https://www.teeworlds.com/forum/viewtopic.php?pid=117570)

这是目前已知**唯一**在 DDNet 上尝试机器学习方法的公开项目：

| 项目 | 详情 |
|------|------|
| **作者** | Paszczak |
| **方法** | 神经网络 + 遗传算法（Neuroevolution） |
| **平台** | 直接嵌入 Teeworlds 客户端 |
| **训练方式** | 实时在线学习（在游玩地图的同时学习） |
| **适应度函数** | 仅使用**距出生点的距离** |
| **种群规模** | 100 个模拟 = 1 代 |
| **训练速度** | 约 4-5 代后出现明显改善 |
| **局限性** | 只能处理**单向直线前进**的地图；无法处理需要折返的地图（如 Aip-Gores）；一次只能运行一个模拟 |

**评价**：这是一个概念验证性质的项目，证明了 DDNet 上使用进化算法是可行的。但受限于 2016 年的方法和计算资源，远未达到实用水平。

### 3.2 现有的规则式 Bot 项目

DDNet 生态中有多个**非 ML 的自动化 Bot**：

| 项目 | 语言 | 描述 |
|------|------|------|
| [**fluffytw**](https://github.com/krxclient/fluffytw) | C++ | 开源 DDNet Bot 库，支持输入注入、ESP 覆盖层、碰撞访问 |
| [**chillerbot-ng**](https://github.com/chillerbot/chillerbot-ng) | C++ | 无头（Headless）控制台客户端，ASCII 地图渲染，支持键盘控制 |
| [**chillerbot-ux**](https://github.com/chillerbot/chillerbot-ux) | C++ | 完整的 DDNet 客户端分支，含自动化功能 |

这些 Bot 依赖**硬编码规则或手动操作**，不具备学习能力，但可以作为 RL agent **注入控制的接口**。

---

## 4. 可用的基础设施与工具

### 4.1 物理引擎实现

| 库 | 语言 | 描述 | 适用场景 |
|----|------|------|----------|
| [**DDNet 源码**](https://github.com/ddnet/ddnet) `CCharacterCore` | C++ | DDNet 官方物理引擎，确定性、tick 级别精确 | 生成真实训练数据，或以 C++ 构建训练环境 |
| [**twgame**](https://docs.rs/crate/twgame/) | Rust | 独立的 DDNet 物理 reimplementation，覆盖所有机制 | 以 Rust 构建训练环境，或通过 FFI 暴露给 Python |
| [**ddnet-rs**](https://github.com/ddnet/ddnet-rs) | Rust | DDNet 完整 Rust 重写，100+ crates，含游戏服务器 | 完整可控的游戏服务器，适合大规模并行训练 |

### 4.2 地图解析

| 库 | 语言 | 描述 |
|----|------|------|
| [**twmap**](https://docs.rs/crate/twmap/) | Rust | 解析/编辑/保存 Teeworlds & DDNet 地图，支持 `.map` 和 `.twmap.tar` |

### 4.3 数据记录与回放

| 系统 | 描述 |
|------|------|
| **DDNet Demo 系统** | 录制网络快照、玩家输入和游戏消息 |
| **Teehistorian** | 详细游戏事件记录（输入、伤害、瓦片交互、检查点等） |
| [**teehistorian-replayer**](https://docs.rs/crate/teehistorian-replayer/) | 以比实时更快的速度回放 Teehistorian 文件 |

### 4.4 DDNet 模拟器的关键特征

- **确定性物理**：相同输入 → 相同结果，这是 RL 训练的**关键前提**
- **50 tick/s 固定频率**：输入/状态的时间分辨率已确定
- **可加速运行**：teehistorian-replayer 已证明可以**比实时更快**地运行物理模拟，这意味着可以加速训练

---

## 5. 推荐技术路线

基于调研，我推荐以下**分层递进**的技术路线：

### 路线 A：基于 DDNet 物理引擎的 Gym 环境 + PPO（推荐首选）

```
DDNet CCharacterCore (C++) / twgame (Rust)
        ↓ Python FFI 封装 (PyO3 / ctypes)
    OpenAI Gym 环境
        ↓
    PPO / SAC 训练 (Stable-Baselines3 / CleanRL)
```

**优点**：
- 利用 DDNet 确定性的物理引擎，训练数据完全准确
- 可以在 tick 级别进行精确控制
- 可以**加速运行**（比实时快 100-1000x）
- PPO 是连续控制+平台跳跃的成熟方案

**缺点**：
- 需要编写 Python-C++/Rust 桥接层（~1-2 周工作量）
- 地图逻辑（检查点、传送带等）需要额外实现

### 路线 B：纯视觉方法（像素输入 + 模仿学习）

```
DDNet 客户端
        ↓ 屏幕捕获
    视觉编码器 (CNN/ViT)
        ↓
    行为克隆 / DAgger / NitroGen 式大模型
```

**优点**：
- 不需要修改游戏源码
- 可以从人类 replay 数据中学习（DDNet 有大量排行榜 replay）

**缺点**：
- 需要大量人类演示数据（NitroGen 用了 40,000 小时）
- 训练和推理需要 GPU，速度慢
- 缺乏物理模型的先验知识，样本效率低

### 路线 C：分层混合方法（最可能成功的长期方案）

```
高层策略网络（PPO/MCTS）
    ↓ 选择子目标/技巧
低层技能控制器（硬编码 + 微调RL）
    ↓ 执行具体操作
DDNet 物理模拟
```

**优点**：
- 高层处理关卡推理（"这里需要 Hammerfly 过去"）
- 低层处理精确执行（"Hammerfly 需要每 4 tick 按一次锤子"）
- 结合了规则方法的精确性和 RL 的泛化能力

**缺点**：
- 架构最复杂
- 需要定义技能原语（Skill Primitives）

---

## 6. 训练环境架构设计

### 6.1 推荐架构（基于路线 A）

```
┌─────────────────────────────────────────────┐
│                  Python 训练循环              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Policy   │  │  PPO     │  │  Reward   │  │
│  │  Network  │←─│  Trainer │──│  Computer │  │
│  └─────┬─────┘  └──────────┘  └─────┬─────┘  │
│        │ action                     │ reward  │
│  ┌─────▼────────────────────────────▼──────┐  │
│  │         DDNet Gym Environment           │  │
│  │  ┌──────────────────────────────────┐   │  │
│  │  │  CCharacterCore (via PyO3 FFI)   │   │  │
│  │  │  - 确定性物理 tick              │   │  │
│  │  │  - 碰撞检测                      │   │  │
│  │  │  - 地图瓦片查询                  │   │  │
│  │  └──────────────────────────────────┘   │  │
│  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### 6.2 状态空间（Observation Space）

建议使用**混合状态表示**（非纯像素）：

| 类别 | 具体信息 | 维度 |
|------|----------|------|
| **角色状态** | 位置 (x,y)、速度 (vx,vy)、是否在地面、跳跃次数剩余、是否冻结 | ~10 |
| **钩索状态** | 是否在钩中、钩中目标类型（墙/玩家）、钩索角度、钩索长度 | ~8 |
| **环境感知** | 角色周围的瓦片类型网格（如 15×15 的 tile grid） | ~225 |
| **关卡进度** | 已过检查点、距下一个检查点的向量、距终点的向量 | ~6 |
| **全局信息** | 当前 tick、团队状态（如果是双人模式） | ~4 |

**总计：~250 维**的状态向量，可以被一个 MLP 策略网络高效处理。

### 6.3 动作空间（Action Space）

| 动作 | 类型 | 范围 |
|------|------|------|
| 水平移动 | 离散 | {-1, 0, 1}（左/停/右） |
| 跳跃 | 离散 | {0, 1}（按下/释放） |
| 钩索 | 混合 | {0, 1}（是否发射）+ 角度 ∈ [0, 360°) 连续 |
| 武器使用 | 离散 | {0, 1}（锤子） |
| 瞄准角度 | 连续 | [0, 360°) |

**总计：约 6 个独立的动作维度**（4 离散 + 2 连续）。

### 6.4 奖励函数设计

这是**最关键的设计决策之一**。建议分层奖励：

```
总奖励 = 进度奖励 + 存活奖励 + 通关奖励 + 技巧奖励 - 惩罚
```

| 奖励项 | 权重 | 描述 |
|--------|------|------|
| **检查点通过** | +10.0 | 每通过一个检查点 |
| **进度增量** | +0.1/帧 | 距终点距离的减少量（稠密奖励） |
| **通关** | +100.0 | 到达终点 |
| **存活** | +0.001/tick | 鼓励 agent 不自杀 |
| **死亡** | -5.0 | 接触到即死瓦片或掉出地图 |
| **时间惩罚** | -0.0001/tick | 轻微鼓励快速通关 |
| **后退惩罚** | -0.01/帧 | 当 agent 远离终点时 |

### 6.5 并行化

- 使用 **向量化环境**（Vectorized Env），在单个 GPU 上同时运行 64-256 个 DDNet 模拟实例
- twgame (Rust) 或 CCharacterCore (C++) 体积小、无渲染，可以在单机上运行上千个并行实例
- 这可以将训练时间从数周缩短到数天

---

## 7. 算法选择与对比

| 算法 | 适合度 | 理由 |
|------|--------|------|
| **PPO** | ⭐⭐⭐⭐⭐ | 连续+离散混合动作空间的首选；稳定、成熟；CleanRL/Stable-Baselines3 有良好实现 |
| **SAC** | ⭐⭐⭐⭐ | 连续控制的 SOTA，但处理离散动作需 Discrete SAC 变体 |
| **DQN/DDQN** | ⭐⭐ | 仅支持离散动作，无法处理连续的钩索角度 |
| **遗传算法（GA/NEAT）** | ⭐⭐⭐ | Paszczak 已验证可行；适合初始探索；样本效率低于梯度方法 |
| **MCTS + NN（AlphaZero 式）** | ⭐⭐⭐ | 适合需要长期规划的关卡；实现复杂度高；需要可回滚的模拟器 |
| **行为克隆（BC）** | ⭐⭐⭐ | 可以从人类 replay 预训练；单独使用泛化差；建议 + Dagger 或 + RL fine-tune |
| **HRL（分层RL）** | ⭐⭐⭐⭐ | 路线 C 的核心；处理长 horizon 问题的理论最佳方案 |

### 推荐算法组合

**第一阶段（概念验证）**：PPO + 稠密奖励 + 简单单人地图
**第二阶段（提升泛化）**：PPO + 课程学习（逐渐增加地图难度）+ 领域随机化
**第三阶段（处理双人）**：MAPPO（Multi-Agent PPO）或自对弈（Self-Play）
**第四阶段（最终系统）**：HRL（高层目标选择 + 低层技能策略）

---

## 8. 实施路线图

### Phase 0：基础设施搭建（2-4 周）

- [ ] 选择物理后端（推荐 **twgame** Rust crate，通过 PyO3 绑定到 Python）
- [ ] 实现地图加载器（解析 `.map` 文件，提取瓦片碰撞数据、检查点位置、实体信息）
- [ ] 实现基础 Gym 环境：reset（加载地图、初始化角色）、step（执行动作，推进物理模拟）、get_obs（构建状态向量）、get_reward
- [ ] 验证确定性：相同输入序列 → 相同轨迹
- [ ] 实现比实时加速运行（目标：≥100x 实时速度）

### Phase 1：单人简单关卡验证（4-6 周）

- [ ] 选择 3-5 张 Novice 难度单人地图作为测试集
- [ ] 训练 PPO agent，目标：在训练地图上通关
- [ ] 调优奖励函数、网络架构、超参数
- [ ] 目标指标：训练地图通关率 >80%

### Phase 2：泛化与课程学习（4-8 周）

- [ ] 实现课程学习：自动按难度排序地图，从易到难训练
- [ ] 引入领域随机化：摩擦系数、重力、初始位置微调
- [ ] 在未见过的 Moderate 难度地图上测试 zero-shot 性能
- [ ] 如果泛化不好，考虑加入 RNN/LSTM 策略网络以处理时序信息

### Phase 3：双人合作与复杂地图（8-12 周）

- [ ] 实现多智能体环境（两个 Tee 同时控制或自对弈）
- [ ] 探索 MAPPO 或自对弈方法
- [ ] 处理 Hammerfly、Hookfly 等双人技巧
- [ ] 在 Brutal 难度地图上测试

### Phase 4：集成与部署（4-6 周）

- [ ] 将训练好的策略部署为 DDNet 客户端插件
- [ ] 实现在线推理（50Hz 实时）
- [ ] 可选：Web 可视化展示 Agent 通关过程

---

## 9. 风险与开放问题

| 风险 | 严重程度 | 缓解策略 |
|------|----------|----------|
| **长 horizon 无法收敛** | 高 | 使用课程学习从短关卡开始；HRL 分解为子任务；检查点作为子目标 |
| **模拟器性能不足** | 中 | twgame/ddnet-rs 的 Rust 实现已证明可高速运行；使用向量化并行 |
| **双人协作难以学习** | 高 | 先完成单人系统；双人可从人类 replay 中预训练 |
| **某些地图需要特定的 trick 知识** | 中 | 这些 trick 可以作为技能原语预先编程，RL 只负责"何时使用" |
| **DDNet 版本更新破坏兼容性** | 低 | 物理引擎相对稳定；使用固定版本的 twgame |
| **Legacy 地图（0.6）与新版地图（0.7）的格式差异** | 低 | twmap 库已支持双向格式 |

### 关键开放问题

1. **神经网络架构**：MLP 是否足够，还是需要 Transformer/GNN 来处理瓦片网格的空间结构？
2. **单智能体还是多智能体**：Single-agent 用 Dummy（本地双人）模式是否可行？
3. **从零训练 vs 预训练**：是否值得先收集人类 replay 数据进行行为克隆预训练？
4. **地图表示**：是否将所有地图瓦片信息编码进状态，还是仅编码局部窗口？

---

## 10. 参考文献与资源

### DDNet 相关

| 资源 | 链接 |
|------|------|
| DDNet 官方仓库 | https://github.com/ddnet/ddnet |
| DDNet 物理引擎源码 | `src/game/gamecore.cpp` (CCharacterCore) |
| DDNet Rust 重写 | https://github.com/ddnet/ddnet-rs |
| twgame 物理库 (Rust) | https://docs.rs/crate/twgame/ |
| twmap 地图解析库 (Rust) | https://docs.rs/crate/twmap/ |
| fluffytw Bot 库 (C++) | https://github.com/krxclient/fluffytw |
| chillerbot-ng 无头客户端 | https://github.com/chillerbot/chillerbot-ng |
| chillerbot-ux 自动化客户端 | https://github.com/chillerbot/chillerbot-ux |
| Teehistorian Replayer | https://github.com/ddnet/ddnet/issues/2937 |
| DDNet 游戏机制详解 | https://deepwiki.com/ddnet/ddnet |
| Paszczak 的 NN+GA Bot (2016) | https://www.teeworlds.com/forum/viewtopic.php?pid=117570 |

### RL 方法参考

| 资源 | 描述 |
|------|------|
| [Kinetix](https://arxiv.org/abs/2410.23208) | 2D 物理 RL 通用 agent，ICLR 2025 Oral，使用 JAX 加速、Transformer 策略、程序化生成任务 |
| [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) | 成熟的游戏 RL 框架，PPO/SAC，课程学习，模仿学习 |
| [Neural-Jump](https://github.com/Ayfri/Neural-Jump) | 简单平台跳跃 + 神经进化，PyTorch 实现 |
| [NVIDIA NitroGen](https://github.com/MineDojo/NitroGen) | 视觉→动作大模型，1000+ 游戏，40K 小时训练 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | 单文件 RL 实现，PPO/SAC/DQN，适合研究和修改 |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | 生产级 RL 库，PPO/SAC/TD3 等 |

### 关键论文

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- **SAC**: Haarnoja et al., "Soft Actor-Critic", 2018
- **MAPPO**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", 2022
- **HRL**: Nachum et al., "Data-Efficient Hierarchical Reinforcement Learning", 2018
- **Kinetix**: Matthews et al., "Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks", ICLR 2025

---

## 总结

用 AI/RL 让 DDNet 自动通关是一个**技术上可行、但工程上具有挑战性的项目**。关键判断：

1. **确定性物理引擎是最大优势**——DDNet 的 `CCharacterCore` 是确定性的，这为 RL 训练提供了完美的模拟环境，无需处理随机性
2. **已有成熟的底层基础设施**——twgame (Rust)、twmap、chillerbot、fluffytw 等项目提供了坚实的技术基础
3. **PPO 是首选算法**——在平台跳跃类游戏中已被反复验证有效
4. **分层强化学习（HRL）是最有希望的长期方案**——DDNet 关卡的层次结构（关卡 → 段落 → 技巧 → 按键序列）天然适合分层分解
5. **建议从简单单人地图开始**——先证明基本可行性，再逐步扩展复杂度和通用性

预计**最小的可行原型（MVP）需要约 6-10 周的全职开发**，能够在少数 Novice 难度地图上实现自动通关。
