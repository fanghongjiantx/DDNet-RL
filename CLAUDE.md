# DDNet-RL: 用强化学习自动通关 DDNet

## 项目目标
训练 AI agent 使用强化学习方法自动通过 DDNet（DDraceNetwork）平台的关卡。

## 技术方案概要
- **算法**: PPO（首选），后续演进到分层强化学习 (HRL)
- **物理后端**: twgame (Rust) 或 CCharacterCore (C++)，通过 PyO3/ctypes 绑定到 Python
- **训练环境**: 自定义 OpenAI Gym 环境，确定性物理 + 向量化并行
- **动作空间**: 混合（离散：移动/跳跃/武器 + 连续：钩索角度/瞄准角度）
- **状态空间**: 混合状态表示（角色物理状态 + 环境瓦片网格 + 进度信息），非纯像素

## 实施阶段
- Phase 0: 基础设施（物理引擎绑定 + Gym 环境 + 地图加载）
- Phase 1: 单人简单关卡 PPO 验证
- Phase 2: 课程学习 + 泛化
- Phase 3: 双人合作 + 复杂地图
- Phase 4: 部署为 DDNet 客户端插件

## 关键资源
- 调研报告: `DDNet_AI_研究报告.md`
- DDNet 物理引擎: `src/game/gamecore.cpp` (CCharacterCore) in github.com/ddnet/ddnet
- Rust 物理库: docs.rs/crate/twgame/
- Rust 完整重写: github.com/ddnet/ddnet-rs
- 地图解析: docs.rs/crate/twmap/

## 当前状态
调研阶段完成，准备进入 Phase 0 基础设施搭建。
