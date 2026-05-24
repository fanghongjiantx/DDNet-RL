# DDNet-RL

用强化学习自动通关 [DDNet](https://ddnet.org/)（DDraceNetwork）平台跳跃关卡。

## 当前状态

**Phase 0 ✅** — 基础设施搭建完成。

- 物理后端：twgame (Rust) via PyO3，确定性验证通过
- 地图加载：twmap，支持解析 `.map` 格式
- Gym 环境：`ddnet_rl.DDNetEnv`，标准 gymnasium 接口
- 加速性能：~740K steps/s，约 14,800x 实时速度

## 项目结构

```
├── rust_backend/       # Rust 物理引擎（PyO3 绑定）
│   ├── Cargo.toml
│   └── src/lib.rs
├── ddnet_rl/           # Python Gym 环境
│   ├── __init__.py
│   └── env.py
├── maps/               # 测试地图
├── requirements.txt
├── pyproject.toml
└── DDNet_AI_研究报告.md  # 技术调研报告
```

## 快速开始

### 安装

```bash
# 安装 Rust（如未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 编译并安装 Python 包
cd rust_backend
maturin develop --release
cd ..

# 安装 Python 依赖
pip install -r requirements.txt
```

### 使用

```python
from ddnet_rl import DDNetEnv

env = DDNetEnv("maps/trainmap1.map")
obs, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # 替换为你的策略
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## 技术栈

- **物理引擎**: [twgame](https://crates.io/crates/twgame) v0.11 (Rust)
- **地图解析**: [twmap](https://crates.io/crates/twmap) v0.14 (Rust)
- **Python 绑定**: [PyO3](https://pyo3.rs/) v0.23
- **RL 框架**: [Gymnasium](https://gymnasium.farama.org/)

## 路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 基础设施（物理引擎 + Gym 环境 + 地图加载） | ✅ 完成 |
| 1 | 单人简单关卡 PPO 验证 | 🔜 下一步 |
| 2 | 课程学习 + 泛化 | 待开始 |
| 3 | 双人合作 + 复杂地图 | 待开始 |
| 4 | 部署为 DDNet 客户端插件 | 待开始 |

## 许可

MIT License
