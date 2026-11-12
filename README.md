# MiniMax-H3 AI-Automated Pipeline (Colab A100 40GB Edition)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/flynn-os/MiniMaxH3_pipeline/blob/main/notebooks/MiniMaxH3_Colab_A100_Runner.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

基于 **`deepbeepmeep/WanGP` 无头 Python API (`shared.api`)** + **MiniMax 官方 `h3-prompt-writing` 提示词规范** 构建的自动化音视频生成流水线。

专为三大目标设计：
1. **本机零算力、零大文件囤积 (`Zero-Local-Disk`)**：本机仅保留几 KB 的文本配置与分镜 JSON；所有大体积权重 (`cache/`)、参考图/音视频 (`assets/`) 与生成成片 (`outputs/`) **100% 存储在 Google Drive**（配合 macOS FileProvider 流式挂载，本机物理磁盘占用为 `0B`）。
2. **AI 全程自动化编排 (`Agent-First`)**：内置导演与提示词规范技能，一句话生成符合 `32px` 网格、`17*n+5` 帧数与 `<d>[Language] ...</d>` 口型同步规范的任务队列。
3. **Colab `A100 (40GB)` 极速批跑即销毁 (`Zero-Idle-CU`)**：单次加载模型顺序渲染全部镜头，刷盘同步 Google Drive 后自动调用 `google.colab.runtime.unassign()` 释放 GPU 停止计费。

---

## 一、核心架构与云地分层目录

### 1. 本地轻量控制仓库（仅代码与 JSON）
```text
MiniMaxH3_pipeline/
├── AGENTS.md                                 # Agent 自动加载的仓库级工作流契约
├── .agents/skills/
│   ├── h3-prompt-writing/                    # 官方 MiniMax-H3 音画同步提示词规范 (T2VA/FL2VA/Ref2VA)
│   └── minimax-h3-director/                  # 自动化导演编排技能 (一句话 -> 分镜 -> 编译校验)
├── configs/
│   └── presets.json                          # 对齐 WanGP v19.29 的 4 大黄金档位与标准 32px 分辨率字典
├── scripts/
│   ├── build_queue.py                        # [本机运行] 零依赖 Storyboard 编译器与 drive:// 云端 URI 校验器
│   ├── sync_drive.py                         # [本机运行] Google Drive 流式工作区管理 (push/pull/stage-asset/evict)
│   ├── colab_bootstrap.sh                    # [Colab运行] 一键部署 WanGP、规避 PyTorch 2.8/2.9 坑并绑定云盘缓存
│   └── colab_runner.py                       # [Colab运行] 无头批量渲染 + 战报输出 + 自动断连释放 A100 (亦支持 MCP/WebUI)
├── jobs/
│   ├── demo_cyberpunk_ramen.storyboard.json  # 双镜头霓虹雨夜拉面示例分镜脚本（含中文台词口型与环境立体声）
│   └── demo_cyberpunk_ramen.wangp.json       # 编译后的标准可执行队列
└── notebooks/
    └── MiniMaxH3_Colab_A100_Runner.ipynb     # Google Colab 一键运行工作台
```

### 2. Google Drive 云端重资产工作区 (`MyDrive/MiniMaxH3_Workspace/`)
```text
MiniMaxH3_Workspace/
├── assets/{images,videos,audio}/             # 参考图、参考视频、参考音色（通过 drive://assets/... 直接引用）
├── inbox/                                    # 待渲染队列（跑完自动归档至 inbox/processed/）
├── outputs/                                  # 渲染产出的 24fps + 32kHz 立体声 MP4 成片（按需流式播放，0 本地磁盘占用）
├── reports/                                  # ~1KB 轻量 JSON 执行战报（供本地 Agent 读取验收）
└── cache/{ckpts,loras,hf}/                   # 持久化模型权重与蒸馏 LoRA（首次下载，跨实例秒加载）
```

---

## 二、预设档位一览（`configs/presets.json`）

| 预设名 (`preset`) | 适用任务 | 底层模型与加速 LoRA | 默认步数 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **`fl2va_8step_quality`** *(默认)* | 文生音视频 (`T2VA`) / 首帧 (`I2VA`) / 首尾帧 (`FL2VA`) | `minimax_h3_fl2va_pruned` + LightX2V 8-step LoRA | **8 步** | 日常主力出片，画质、动态与立体声音效最佳平衡 |
| **`fl2va_4step_fast`** | 快速构图与台词节奏验证 | `minimax_h3_fl2va_pruned` + DMAD 4-step Critic LoRA | **4 步** | 极速草稿打样，极省 Colab CU |
| **`ref2va_8step_quality`** | 人物一致性 / 视频重绘 / 音色克隆参考 (`Ref2VA`) | `minimax_h3_ref2va_pruned` + LightX2V Ref2V 8-step LoRA | **8 步** | 传入 `image_refs` / `audio_guide` (`drive://assets/...`) 保持跨镜头角色与声音统一 |
| **`ref2va_2k_ic`** | 二阶段高细节重绘成片 | `minimax_h3_ref2va_pruned` + HyperFlow 8步 + Regenerate 2K IC 3步 | **8 + 3 步** | 保持角色身份、口型与原音轨的高清终极出片 |

---

## 三、全流程快速上手

### 1. 重素材直接上云，释放本机磁盘（可选，仅 `I2VA` / `Ref2VA` 需要）
```bash
# 将本地参考图/视频/音频迁入 Google Drive 并删除本地副本，自动返回 drive://assets/... URI
python3 scripts/sync_drive.py stage-asset /path/to/character.png --subdir images --move
```

### 2. 编译分镜并投递到云端队列（0 CU 消耗）
```bash
# 校验 32px 分辨率网格、17*n+5 帧数与 H3 提示词结构
python3 scripts/build_queue.py jobs/demo_cyberpunk_ramen.storyboard.json --strict-prompt

# 同步轻量工程与任务单至 Google Drive inbox/
python3 scripts/sync_drive.py push --job jobs/demo_cyberpunk_ramen.wangp.json
```

### 3. 打开 Colab `A100 (40GB)` 点火出片
点击顶部的 **Open in Colab** 徽章（或在 Google Drive 中打开 `MiniMaxH3_Colab_A100_Runner.ipynb`），选择 **`A100 GPU`** 并运行前 3 个单元格：
- 渲染完成后，MP4 直接保存在 `MiniMaxH3_Workspace/outputs/`，Colab 实例自动调用 `runtime.unassign()` 断开停止扣点。

### 4. 本地流式验收与空间归零维护
```bash
# 仅拉回 ~1KB 的 JSON 战报并列出 ./drive_workspace/outputs/ 下的云端流式视频路径（不下载 MP4）
python3 scripts/sync_drive.py pull

# 查看 Google Drive 云端大小 vs 本机物理磁盘占用对比
python3 scripts/sync_drive.py status

# 一键清理本地残留并驱逐 macOS FileProvider 缓存块，恢复 0B 本地占用
python3 scripts/sync_drive.py evict
```

---

## 四、合规与致谢声明

* 云端推理底座调用开源项目 **[deepbeepmeep/WanGP](https://github.com/deepbeepmeep/WanGP)**（`shared/api.py` 与 `mmgp` 显存管理引擎），遵循其使用条款。
* 模型权重与提示词规范源自 **[MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)**（遵循 MiniMax H3 Community License Agreement）及 **[ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)** 蒸馏 LoRA 生态。
