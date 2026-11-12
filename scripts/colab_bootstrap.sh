#!/usr/bin/env bash
# ==============================================================================
# colab_bootstrap.sh — Google Colab A100 (40GB) 一键环境初始化与云盘全量存储绑定脚本
#
# 功能：
#   1. 校验 GPU 型号与显存（推荐 A100 40GB，兼容 L4 24GB / G4 96GB）。
#   2. 创建并绑定 Google Drive 持久化目录（cache / assets / inbox / outputs / reports），
#      确保所有大体积权重、参考图/音视频素材与生成结果 100% 存储在 Google Drive。
#   3. 克隆或增量更新 deepbeepmeep/WanGP，规避 PyTorch 2.8/2.9 的 3D VAE 显存暴涨问题。
#   4. 写入 wgp_config.json 并建立软链，使 shared.api / CLI / WebUI 共享同一份缓存与输出目录。
# ==============================================================================
set -euo pipefail

WORKSPACE_DRIVE="${WORKSPACE_DRIVE:-/content/drive/MyDrive/MiniMaxH3_Workspace}"
WANGP_DIR="${WANGP_DIR:-/content/WanGP}"

echo "============================================================"
echo "🚀 [MiniMax-H3 Pipeline] Colab Bootstrap Starting..."
echo "   Persistent Cloud Drive : ${WORKSPACE_DRIVE}"
echo "   WanGP Runtime Dir      : ${WANGP_DIR}"
echo "============================================================"

# 1. GPU 与系统内存体检
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ 未检测到 NVIDIA GPU！请在 Colab 菜单「代码执行程序 -> 更改运行时类型」中选择 A100 GPU。"
  exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
echo "✅ 检测到 GPU: ${GPU_INFO}"
python3 - << 'PY'
import shutil, os
mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3)
total_disk, used_disk, free_disk = shutil.disk_usage("/")
print(f"✅ 系统内存 (Host RAM): {mem_gb:.1f} GB | 云端实例临时磁盘剩余: {free_disk / (1024.**3):.1f} GB")
if mem_gb < 30:
    print("⚠️  警告: 当前系统内存不足 30GB（疑似免费 T4 默认档），加载 MiniMax-H3 极易发生 Host RAM OOM！强烈建议切换至 A100 (40GB)。")
PY

# 2. 初始化 Google Drive 全量持久化目录结构（权重、输入素材、渲染成片全在云端）
mkdir -p \
  "${WORKSPACE_DRIVE}/cache/ckpts" \
  "${WORKSPACE_DRIVE}/cache/loras" \
  "${WORKSPACE_DRIVE}/cache/hf" \
  "${WORKSPACE_DRIVE}/assets/images" \
  "${WORKSPACE_DRIVE}/assets/videos" \
  "${WORKSPACE_DRIVE}/assets/audio" \
  "${WORKSPACE_DRIVE}/inbox/processed" \
  "${WORKSPACE_DRIVE}/inbox/failed" \
  "${WORKSPACE_DRIVE}/outputs" \
  "${WORKSPACE_DRIVE}/reports" \
  "${WORKSPACE_DRIVE}/logs"

export HF_HOME="${WORKSPACE_DRIVE}/cache/hf"

# 3. 克隆或更新 WanGP
if [ ! -d "${WANGP_DIR}/.git" ]; then
  echo "📦 正在克隆 deepbeepmeep/WanGP ..."
  git clone --depth 1 https://github.com/deepbeepmeep/WanGP.git "${WANGP_DIR}"
else
  echo "🔄 检测到已有 WanGP 目录，正在拉取最新更新..."
  git -C "${WANGP_DIR}" pull --ff-only || true
fi

# 4. 检查 PyTorch 版本（规避 WanGP 官方指出的 PyTorch 2.8 内存泄漏与 2.9 3D-VAE 显存爆炸 Bug）
TORCH_MINOR=$(python3 -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')" 2>/dev/null || echo "none")
echo "🔍 当前预装 PyTorch 主次版本: ${TORCH_MINOR}"
if [[ "${TORCH_MINOR}" == "2.8" || "${TORCH_MINOR}" == "2.9" || "${TORCH_MINOR}" == "none" ]]; then
  echo "⚡ 正在升级 PyTorch 至 WanGP 推荐的 2.10.0 (规避 2.8/2.9 的 VAE 显存暴涨问题)..."
  pip install -q --upgrade torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126 || \
  pip install -q --upgrade torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
fi

# 5. 安装 WanGP 核心依赖
echo "📦 正在安装 WanGP Python 依赖..."
pip install -q -r "${WANGP_DIR}/requirements.txt"

# 6. 尝试安装 SageAttention 2（可选加速，失败自动优雅回退至 sdpa/flash，绝不中断部署）
if ! python3 -c "import sageattention" >/dev/null 2>&1; then
  echo "⚡ 尝试安装 SageAttention 2 加速算子（若无匹配预编译轮子将自动回退至 SDPA）..."
  pip install -q sageattention --no-build-isolation 2>/dev/null || echo "ℹ️  跳过编译型 SageAttention，运行时将自动使用 PyTorch SDPA / FlashAttention。"
fi

# 7. 写入 wgp_config.json 并建立 LoRA URL 缓存软链，实现跨会话秒启动
python3 - "${WANGP_DIR}" "${WORKSPACE_DRIVE}" << 'PY'
import json, sys
from pathlib import Path

wangp_dir = Path(sys.argv[1])
drive_dir = Path(sys.argv[2])

cfg_path = wangp_dir / "wgp_config.json"
cfg = {}
if cfg_path.is_file():
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}

ckpts_dir = str(drive_dir / "cache" / "ckpts")
loras_dir = str(drive_dir / "cache" / "loras")
outputs_dir = str(drive_dir / "outputs")

cfg.update({
    "checkpoints_paths": [ckpts_dir],
    "loras_root": loras_dir,
    "save_path": outputs_dir,
    "image_save_path": outputs_dir,
    "audio_save_path": outputs_dir,
    "mesh_save_path": outputs_dir,
})
cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

drive_lora_cache = drive_dir / "cache" / "loras_url_cache_v2.json"
if not drive_lora_cache.exists():
    drive_lora_cache.write_text("{}\n", encoding="utf-8")

local_lora_cache = wangp_dir / "loras_url_cache_v2.json"
if local_lora_cache.exists() and not local_lora_cache.is_symlink():
    local_lora_cache.unlink()
if not local_lora_cache.exists():
    local_lora_cache.symlink_to(drive_lora_cache)

print(f"✅ 已绑定 Google Drive 持久化配置: {cfg_path}")
PY

echo "============================================================"
echo "🎉 Bootstrap 完成！模型权重、参考素材与渲染成片均已绑定至 Google Drive: ${WORKSPACE_DRIVE}"
echo "============================================================"
