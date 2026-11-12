#!/usr/bin/env python3
"""
build_queue.py — 本地零依赖任务单编译与校验工具 (macOS / Linux / Colab 通用)

用途：
  将 AI Agent 编写的高层分镜描述 (storyboard JSON) 编译并严格校验为
  WanGP (`shared.api.submit_manifest` / `wgp.py --process`) 直接可执行的任务单。

核心校验规则（对齐 MiniMax-H3 硬件与模型约束）：
  1. 分辨率长宽必须同时为 32 的整数倍（支持 `768p_16_9` 等别名自动映射）。
  2. 帧数遵循 MiniMax-H3 的 `17 * n + 5` 对齐规则，有效区间 `[124, 362]`（24 FPS 下约 5.1s ~ 15.1s）。
  3. 提示词结构检查：Base 模式检查 `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`；
     Ref2VA 模式检查 `subject_definitions` / `summary` / `retention_analysis` / `detailed_description`。
  4. 云端媒体直连 (`drive://assets/...`)：通过 `os.stat` 仅校验云端元数据而绝不下载或打包大体积媒体到本地磁盘。
  5. 默认关闭云端 `prompt_enhancer`（避免在 Colab 额外加载 27B 扩写模型浪费显存与 CU）。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRESETS_PATH = REPO_ROOT / "configs" / "presets.json"

H3_FPS = 24
H3_MIN_FRAMES = 124  # 17 * 7 + 5 = 124 (~5.17s)
H3_MAX_FRAMES = 362  # 17 * 21 + 5 = 362 (~15.08s)

BASE_PROMPT_SECTIONS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
REF_PROMPT_SECTIONS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


def detect_local_drive_workspace(workspace_name: str = "MiniMaxH3_Workspace") -> Path | None:
    """探测本机挂载的 Google Drive 工作区路径（仅用于元数据校验，不读取文件内容）。"""
    env_root = os.environ.get("MINIMAX_H3_DRIVE_ROOT")
    if env_root:
        cand = Path(env_root).expanduser() / workspace_name
        if cand.is_dir():
            return cand

    symlink_ws = REPO_ROOT / "drive_workspace"
    if symlink_ws.is_dir():
        return symlink_ws

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if cloud_storage.is_dir():
        for gd in sorted(cloud_storage.glob("GoogleDrive-*")):
            for sub in ("My Drive", "我的云端硬盘", "マイドライブ"):
                cand = gd / sub / workspace_name
                if cand.is_dir():
                    return cand
    return None


def align_h3_frames(duration_or_frames: int | float | str) -> int:
    """将秒数（如 '8s' / 8.0）或原始帧数向上对齐到 MiniMax-H3 要求的 17*n + 5 网格（范围 124..362）。"""
    if isinstance(duration_or_frames, str):
        s = duration_or_frames.strip().lower()
        if s.endswith("s"):
            seconds = float(s[:-1])
            raw_frames = round(seconds * H3_FPS)
        else:
            raw_frames = int(s)
    elif isinstance(duration_or_frames, float):
        raw_frames = round(duration_or_frames * H3_FPS)
    else:
        raw_frames = duration_or_frames * H3_FPS if duration_or_frames <= 20 else duration_or_frames

    raw_frames = max(H3_MIN_FRAMES, min(H3_MAX_FRAMES, raw_frames))
    n = math.ceil((raw_frames - 5) / 17)
    aligned = 17 * n + 5
    return max(H3_MIN_FRAMES, min(H3_MAX_FRAMES, aligned))


def resolve_resolution(res: str, aliases: dict[str, str]) -> str:
    resolved = aliases.get(res, res).strip().lower()
    m = re.fullmatch(r"(\d+)x(\d+)", resolved)
    if not m:
        raise ValueError(
            f"非法分辨率格式 '{res}'（支持 '960x544' 或预设别名 {list(aliases.keys())}）"
        )
    w, h = int(m.group(1)), int(m.group(2))
    if w % 32 != 0 or h % 32 != 0:
        raise ValueError(f"分辨率 {w}x{h} 宽高必须同时为 32 的倍数（MiniMax-H3 网格约束）")
    return f"{w}x{h}"


def validate_h3_prompt(prompt: str, model_type: str, strict: bool = False) -> list[str]:
    warnings: list[str] = []
    if not prompt or not prompt.strip():
        raise ValueError("prompt 不能为空")
    is_ref = "ref2va" in model_type
    expected = REF_PROMPT_SECTIONS if is_ref else BASE_PROMPT_SECTIONS
    missing = [sec for sec in expected if sec not in prompt]
    if missing:
        msg = (
            f"模型 '{model_type}' 提示词缺少官方 h3-prompt-writing 推荐段落: {missing}。"
            " 建议先按 .agents/skills/h3-prompt-writing 规范扩写以获得最佳音画同步质量。"
        )
        if strict:
            raise ValueError(msg)
        warnings.append(msg)
    return warnings


def validate_drive_uri(uri: str, drive_ws: Path | None) -> str | None:
    """仅用 os.stat 检查 drive:// 指向的云端素材是否存在，不触发 FileProvider 下载数据块。"""
    if not uri.startswith("drive://"):
        return None
    rel = uri[len("drive://"):].lstrip("/")
    if drive_ws is None:
        return None
    target = drive_ws / rel
    try:
        os.stat(target)
    except FileNotFoundError:
        return f"未在 Google Drive 工作区找到云端素材 '{uri}' (期望路径: {target})"
    return None


def compile_shot(
    shot: dict[str, Any],
    presets_cfg: dict[str, Any],
    default_preset: str,
    drive_ws: Path | None,
    strict_prompt: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    preset_name = shot.get("preset", default_preset)
    presets_map = presets_cfg["presets"]
    if preset_name not in presets_map:
        raise KeyError(f"未知 preset '{preset_name}'，可选值: {list(presets_map.keys())}")

    task = copy.deepcopy(presets_map[preset_name]["settings"])

    if "prompt" not in shot:
        raise ValueError(f"镜头 '{shot.get('id', '<unnamed>')}' 缺少 prompt 字段")
    task["prompt"] = shot["prompt"].strip()

    if "resolution" in shot:
        task["resolution"] = resolve_resolution(shot["resolution"], presets_cfg["resolutions"])
    else:
        task["resolution"] = resolve_resolution(task["resolution"], presets_cfg["resolutions"])

    if "duration_seconds" in shot:
        task["video_length"] = align_h3_frames(float(shot["duration_seconds"]))
    elif "video_length" in shot:
        task["video_length"] = align_h3_frames(shot["video_length"])

    if "seed" in shot and shot["seed"] is not None:
        task["seed"] = int(shot["seed"])

    if "num_inference_steps" in shot:
        task["num_inference_steps"] = int(shot["num_inference_steps"])

    warnings = validate_h3_prompt(task["prompt"], task["model_type"], strict=strict_prompt)

    for media_key in ("image_start", "image_end", "image_refs", "video_source", "audio_guide", "audio_guide2"):
        if media_key in shot and shot[media_key]:
            val = shot[media_key]
            task[media_key] = val
            if media_key == "audio_guide" and not task.get("audio_prompt_type"):
                task["audio_prompt_type"] = "A"
            if isinstance(val, str):
                err = validate_drive_uri(val, drive_ws)
                if err:
                    warnings.append(err)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        err = validate_drive_uri(item, drive_ws)
                        if err:
                            warnings.append(err)

    if "overrides" in shot and isinstance(shot["overrides"], dict):
        task.update(shot["overrides"])

    if "id" in shot:
        task["_shot_id"] = str(shot["id"])
    if "title" in shot:
        task["_shot_title"] = str(shot["title"])

    return task, warnings


def collect_local_media(tasks: list[dict[str, Any]], base_dir: Path) -> list[tuple[Path, str]]:
    """收集本地存在的参考媒体文件（跳过 drive:// 与 http(s)://），仅在无 Drive 环境需要离线 ZIP 时使用。"""
    collected: list[tuple[Path, str]] = []
    media_keys = ("image_start", "image_end", "video_source", "audio_guide", "audio_guide2")

    def _resolve_and_rewrite(val: str, idx: int, key: str) -> str:
        if val.startswith(("http://", "https://", "drive://", "/content/")):
            return val
        p = Path(val)
        local_p = p if p.is_absolute() else (base_dir / p)
        if local_p.is_file():
            arcname = f"assets/shot_{idx:02d}_{key}_{local_p.name}"
            collected.append((local_p.resolve(), arcname))
            return arcname
        return val

    for idx, t in enumerate(tasks, start=1):
        for k in media_keys:
            if isinstance(t.get(k), str) and t[k]:
                t[k] = _resolve_and_rewrite(t[k], idx, k)
        if isinstance(t.get("image_refs"), list):
            new_refs = []
            for r_i, ref in enumerate(t["image_refs"]):
                if isinstance(ref, str) and ref:
                    new_refs.append(_resolve_and_rewrite(ref, idx, f"ref{r_i}"))
            t["image_refs"] = new_refs
    return collected


def main() -> int:
    parser = argparse.ArgumentParser(description="将分镜 Storyboard JSON 编译为 WanGP 可执行的 MiniMax-H3 任务队列")
    parser.add_argument("storyboard", type=Path, help="输入的分镜脚本 JSON 路径（例如 jobs/demo_cyberpunk_ramen.storyboard.json）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出的 .wangp.json 路径（默认同名替换后缀）")
    parser.add_argument("--presets", type=Path, default=DEFAULT_PRESETS_PATH, help="presets.json 配置文件路径")
    parser.add_argument("--pack-zip", action="store_true", help="可选：打包本地参考素材为 .bundle.zip（推荐使用 drive:// 免打包省空间）")
    parser.add_argument("--strict-prompt", action="store_true", help="若提示词缺少官方推荐段落则直接报错退出")
    args = parser.parse_args()

    presets_cfg = json.loads(args.presets.read_text(encoding="utf-8"))
    sb = json.loads(args.storyboard.read_text(encoding="utf-8"))

    default_preset = sb.get("default_preset", "fl2va_8step_quality")
    shots = sb.get("shots", [])
    if not shots:
        print("❌ Storyboard 中没有任何 shots 条目！", file=sys.stderr)
        return 1

    drive_ws = detect_local_drive_workspace()
    compiled_tasks: list[dict[str, Any]] = []
    all_warnings: list[str] = []

    for i, shot in enumerate(shots, start=1):
        shot.setdefault("id", f"shot_{i:02d}")
        task, warnings = compile_shot(shot, presets_cfg, default_preset, drive_ws, strict_prompt=args.strict_prompt)
        compiled_tasks.append(task)
        for w in warnings:
            all_warnings.append(f"[{shot['id']}] {w}")

    if args.output is not None:
        out_json = args.output
    else:
        stem = args.storyboard.name.replace(".storyboard.json", "").replace(".json", "")
        out_json = args.storyboard.parent / f"{stem}.wangp.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(compiled_tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total_frames = sum(int(t["video_length"]) for t in compiled_tasks)
    est_sec = total_frames / H3_FPS
    print(f"✅ 已生成轻量任务清单: {out_json} ({out_json.stat().st_size} bytes)")
    print(f"   镜头数量: {len(compiled_tasks)} | 合计帧数: {total_frames} 帧 (~{est_sec:.1f}s 24fps 立体声视频)")

    if args.pack_zip:
        zip_tasks = copy.deepcopy(compiled_tasks)
        media_files = collect_local_media(zip_tasks, args.storyboard.parent)
        out_zip = out_json.with_suffix(".bundle.zip")
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.wangp.json", json.dumps(zip_tasks, indent=2, ensure_ascii=False))
            for src_path, arcname in media_files:
                zf.write(src_path, arcname)
        print(f"📦 已打包离线任务包: {out_zip} (内嵌本地参考素材 {len(media_files)} 个)")

    for w in all_warnings:
        print(f"⚠️  {w}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
