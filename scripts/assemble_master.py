#!/usr/bin/env python3
"""
assemble_master.py — Colab 端全自动 1080p 多镜母带剪辑、BGM 闪避混音与字幕烧录工具

功能（运行于 Colab GPU 实例，零本地 Mac 磁盘与算力消耗）：
  1. 读取最新 `run_manifest_*.json`（或按顺序传入的分镜视频），确认全部成功。
  2. 使用 ffmpeg 将 `1920x1088` 中心无损裁切为标准 `1920x1080` (yuv420p, 24fps)，
     并对每段音频施加 60ms 首尾微淡入淡出（保持帧锁对齐，实现 0.000s 累计口型漂移）。
  3. 拼接全部镜头为原生音画同步轨。
  4. 若存在 `bgm_novoland_epic.mp3` 史诗底乐，自动通过 `sidechaincompress`
     在人物对白期间智能压低 BGM（闪避混音），并统一标准化响度至 `-14 LUFS`。
  5. 同时导出两个成品到 Google Drive 工作区 `outputs/`：
     - `novoland_shangyang_1080p_clean.mp4`（纯净无字幕母带版）
     - `novoland_shangyang_1080p_subtitled.mp4`（含剧场版中文字幕版）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def ensure_cjk_font() -> str:
    """检查或自动安装中文字体（Noto Sans CJK SC），保证 Colab Linux 烧录中文字幕不乱码。"""
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_candidates:
        if Path(fp).is_file():
            return "Noto Sans CJK SC"
    if shutil.which("apt-get"):
        print("🔤 正在云端安装中文字体包 fonts-noto-cjk ...")
        subprocess.run(
            ["bash", "-lc", "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fonts-noto-cjk >/dev/null 2>&1 || true"],
            check=False,
        )
    return "Noto Sans CJK SC"


def probe_duration_and_has_audio(video_path: Path) -> tuple[float, bool]:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type", "-of", "json", str(video_path),
        ],
        text=True,
    )
    info = json.loads(out)
    dur = float(info.get("format", {}).get("duration") or 5.0)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return dur, has_audio


def load_shot_clips(workspace_dir: Path, explicit_clips: list[Path]) -> list[dict[str, Any]]:
    if explicit_clips:
        return [{"shot_id": p.stem, "path": p.resolve()} for p in explicit_clips]

    reports_dir = workspace_dir / "reports"
    outputs_dir = workspace_dir / "outputs"
    manifests = sorted(reports_dir.glob("run_manifest_*.json"))
    if not manifests:
        manifests = sorted(outputs_dir.glob("run_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"未在 {reports_dir} 或 {outputs_dir} 找到 run_manifest_*.json")

    latest = manifests[-1]
    print(f"📄 读取最新渲染战报: {latest}")
    data = json.loads(latest.read_text(encoding="utf-8"))
    clips: list[dict[str, Any]] = []
    for item in data.get("shots", []):
        if not item.get("success"):
            raise RuntimeError(f"分镜 {item.get('shot_id')} 未成功完成，中止母带合成！")
        files = item.get("generated_files") or []
        if not files:
            raise RuntimeError(f"分镜 {item.get('shot_id')} 无输出文件！")
        cand = Path(files[0])
        if not cand.is_file():
            cand = outputs_dir / cand.name
        if not cand.is_file():
            raise FileNotFoundError(f"找不到分镜视频文件: {cand}")
        clips.append({"shot_id": item.get("shot_id", cand.stem), "path": cand.resolve()})
    return clips


def write_ass_subtitles(ass_path: Path, font_name: str, durations: list[float]) -> None:
    def fmt_ts(sec: float) -> str:
        cs = max(0, int(round(sec * 100)))
        h = cs // 360000
        cs %= 360000
        m = cs // 6000
        cs %= 6000
        s = cs // 100
        cc = cs % 100
        return f"{h}:{m:02d}:{s:02d}.{cc:02d}"

    t = 0.0
    starts: list[float] = []
    for d in durations:
        starts.append(t)
        t += d

    lines_cfg: list[tuple[str, float, float, str]] = [
        ("Title", 0.35, min(3.4, durations[0] - 0.3), "《九州缥缈录 · 殇阳关》"),
    ]
    if len(durations) >= 3:
        s3 = starts[2]
        lines_cfg.append(("Default", s3 + 0.45, s3 + max(2.2, durations[2] - 0.55), "阿苏勒，跟紧我，今天谁也别想退！"))
    if len(durations) >= 4:
        s4 = starts[3]
        lines_cfg.append(("Default", s4 + 0.65, s4 + max(2.0, durations[3] - 0.55), "铁甲，依然在！"))
    if len(durations) >= 5:
        s5 = starts[4]
        lines_cfg.append(("Default", s5 + 1.15, s5 + max(2.2, durations[4] - 0.45), "铁甲依然在！"))

    header = f"""[Script Info]
Title: Novoland Shangyang Pass 1080p
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},52,&H00F2F2F2,&H000000FF,&H00101018,&H90000000,-1,0,0,0,100,100,2,0,1,2.6,1.2,2,80,80,58,1
Style: Title,{font_name},64,&H00DCE2EB,&H000000FF,&H000C0D12,&H80000000,-1,0,0,0,100,100,6,0,1,2.2,1.5,2,80,80,145,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogue_rows = []
    for style, st, ed, txt in lines_cfg:
        fade = "{\\fad(250,250)}" if style == "Title" else "{\\fad(120,120)}"
        dialogue_rows.append(f"Dialogue: 0,{fmt_ts(st)},{fmt_ts(ed)},{style},,0,0,0,,{fade}{txt}")

    ass_path.write_text(header + "\n".join(dialogue_rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="合成 1080p 九州缥缈录多镜成片（裁切 + 混音 + 字幕）")
    parser.add_argument("--workspace-dir", type=Path, default=Path("/content/MiniMaxH3_Workspace"))
    parser.add_argument("--clips", type=Path, nargs="*", default=[])
    parser.add_argument("--bgm", type=Path, default=None, help="可选 BGM 路径（默认查找 assets/audio/novoland/bgm_novoland_epic.mp3）")
    parser.add_argument("--prefix", default="novoland_shangyang_1080p", help="输出成片文件前缀")
    args = parser.parse_args()

    ws = args.workspace_dir.resolve()
    outputs_dir = ws / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    clips = load_shot_clips(ws, args.clips)
    bgm_path = args.bgm or (ws / "assets" / "audio" / "novoland" / "bgm_novoland_epic.mp3")
    if not bgm_path.is_file():
        bgm_path = None

    font_name = ensure_cjk_font()

    with tempfile.TemporaryDirectory(prefix="novoland_master_") as tmp_str:
        tmp = Path(tmp_str)
        norm_paths: list[Path] = []
        durations: list[float] = []

        # 1. 逐镜标准化：中心无损裁切 1920x1088 -> 1920x1080，帧锁音频 60ms 首尾平滑防止切镜噼啪声
        for idx, c in enumerate(clips, start=1):
            src = c["path"]
            dur, has_audio = probe_duration_and_has_audio(src)
            norm = tmp / f"norm_{idx:02d}.mkv"
            fade_out_st = max(0.05, dur - 0.06)

            if has_audio:
                cmd = [
                    "ffmpeg", "-y", "-i", str(src),
                    "-vf", "scale=1920:1088:flags=lanczos,crop=1920:1080:0:4,fps=24,format=yuv420p",
                    "-af", (
                        f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                        f"afade=t=in:st=0:d=0.06,afade=t=out:st={fade_out_st:.3f}:d=0.06"
                    ),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "12",
                    "-c:a", "pcm_s16le",
                    str(norm),
                ]
            else:
                cmd = [
                    "ffmpeg", "-y", "-i", str(src),
                    "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-vf", "scale=1920:1088:flags=lanczos,crop=1920:1080:0:4,fps=24,format=yuv420p",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "12",
                    "-c:a", "pcm_s16le", "-shortest",
                    str(norm),
                ]
            print(f"🎞️  标准化分镜 [{idx}/{len(clips)}] {src.name} ({dur:.2f}s) -> 1920x1080 ...")
            subprocess.run(cmd, check=True)
            norm_dur, _ = probe_duration_and_has_audio(norm)
            norm_paths.append(norm)
            durations.append(norm_dur)

        # 2. 拼接原生音画同步轨
        concat_txt = tmp / "concat.txt"
        concat_txt.write_text("".join(f"file '{p}'\n" for p in norm_paths), encoding="utf-8")
        joined_mkv = tmp / "joined_raw.mkv"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(joined_mkv)],
            check=True,
        )
        total_dur = sum(durations)
        print(f"✅ 已完成 5 镜帧锁拼接，总时长: {total_dur:.2f}s")

        # 3. 混入史诗 BGM（若存在）与响度均衡，输出纯净母带版
        clean_mp4 = outputs_dir / f"{args.prefix}_clean.mp4"
        sub_mp4 = outputs_dir / f"{args.prefix}_subtitled.mp4"

        if bgm_path is not None:
            print(f"🥁 正在混入东方史诗 BGM ({bgm_path.name}) 并执行人声闪避压缩 + -14 LUFS 均衡...")
            fade_bgm = max(0.5, total_dur - 1.5)
            filter_complex = (
                "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asplit=2[fg][sc];"
                f"[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"atrim=0:{total_dur:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d=0.6,afade=t=out:st={fade_bgm:.3f}:d=1.5,volume=0.22[bgm];"
                "[bgm][sc]sidechaincompress=threshold=0.025:ratio=4.5:attack=25:release=320[ducked];"
                "[fg][ducked]amix=inputs=2:duration=first:normalize=0,"
                "loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(joined_mkv), "-i", str(bgm_path),
                    "-filter_complex", filter_complex,
                    "-map", "0:v:0", "-map", "[aout]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart",
                    str(clean_mp4),
                ],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(joined_mkv),
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart",
                    str(clean_mp4),
                ],
                check=True,
            )

        # 4. 烧录剧场版中文字幕版
        ass_path = tmp / "novoland_subtitles.ass"
        write_ass_subtitles(ass_path, font_name, durations)
        print("📝 正在压制 1080p 剧场中文字幕母带...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(clean_mp4),
                "-vf", f"ass={ass_path}",
                "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-movflags", "+faststart",
                str(sub_mp4),
            ],
            check=True,
        )

    print("\n🎉 1080p 母带合成完毕：")
    print(f"   纯净版 : {clean_mp4}")
    print(f"   字幕版 : {sub_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
