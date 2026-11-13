#!/usr/bin/env python3
"""
generate_novoland_assets.py — 《九州缥缈录》Stage 1 & Stage 2 云端直存多模态资产工厂

功能：
  1. 可选读取原著章节文本 (--chapter-txt)，调用 gemini-3.8-flash 抽取角色/场景/名场面设定包；
     默认使用预构建考据设定包 (jobs/novoland_shangyang_1080p.pack.json)。
  2. 调用 Gemini Nano Banana 2.1 (`gemini-nano-banana-2.1`) 生成：
     - 2D 剧场版画风母版卡 (style_anchor_2d.png)
     - 角色立绘设定卡 (char_lvguichen.png / char_jiye.png / char_yingwuyi.png)
     - 无人物空镜场景卡 (scene_shangyang_pass.png / scene_blood_battlefield.png)
     - 多图参考融合生成 5 镜精确 1920x1088 首尾关键帧 (shot_01_start.png .. shot_05_end.png)
  3. 调用 `gemini-3.8-flash-tts` 生成双主角中文音色锚点 WAV (voice_jiye.wav / voice_lvguichen.wav)。
  4. 调用 `lyria-3-clip-preview` / `lyria-3.5` 生成 30 秒纯器乐史诗预告底乐 (bgm_novoland_epic.mp3)。
  5. 所有重资产直接落盘至 Google Drive (`MiniMaxH3_Workspace/assets/{images,audio}/novoland/`)，
     保持本地 Git 仓库绝对轻量化，绝不泄露或打印任何 API Key。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import certifi
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sync_drive  # type: ignore  # noqa: E402

TARGET_W = 1920
TARGET_H = 1088


def load_gemini_api_key() -> str:
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    for cand in (Path.home() / ".env", Path.home() / ".gemini" / ".env"):
        if cand.is_file():
            for line in cand.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    raise RuntimeError(
        "未检测到 GEMINI_API_KEY，请在 ~/.env 中配置后重试。"
    )


def load_image_part(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return {
        "type": "image",
        "mime_type": mime,
        "data": base64.b64encode(raw).decode("ascii"),
    }


def extract_output_bytes(interaction: Any, kind: str) -> bytes:
    prop = getattr(interaction, "output_image" if kind == "image" else "output_audio", None)
    if prop is not None and getattr(prop, "data", None):
        return base64.b64decode(prop.data)
    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", None) == "model_output":
            for block in getattr(step, "content", []) or []:
                if getattr(block, "type", None) == kind and getattr(block, "data", None):
                    return base64.b64decode(block.data)
    raise RuntimeError(f"Gemini 返回中未找到有效的 {kind} 输出数据 (status={getattr(interaction, 'status', '?')})")


def fit_to_exact_1080p_grid(raw_bytes: bytes, out_path: Path, width: int = TARGET_W, height: int = TARGET_H) -> None:
    """将 2K (2752x1536, ~16:9) 关键帧等比高质量重采样并中心微裁为严格的 1920x1088 PNG。"""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    cropped = resized.crop((left, top, left + width, top + height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path, format="PNG", optimize=True)


def generate_image_nb21(
    client: Any,
    prompt: str,
    ref_paths: list[Path],
    out_path: Path,
    exact_grid: bool = False,
) -> None:
    contents: list[dict[str, str]] = [load_image_part(p) for p in ref_paths]
    contents.append({"type": "text", "text": prompt})
    interaction = client.interactions.create(
        model="gemini-nano-banana-2.1",
        input=contents,
        response_format={
            "type": "image",
            "aspect_ratio": "16:9",
            "image_size": "2K",
        },
    )
    img_bytes = extract_output_bytes(interaction, "image")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if exact_grid:
        fit_to_exact_1080p_grid(img_bytes, out_path, TARGET_W, TARGET_H)
    else:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(out_path, format="PNG", optimize=True)


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 1)


def _post_generate_content(model: str, payload: dict[str, Any], timeout: int = 90) -> tuple[str, bytes]:
    """直连 Gemini v1beta generateContent REST 接口获取音频内联数据。"""
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for cand in data.get("candidates", []) or []:
        for part in (cand.get("content") or {}).get("parts", []) or []:
            inline = part.get("inlineData") or {}
            b64 = inline.get("data")
            if b64:
                return str(inline.get("mimeType") or ""), base64.b64decode(b64)
    raise RuntimeError(f"{model} 返回中未包含音频 inlineData: {list(data.keys())}")


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def generate_voice_tts(client: Any, item: dict[str, Any], out_path: Path) -> float:
    del client  # 使用 REST generateContent 以稳定兼容 TTS audio 输出
    last_err: Exception | None = None
    prompt_text = f"Speak with style ({item['style']}): {item['text']}"
    for model_id in ("gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts"):
        for voice_name in (item["voice"], item.get("fallback_voice", "Puck")):
            try:
                payload = {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}
                        },
                    },
                }
                mime, audio_bytes = _post_generate_content(model_id, payload, timeout=60)
                if "L16" in mime or "pcm" in mime.lower():
                    audio_bytes = _pcm_to_wav_bytes(audio_bytes, sample_rate=24000, channels=1)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(audio_bytes)
                dur = wav_duration_sec(out_path)
                if dur < 2.0:
                    raise RuntimeError(f"生成的音频过短 ({dur:.2f}s < 2.0s)")
                return dur
            except Exception as exc:
                last_err = exc
    raise RuntimeError(f"TTS 生成失败 ({item['id']}): {last_err}")


def generate_bgm_lyria(client: Any, bgm_cfg: dict[str, Any], out_path: Path) -> None:
    del client  # 使用 REST generateContent 以稳定兼容 Lyria 音频输出
    last_err: Exception | None = None
    for model_id in ("lyria-3-clip-preview", "lyria-3.5"):
        try:
            payload = {
                "contents": [{"parts": [{"text": bgm_cfg["prompt"]}]}],
                "generationConfig": {"responseModalities": ["AUDIO"]},
            }
            _, audio_bytes = _post_generate_content(model_id, payload, timeout=120)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_bytes)
            return
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Lyria BGM 生成失败: {last_err}")


def maybe_analyze_chapter_txt(client: Any, txt_path: Path, base_pack: dict[str, Any]) -> dict[str, Any]:
    raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")[:120000]
    prompt = (
        "你是资深2D动画导演。请阅读以下《九州缥缈录》章节片段，抽取其中出现的核心人物外貌/甲胄/兵器细节、"
        "关键战场场景视觉特征与最震撼的一句台词，以 JSON 返回：{summary, character_notes, iconic_line}。\n\n"
        + raw_text
    )
    resp = client.interactions.create(model="gemini-3.8-flash", input=prompt)
    out = dict(base_pack)
    out["chapter_analysis_notes"] = getattr(resp, "output_text", "") or ""
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="生成《九州缥缈录》2D史诗动漫全套图像/声线/配乐资产并直存 Google Drive")
    parser.add_argument("--pack", type=Path, default=REPO_ROOT / "jobs" / "novoland_shangyang_1080p.pack.json")
    parser.add_argument("--chapter-txt", type=Path, default=None, help="可选：原著章节 TXT 路径，启用 gemini-3.8-flash 章节分析")
    parser.add_argument("--only", nargs="*", default=[], help="仅重新生成指定 asset id（例如 shot_03_start voice_jiye）")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的云盘资产")
    parser.add_argument("--workspace-name", default="MiniMaxH3_Workspace")
    args = parser.parse_args()

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    api_key = load_gemini_api_key()
    os.environ["GEMINI_API_KEY"] = api_key

    from google import genai  # type: ignore

    client = genai.Client()
    pack = json.loads(args.pack.read_text(encoding="utf-8"))

    if args.chapter_txt is not None:
        if not args.chapter_txt.is_file():
            raise FileNotFoundError(f"找不到章节文本: {args.chapter_txt}")
        print(f"📖 Stage 1: 正在调用 gemini-3.8-flash 分析原著章节 {args.chapter_txt.name} ...")
        pack = maybe_analyze_chapter_txt(client, args.chapter_txt, pack)

    drive_root = sync_drive.find_local_mydrive()
    ws = drive_root / args.workspace_name
    sync_drive.ensure_workspace_layout(ws)

    img_dir = ws / "assets" / "images" / "novoland"
    aud_dir = ws / "assets" / "audio" / "novoland"
    img_dir.mkdir(parents=True, exist_ok=True)
    aud_dir.mkdir(parents=True, exist_ok=True)

    only_set = set(args.only)

    def should_run(asset_id: str, path: Path) -> bool:
        if only_set and asset_id not in only_set:
            return False
        if path.is_file() and not args.force and asset_id not in only_set:
            return False
        return True

    # 1. 画风母版卡
    style_path = img_dir / "style_anchor_2d.png"
    if should_run("style_anchor_2d", style_path):
        print("🎨 [1/4] 生成 2D 剧场版画风母版卡 -> style_anchor_2d.png ...")
        prompt = (
            f"{pack['style_lock']['master_style_prompt']}. "
            f"Palette: {pack['style_lock']['palette']}. "
            "Dramatic composition of ancient bronze armor silhouette and swirling winter embers under a dark winter sky, "
            "demonstrating razor-sharp 2D anime ink outlines and cel-shaded lighting."
        )
        generate_image_nb21(client, prompt, [], style_path, exact_grid=False)
    else:
        print("⏭️  跳过已存在: style_anchor_2d.png")

    # 2. 角色设定卡
    for ch in pack["characters"]:
        p = img_dir / f"{ch['id']}.png"
        if should_run(ch["id"], p):
            print(f"🧑 [2/4] 生成角色设定卡 ({ch['name']}) -> {p.name} ...")
            refs = [img_dir / f"{r}.png" for r in ch["refs"]]
            generate_image_nb21(client, ch["prompt"], refs, p, exact_grid=False)
        else:
            print(f"⏭️  跳过已存在: {p.name}")

    # 3. 无人物空镜卡
    for sc in pack["scenes"]:
        p = img_dir / f"{sc['id']}.png"
        if should_run(sc["id"], p):
            print(f"🏔️  [3/4] 生成无人物场景空镜 -> {p.name} ...")
            refs = [img_dir / f"{r}.png" for r in sc["refs"]]
            generate_image_nb21(client, sc["prompt"], refs, p, exact_grid=False)
        else:
            print(f"⏭️  跳过已存在: {p.name}")

    # 4. 5 镜精确 1920x1088 首尾关键帧（多图融合）
    for kf in pack["keyframes"]:
        p = img_dir / f"{kf['id']}.png"
        if should_run(kf["id"], p):
            print(f"🖼️  [4/4] 多图融合生成 1920x1088 分镜关键帧 -> {p.name} ...")
            refs = [img_dir / f"{r}.png" for r in kf["refs"]]
            generate_image_nb21(client, kf["prompt"], refs, p, exact_grid=True)
        else:
            print(f"⏭️  跳过已存在: {p.name}")

    # 5. 双主角声线参考 WAV
    for v in pack["voices"]:
        vp = aud_dir / f"{v['id']}.wav"
        if should_run(v["id"], vp):
            print(f"🎙️  [TTS] 生成角色音色参考 -> {vp.name} ...")
            dur = generate_voice_tts(client, v, vp)
            print(f"   ✅ {vp.name}: {dur:.2f}s")
        else:
            print(f"⏭️  跳过已存在: {vp.name} ({wav_duration_sec(vp):.2f}s)")

    # 6. 30 秒纯器乐史诗预告 BGM
    bgm_cfg = pack["bgm"]
    bgm_path = aud_dir / f"{bgm_cfg['id']}.mp3"
    if should_run(bgm_cfg["id"], bgm_path):
        print(f"🥁 [Lyria] 生成 30s 东方史诗配乐 -> {bgm_path.name} ...")
        generate_bgm_lyria(client, bgm_cfg, bgm_path)
    else:
        print(f"⏭️  跳过已存在: {bgm_path.name}")

    print("\n✅ 全部《九州缥缈录》多模态基础资产已直存 Google Drive：")
    print(f"   图像目录: {img_dir}")
    print(f"   音频目录: {aud_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
