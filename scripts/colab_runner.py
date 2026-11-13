#!/usr/bin/env python3
"""
colab_runner.py — MiniMax-H3 云端自动化执行引擎 (运行于 Google Colab A100 40GB)

支持三种工作模式：
  1. batch (默认，最省 CU 推荐)：
     通过 WanGP 进程内 `shared.api` 初始化一次模型，顺序跑完队列中的所有分镜任务
     （原生解析 `drive://assets/...` 云盘素材 URI），将 MP4 直接写入 Google Drive `outputs/`，
     生成轻量执行战报至 `reports/run_manifest_<timestamp>.json`，刷盘后自动调用
     `google.colab.runtime.unassign()` 销毁实例停止计费！
  2. mcp (实时 Agent 遥控模式)：
     启动 WanGP 原生 MCP Server v2.1 并自动拉起 Cloudflare Tunnel 公网 HTTPS 地址。
  3. webui (人工可视化调试模式)：
     启动带密码保护的 Gradio 界面 + Cloudflare Tunnel。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any


def detect_best_attention() -> str:
    """默认交由 WanGP 原生 auto 选择当前 GPU 实际可用且最快的核心（sage / flash / sdpa）。"""
    return "auto"


def detect_gpu_vram_and_disk() -> tuple[float, float]:
    """查询云端 GPU 总显存 (GiB) 与根目录剩余磁盘空间 (GiB)，不初始化 PyTorch CUDA 上下文。"""
    vram_gb = 40.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        vram_mib = float(out.strip().splitlines()[0])
        vram_gb = vram_mib / 1024.0
    except Exception:
        pass
    try:
        free_gb = shutil.disk_usage("/").free / (1024.0 ** 3)
    except Exception:
        free_gb = 0.0
    return vram_gb, free_gb


def detect_default_profile() -> str:
    """通过 nvidia-smi 查询显存：A100(40GB/80GB) 用 4，L4(24GB) 用 4.5 保安全。"""
    vram_gb, _ = detect_gpu_vram_and_disk()
    if vram_gb < 28.0:
        return "4.5"
    return "4"


def maybe_upgrade_task_for_80gb_a100(task: dict[str, Any], vram_gb: float, free_gb: float) -> dict[str, Any]:
    """若检测到 80GB A100 且实例磁盘充裕 (>=58GB)，自动把 pruned 20B 升级为 33B 满血模型。"""
    upgrade_map = {
        "minimax_h3_ref2va_pruned": "minimax_h3_ref2va",
        "minimax_h3_fl2va_pruned": "minimax_h3_fl2va",
    }
    model_type = str(task.get("model_type") or "")
    if vram_gb >= 74.0 and free_gb >= 58.0 and model_type in upgrade_map:
        task["model_type"] = upgrade_map[model_type]
    return task


def ensure_cloudflared() -> Path:
    bin_path = Path("/usr/local/bin/cloudflared")
    if bin_path.is_file() and os.access(bin_path, os.X_OK):
        return bin_path
    print("🌐 正在下载 cloudflared 安全隧道客户端...")
    subprocess.run(
        [
            "curl", "-fsSL",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-o", str(bin_path),
        ],
        check=True,
    )
    bin_path.chmod(0o755)
    return bin_path


def disconnect_colab_runtime(grace_seconds: int = 15) -> None:
    """刷盘等待 Google Drive FUSE 同步完毕后，主动释放 Colab GPU Runtime 防止白烧 CU。"""
    print(f"\n🛡️  正在同步 Google Drive 磁盘缓冲区，{grace_seconds} 秒后自动释放 Colab 计算单元...")
    try:
        os.sync()
    except AttributeError:
        pass
    time.sleep(max(1, grace_seconds))
    try:
        from google.colab import runtime  # type: ignore
        print("🔌 正在调用 google.colab.runtime.unassign() 断开实例，停止计费！")
        runtime.unassign()
    except ImportError:
        print("ℹ️  当前非 Google Colab 内核环境，跳过 runtime.unassign()。")


def resolve_media_ref(val: str, base_dir: Path, workspace_dir: Path) -> str:
    """支持 drive:// URI、工作区 assets/ 相对路径以及任务包内相对路径解析。"""
    if not val or val.startswith(("http://", "https://")):
        return val
    if val.startswith("drive://"):
        rel = val[len("drive://"):].lstrip("/")
        return str((workspace_dir / rel).resolve())
    if val.startswith("/"):
        return val
    for root in (base_dir, workspace_dir, workspace_dir / "assets"):
        cand = (root / val).resolve()
        if cand.exists():
            return str(cand)
    return val


def load_job_file(job_path: Path, work_dir: Path, workspace_dir: Path) -> list[dict[str, Any]]:
    """加载 .wangp.json 或解压 .bundle.zip 并将其中的 drive:// 或相对素材路径解析为云端绝对路径。"""
    if job_path.suffix.lower() == ".zip":
        extract_dir = work_dir / job_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(job_path, "r") as zf:
            zf.extractall(extract_dir)
        manifest_candidates = list(extract_dir.glob("*.wangp.json")) + list(extract_dir.glob("*.json"))
        if not manifest_candidates:
            raise FileNotFoundError(f"任务包 {job_path} 中未找到 JSON 清单")
        raw = json.loads(manifest_candidates[0].read_text(encoding="utf-8"))
        base_dir = extract_dir
    else:
        raw = json.loads(job_path.read_text(encoding="utf-8"))
        base_dir = job_path.parent

    tasks = raw if isinstance(raw, list) else [raw]
    media_keys = ("image_start", "image_end", "video_source", "audio_guide", "audio_guide2")

    for t in tasks:
        for k in media_keys:
            val = t.get(k)
            if isinstance(val, str) and val:
                t[k] = resolve_media_ref(val, base_dir, workspace_dir)
        if isinstance(t.get("image_refs"), list):
            t["image_refs"] = [
                resolve_media_ref(ref, base_dir, workspace_dir) if isinstance(ref, str) else ref
                for ref in t["image_refs"]
            ]
    return tasks


def run_batch_mode(args: argparse.Namespace) -> int:
    wangp_dir = args.wangp_dir.resolve()
    workspace_dir = args.workspace_dir.resolve()
    inbox_dir = workspace_dir / "inbox"
    outputs_dir = workspace_dir / "outputs"
    reports_dir = workspace_dir / "reports"
    logs_dir = workspace_dir / "logs"
    for d in (outputs_dir, reports_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.job:
        job_files = [Path(p).resolve() for p in args.job]
    else:
        job_files = sorted(inbox_dir.glob("*.wangp.json")) + sorted(inbox_dir.glob("*.bundle.zip"))

    if not job_files:
        print(f"ℹ️  未在 {inbox_dir} 中发现待执行的 *.wangp.json 或 *.bundle.zip 任务单。")
        return 0

    attention = args.attention if args.attention != "auto" else detect_best_attention()
    profile = args.profile if args.profile != "auto" else detect_default_profile()
    vram_gb, free_gb = detect_gpu_vram_and_disk()
    if vram_gb >= 74.0 and free_gb >= 58.0:
        print(f"🚀 检测到 80GB A100 ({vram_gb:.1f} GiB VRAM | {free_gb:.1f} GiB Free Disk)，自动解锁满血 33B 未剪枝模型！")
    else:
        print(f"🛡️  运行于 {vram_gb:.1f} GiB VRAM ({free_gb:.1f} GiB Free Disk) 安全高画质模式 (Pruned 20B INT8 + Profile {profile})")

    print(f"🎬 准备批量执行 {len(job_files)} 个任务文件 | Attention={attention} | mmgp Profile={profile}")

    sys.path.insert(0, str(wangp_dir))
    os.chdir(wangp_dir)
    os.environ.setdefault("HF_HOME", str(workspace_dir / "cache" / "hf"))

    from shared.api import init  # type: ignore

    cli_args = [
        "--profile", str(profile),
        "--output-dir", str(outputs_dir),
    ]
    if attention != "auto":
        cli_args = ["--attention", attention] + cli_args
    print(f"⚙️  初始化 WanGP Session: output_dir={outputs_dir} | cli_args={cli_args}")
    session = init(root=wangp_dir, output_dir=outputs_dir, cli_args=cli_args)

    run_started = dt.datetime.now(dt.timezone.utc)
    summary_records: list[dict[str, Any]] = []
    any_failed = False

    with tempfile.TemporaryDirectory(prefix="h3_batch_") as tmp_str:
        tmp_dir = Path(tmp_str)

        for jf_idx, jf in enumerate(job_files, start=1):
            print(f"\n==================== 任务单 [{jf_idx}/{len(job_files)}] {jf.name} ====================")
            file_failed = False
            try:
                tasks = load_job_file(jf, tmp_dir, workspace_dir)
            except Exception as exc:
                any_failed = True
                print(f"❌ 解析任务单失败 {jf}: {exc}")
                summary_records.append({"job_file": str(jf), "success": False, "error": str(exc)})
                continue

            for t_idx, raw_task in enumerate(tasks, start=1):
                raw_task = maybe_upgrade_task_for_80gb_a100(raw_task, vram_gb, free_gb)
                shot_id = raw_task.pop("_shot_id", f"shot_{t_idx:02d}")
                shot_title = raw_task.pop("_shot_title", "")
                t0 = time.time()
                print(
                    f"\n▶️  [{shot_id}] {shot_title} | model={raw_task.get('model_type')} "
                    f"| res={raw_task.get('resolution')} | frames={raw_task.get('video_length')} "
                    f"| steps={raw_task.get('num_inference_steps')}"
                )

                try:
                    job = session.submit_task(raw_task)
                    last_step_msg = ""
                    for event in job.events.iter(timeout=0.5):
                        if event.kind == "progress":
                            p = event.data
                            msg = f"   ⏳ {p.phase}: step {p.current_step}/{p.total_steps} ({p.progress}%)"
                            if msg != last_step_msg:
                                print(msg, flush=True)
                                last_step_msg = msg
                        elif event.kind in ("info", "status"):
                            txt = str(event.data or "").strip()
                            if txt:
                                print(f"   [{event.kind}] {txt}", flush=True)
                        elif event.kind == "error":
                            print(f"   [error] {event.data}", flush=True)
                        elif event.kind == "stream":
                            line = event.data.text.strip()
                            if (
                                line
                                and "%|" not in line
                                and "B/s]" not in line
                                and any(k in line.lower() for k in ("error", "warn", "saved", "traceback", "exception"))
                            ):
                                print(f"   [{event.data.stream}] {line}", flush=True)

                    res = job.result()
                    elapsed = round(time.time() - t0, 2)
                    files = [str(f) for f in getattr(res, "generated_files", [])]
                    if res.success and files:
                        drive_uris = [
                            f"drive://outputs/{Path(f).name}" for f in files
                        ]
                        print(f"✅ [{shot_id}] 完成 ({elapsed}s) -> {files}")
                        summary_records.append({
                            "job_file": jf.name,
                            "shot_id": shot_id,
                            "shot_title": shot_title,
                            "success": True,
                            "elapsed_sec": elapsed,
                            "generated_files": files,
                            "drive_uris": drive_uris,
                        })
                    else:
                        errs = [e.message for e in getattr(res, "errors", [])]
                        if not errs and not files:
                            errs = ["WanGP exited without producing output files (check [info]/[error] logs above)"]
                        print(f"❌ [{shot_id}] 渲染失败 ({elapsed}s): {errs}")
                        file_failed = True
                        any_failed = True
                        summary_records.append({
                            "job_file": jf.name,
                            "shot_id": shot_id,
                            "shot_title": shot_title,
                            "success": False,
                            "elapsed_sec": elapsed,
                            "errors": errs,
                        })
                except Exception as exc:
                    elapsed = round(time.time() - t0, 2)
                    tb = traceback.format_exc()
                    print(f"❌ [{shot_id}] 未捕获异常 ({elapsed}s):\n{tb}")
                    file_failed = True
                    any_failed = True
                    summary_records.append({
                        "job_file": jf.name,
                        "shot_id": shot_id,
                        "shot_title": shot_title,
                        "success": False,
                        "elapsed_sec": elapsed,
                        "errors": [str(exc)],
                    })

            if jf.parent == inbox_dir:
                dest_dir = inbox_dir / ("failed" if file_failed else "processed")
                dest_dir.mkdir(parents=True, exist_ok=True)
                ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = dest_dir / f"{jf.stem}_{ts_tag}{jf.suffix}"
                shutil.move(str(jf), str(dest_path))
                print(f"📁 已归档任务单 -> {dest_path}")

    stamp = run_started.strftime("%Y%m%d_%H%M%S")
    report = {
        "started_at_utc": run_started.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attention": attention,
        "profile": profile,
        "all_succeeded": not any_failed,
        "shots": summary_records,
    }
    report_text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    manifest_report = reports_dir / f"run_manifest_{stamp}.json"
    manifest_output = outputs_dir / f"run_manifest_{stamp}.json"
    manifest_report.write_text(report_text, encoding="utf-8")
    manifest_output.write_text(report_text, encoding="utf-8")
    print(f"\n📊 执行战报已写入 Google Drive: {manifest_report}")
    return 1 if any_failed else 0


def run_tunnel_server(args: argparse.Namespace, is_mcp: bool) -> int:
    wangp_dir = args.wangp_dir.resolve()
    workspace_dir = args.workspace_dir.resolve()
    outputs_dir = workspace_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    attention = args.attention if args.attention != "auto" else detect_best_attention()
    profile = args.profile if args.profile != "auto" else detect_default_profile()
    passphrase = args.password or os.environ.get("H3_PASS") or secrets.token_urlsafe(12)
    port = args.port or (8000 if is_mcp else 7860)

    cf_bin = ensure_cloudflared()
    tunnel_proc = subprocess.Popen(
        [str(cf_bin), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    public_url = None
    url_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 25
    while time.time() < deadline and tunnel_proc.stdout is not None:
        line = tunnel_proc.stdout.readline()
        if not line:
            break
        m = url_re.search(line)
        if m:
            public_url = m.group(0)
            break

    if not public_url:
        tunnel_proc.terminate()
        raise RuntimeError("未能获取 Cloudflare Tunnel 公网 URL，请检查网络连通性")

    print("\n" + "=" * 68)
    if is_mcp:
        print("🤖 WanGP MCP Server v2.1 已就绪（供本地 Agent 远程挂载）")
        print(f"   Public MCP Origin : {public_url}")
        print(f"   MCP Passphrase    : {passphrase}")
    else:
        print("🌐 WanGP 可视化 WebUI 已就绪")
        print(f"   Public URL        : {public_url}")
        print(f"   Login Password    : {passphrase}")
    print("   ⚠️  注意：交互服务运行期间会持续消耗 Colab CU，用完请手动断开运行时！")
    print("=" * 68 + "\n", flush=True)

    cmd = [
        sys.executable, "wgp.py",
        "--attention", attention,
        "--profile", str(profile),
        "--output-dir", str(outputs_dir),
    ]
    if is_mcp:
        cmd += [
            "--mcp",
            "--mcp-transport", "streamable-http",
            "--mcp-host", "127.0.0.1",
            "--mcp-port", str(port),
            "--mcp-auth",
            "--mcp-auth-password", passphrase,
            "--mcp-auth-url", public_url,
            "--mcp-async",
        ]
    else:
        cmd += [
            "--listen",
            "--server-port", str(port),
            "--auth",
            "--auth-password", passphrase,
            "--public-url", public_url,
        ]

    try:
        return subprocess.call(cmd, cwd=str(wangp_dir))
    finally:
        tunnel_proc.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniMax-H3 Colab A100 自动化运行器")
    parser.add_argument("--mode", choices=["batch", "mcp", "webui"], default="batch", help="运行模式（默认 batch 跑完即断连）")
    parser.add_argument("--job", nargs="*", help="指定要执行的 .wangp.json 或 .bundle.zip（不传则自动扫描 Drive inbox/）")
    parser.add_argument("--workspace-dir", type=Path, default=Path("/content/drive/MyDrive/MiniMaxH3_Workspace"))
    parser.add_argument("--wangp-dir", type=Path, default=Path("/content/WanGP"))
    parser.add_argument("--attention", default="auto", choices=["auto", "sage2", "flash", "sdpa"])
    parser.add_argument("--profile", default="auto", help="WanGP mmgp 显存档位（auto 默认 A100 用 4，L4 用 4.5）")
    parser.add_argument("--port", type=int, default=None, help="mcp 或 webui 本地监听端口")
    parser.add_argument("--password", type=str, default=None, help="mcp 或 webui 访问口令（不传则自动随机生成）")
    parser.add_argument(
        "--auto-disconnect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="batch 模式结束后是否自动调用 runtime.unassign() 释放 GPU（默认开启！）",
    )
    parser.add_argument("--grace-seconds", type=int, default=15, help="断连前等待 Google Drive 同步刷盘秒数")
    args = parser.parse_args()

    if args.mode == "mcp":
        return run_tunnel_server(args, is_mcp=True)
    if args.mode == "webui":
        return run_tunnel_server(args, is_mcp=False)

    exit_code = 0
    try:
        exit_code = run_batch_mode(args)
    finally:
        if args.auto_disconnect:
            disconnect_colab_runtime(grace_seconds=args.grace_seconds)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
