#!/usr/bin/env python3
"""
launch_colab_cli.py — 基于本机 `colab` CLI (google-colab-cli) 的 100% 免浏览器无头渲染器

全自动零人工交互流程：
  1. `colab new -s <session> --gpu A100` 申请云端 A100 (40GB) 实例
  2. 默认使用云端高速 NVMe (`/content/MiniMaxH3_Workspace`) 免交互初始化与渲染（若传 `--mount-drive` 则尝试挂载云端 `/content/drive`）：
     - 若任务单引用了 `drive://assets/...`，自动从本机挂载的 Google Drive 按需流式上传所需参考素材至云端实例
  3. 渲染完成后，通过 `colab download` 将成片与 `run_manifest_*.json` 直接写入本机 Google Drive 云盘目录 (`MiniMaxH3_Workspace/outputs/`)
  4. 调用 `sync_drive.py evict` 释放本机 FileProvider 缓存块（保持 Mac 物理磁盘 0B 占用）
  5. `finally` 块强制执行 `colab stop -s <session>` 立即销毁 A100 实例停止计费！
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sync_drive  # type: ignore  # noqa: E402


def find_colab_bin() -> str:
    found = shutil.which("colab")
    if found:
        return found
    cand = Path.home() / ".local" / "bin" / "colab"
    if cand.is_file():
        return str(cand)
    raise FileNotFoundError("未找到 `colab` CLI，请先安装: uv tool install google-colab-cli")


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, check=check)


def exec_remote_python(colab_bin: str, session: str, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [colab_bin, "exec", "-s", session],
        input=code,
        text=True,
        capture_output=False,
        check=False,
    )


def upload_referenced_drive_assets(colab_bin: str, session: str, job_path: Path, local_ws: Path, remote_ws: str) -> None:
    """若免挂载模式下任务引用了 drive://assets/...，自动按需流式上传对应素材到云端实例。"""
    if job_path.suffix.lower() != ".json":
        return
    raw = json.loads(job_path.read_text(encoding="utf-8"))
    tasks = raw if isinstance(raw, list) else [raw]
    uris: set[str] = set()
    for t in tasks:
        for k in ("image_start", "image_end", "video_source", "audio_guide", "audio_guide2"):
            v = t.get(k)
            if isinstance(v, str) and v.startswith("drive://"):
                uris.add(v[len("drive://"):].lstrip("/"))
        for ref in t.get("image_refs", []) or []:
            if isinstance(ref, str) and ref.startswith("drive://"):
                uris.add(ref[len("drive://"):].lstrip("/"))

    for rel in sorted(uris):
        local_file = local_ws / rel
        if local_file.is_file():
            remote_target = f"{remote_ws}/{rel}"
            exec_remote_python(
                colab_bin,
                session,
                f"import pathlib; pathlib.Path({remote_target!r}).parent.mkdir(parents=True, exist_ok=True)",
            )
            run_cmd([colab_bin, "upload", "-s", session, str(local_file), remote_target], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="通过 colab CLI 免浏览器全无头启动 A100 渲染并归档至 Google Drive")
    parser.add_argument("--job", type=Path, required=True, nargs="+", help="待渲染的 .wangp.json 任务单路径")
    parser.add_argument("--session", "-s", default=f"h3-{int(time.time()) % 100000}", help="Colab CLI 会话名")
    parser.add_argument("--gpu", default="A100", choices=["A100", "L4", "G4", "H100", "T4"], help="GPU 型号（默认 A100）")
    parser.add_argument("--mount-drive", action="store_true", help="在云端 VM 内执行交互式 colab drivemount（需终端回车授权）")
    parser.add_argument("--workspace-name", default="MiniMaxH3_Workspace", help="Google Drive 工作区目录名")
    parser.add_argument("--keep-alive", action="store_true", help="跑完后保留实例不销毁（默认立即销毁省 CU）")
    args = parser.parse_args()

    colab_bin = find_colab_bin()
    drive_root = sync_drive.find_local_mydrive()
    local_ws = drive_root / args.workspace_name
    sync_drive.ensure_workspace_layout(local_ws)

    session_created = False
    try:
        run_cmd([colab_bin, "new", "-s", args.session, "--gpu", args.gpu], check=True)
        session_created = True

        if args.mount_drive:
            run_cmd([colab_bin, "drivemount", "-s", args.session, "/content/drive"], check=True)
            remote_ws = f"/content/drive/MyDrive/{args.workspace_name}"
        else:
            remote_ws = "/content/MiniMaxH3_Workspace"

        bootstrap_code = f"""
import os, subprocess, pathlib
pipeline = pathlib.Path("/content/MiniMaxH3_pipeline")
if not pipeline.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/flynn-os/MiniMaxH3_pipeline.git", str(pipeline)], check=True)
env = {{**os.environ, "WORKSPACE_DRIVE": {remote_ws!r}, "WANGP_DIR": "/content/WanGP"}}
subprocess.run(["bash", str(pipeline / "scripts" / "colab_bootstrap.sh")], env=env, check=True)
"""
        if exec_remote_python(colab_bin, args.session, bootstrap_code).returncode != 0:
            return 1

        remote_jobs: list[str] = []
        for j in args.job:
            j_path = j.resolve()
            if not j_path.is_file():
                raise FileNotFoundError(f"任务单不存在: {j_path}")
            if not args.mount_drive:
                upload_referenced_drive_assets(colab_bin, args.session, j_path, local_ws, remote_ws)
            remote_j = f"{remote_ws}/inbox/{j_path.name}"
            run_cmd([colab_bin, "upload", "-s", args.session, str(j_path), remote_j], check=True)
            remote_jobs.append(remote_j)

        render_code = f"""
import os, subprocess, sys
rc = subprocess.call([
    sys.executable, "/content/MiniMaxH3_pipeline/scripts/colab_runner.py",
    "--mode", "batch",
    "--workspace-dir", {remote_ws!r},
    "--no-auto-disconnect",
    "--job", *{remote_jobs!r},
])
os.sync()
if rc != 0:
    raise SystemExit(rc)
"""
        if exec_remote_python(colab_bin, args.session, render_code).returncode != 0:
            return 1

        if not args.mount_drive:
            list_out = subprocess.run(
                [colab_bin, "exec", "-s", args.session],
                input=(
                    "import json, pathlib\n"
                    f"ws = pathlib.Path({remote_ws!r})\n"
                    "files = [str(p) for p in sorted((ws/'outputs').glob('*')) if p.is_file()]\n"
                    "print('H3_FILES:' + json.dumps(files))\n"
                ),
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            for line in list_out.splitlines():
                if line.startswith("H3_FILES:"):
                    for rf in json.loads(line[len("H3_FILES:"):]):
                        fname = Path(rf).name
                        sub = "reports" if fname.startswith("run_manifest_") else "outputs"
                        dst = local_ws / sub / fname
                        run_cmd([colab_bin, "download", "-s", args.session, rf, str(dst)], check=True)
                        if sub == "reports":
                            local_rep = REPO_ROOT / "jobs" / "reports"
                            local_rep.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(dst, local_rep / fname)

            sync_drive.evict_cloud_items([local_ws / "outputs", local_ws / "assets"])
        return 0
    finally:
        if session_created and not args.keep_alive:
            print(f"\n🔌 正在通过 colab stop 释放云端 GPU 会话 '{args.session}'，停止计费...")
            run_cmd([colab_bin, "stop", "-s", args.session], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
