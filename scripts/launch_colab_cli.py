#!/usr/bin/env python3
"""
launch_colab_cli.py — 基于本地 `colab` CLI (google-colab-cli) 的全无头端到端云端渲染器

无需打开浏览器！在本机一条命令自动完成：
  1. 同步轻量任务单至 Google Drive 工作区（可选）
  2. `colab new -s <session> --gpu A100` 申请云端 A100 40GB 实例
  3. `colab drivemount -s <session>` 将 Google Drive 挂载至 `/content/drive`
  4. `colab exec` 执行 Bootstrap（缓存绑定 + WanGP 环境就绪）与批量音视频渲染
  5. `finally` 块强制执行 `colab stop -s <session>` 释放实例，100% 杜绝算力空转扣点！
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_colab_bin() -> str:
    found = shutil.which("colab")
    if found:
        return found
    cand = Path.home() / ".local" / "bin" / "colab"
    if cand.is_file():
        return str(cand)
    raise FileNotFoundError(
        "未找到 `colab` CLI，请先安装: uv tool install google-colab-cli"
    )


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, check=check)


def exec_remote_python(colab_bin: str, session: str, code: str) -> int:
    proc = subprocess.run(
        [colab_bin, "exec", "-s", session],
        input=code,
        text=True,
        check=False,
    )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="通过 colab CLI 全无头启动 A100 渲染完自动销毁实例")
    parser.add_argument("--job", type=Path, nargs="*", default=[], help="待投递并渲染的 .wangp.json 任务单")
    parser.add_argument("--session", "-s", default=f"h3-{int(time.time()) % 100000}", help="Colab CLI 会话名")
    parser.add_argument("--gpu", default="A100", choices=["A100", "L4", "G4", "H100", "T4"], help="GPU 型号（默认 A100）")
    parser.add_argument("--workspace-name", default="MiniMaxH3_Workspace", help="Google Drive 工作区目录名")
    parser.add_argument("--keep-alive", action="store_true", help="跑完后保留实例不销毁（默认跑完立即销毁省 CU）")
    args = parser.parse_args()

    colab_bin = find_colab_bin()

    # 若传入本地 job，先同步至 Google Drive 工作区
    sync_cmd = [sys.executable, str(REPO_ROOT / "scripts" / "sync_drive.py"), "push"]
    for j in args.job:
        sync_cmd += ["--job", str(j)]
    run_cmd(sync_cmd, check=True)

    session_created = False
    try:
        run_cmd([colab_bin, "new", "-s", args.session, "--gpu", args.gpu], check=True)
        session_created = True

        run_cmd([colab_bin, "drivemount", "-s", args.session, "/content/drive"], check=True)

        ws_remote = f"/content/drive/MyDrive/{args.workspace_name}"
        bootstrap_code = f"""
import subprocess, pathlib
ws = pathlib.Path({ws_remote!r})
pipeline = ws / "pipeline"
if not (pipeline / "scripts" / "colab_bootstrap.sh").is_file():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/flynn-os/MiniMaxH3_pipeline.git", str(pipeline)], check=True)
subprocess.run(
    ["bash", str(pipeline / "scripts" / "colab_bootstrap.sh")],
    env={{**__import__("os").environ, "WORKSPACE_DRIVE": str(ws), "WANGP_DIR": "/content/WanGP"}},
    check=True,
)
"""
        rc = exec_remote_python(colab_bin, args.session, bootstrap_code)
        if rc != 0:
            return rc

        render_code = f"""
import os, subprocess, sys
ws = {ws_remote!r}
runner = f"{{ws}}/pipeline/scripts/colab_runner.py"
rc = subprocess.call([
    sys.executable, runner,
    "--mode", "batch",
    "--workspace-dir", ws,
    "--no-auto-disconnect",
])
os.sync()
if rc != 0:
    raise SystemExit(rc)
"""
        rc = exec_remote_python(colab_bin, args.session, render_code)
        return rc
    finally:
        if session_created and not args.keep_alive:
            print(f"\n🔌 正在通过 colab stop 释放云端 GPU 会话 '{args.session}'，停止计费...")
            run_cmd([colab_bin, "stop", "-s", args.session], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
