#!/usr/bin/env python3
"""
launch_colab_cli.py — 基于本机 `colab` CLI (google-colab-cli) 的全无头抗断连渲染器

架构亮点（彻底规避本机代理 WebSocket 10s 超时与断连风险）：
  1. `colab new -s <session> --gpu A100` 申请云端 A100 (40GB) 实例
  2. 通过纯 HTTPS REST (`colab upload`) 上传任务单、按需云盘参考素材与后台守护脚本
  3. 使用带 60s 握手补丁与 `os._exit(0)` 的单次启动器在云端拉起独立守护进程 (`start_new_session=True`)
  4. 全程通过毫秒级纯 REST (`colab download`) 轮询 `/content/h3_worker.status` 与进度日志，无视本地网络抖动
  5. 渲染完成后直接将 MP4 与战报下载至 Google Drive `MiniMaxH3_Workspace/{outputs,reports}/`，驱逐本机缓存恢复 0B 占用，并在 `finally` 中强制 `colab stop` 释放 A100！
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sync_drive  # type: ignore  # noqa: E402


def clean_colab_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    return env


def find_colab_bins() -> tuple[str, str]:
    colab_bin = shutil.which("colab") or str(Path.home() / ".local" / "bin" / "colab")
    if not Path(colab_bin).is_file():
        raise FileNotFoundError("未找到 `colab` CLI，请先安装: uv tool install google-colab-cli")
    uv_py = Path.home() / ".local" / "share" / "uv" / "tools" / "google-colab-cli" / "bin" / "python"
    py_bin = str(uv_py) if uv_py.is_file() else sys.executable
    return colab_bin, py_bin


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, env=clean_colab_env(), check=check)


def spawn_detached_remote_worker(py_bin: str, session: str, script_remote: str) -> None:
    launcher_py = f"""
import os
from jupyter_kernel_client.wsclient import KernelWebSocketClient
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

orig_init = KernelWebSocketClient.__init__
def patched_init(self, *args, **kwargs):
    kwargs["timeout"] = 60.0
    orig_init(self, *args, **kwargs)
KernelWebSocketClient.__init__ = patched_init

s = state.get_session({session!r})
rt = ColabRuntime(s.url, s.token, session_name=s.name)
out = rt.execute_code('''
import subprocess
p = subprocess.Popen(
    ["/bin/bash", {script_remote!r}],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print("STARTED_PID:", p.pid)
''')
print(out)
os._exit(0)
"""
    subprocess.run([py_bin, "-c", launcher_py], text=True, env=clean_colab_env(), check=True)


def create_colab_session(colab_bin: str, session: str, gpu: str, high_mem_first: bool) -> None:
    """优先尝试申请高显存 (--high-mem，如 80GB A100)，若库存不足或失败则自动回退到标准同型 GPU（如 40GB A100）。"""
    if high_mem_first:
        try:
            print(f"🚀 尝试申请高显存 {gpu} (--high-mem，例如 80GB A100)...", flush=True)
            run_cmd([colab_bin, "new", "-s", session, "--gpu", gpu, "--high-mem"], check=True)
            print(f"✅ 成功申请 --high-mem {gpu} 会话 '{session}'！", flush=True)
            return
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  未能抢到 --high-mem {gpu} (exit {exc.returncode})，正在自动回退到标准 {gpu} (40GB)...", flush=True)
            subprocess.run([colab_bin, "stop", "-s", session], env=clean_colab_env(), capture_output=True, check=False)
    run_cmd([colab_bin, "new", "-s", session, "--gpu", gpu], check=True)


def upload_drive_rel_paths(colab_bin: str, session: str, rels: set[str], local_ws: Path, remote_ws: str) -> None:
    for rel in sorted(rels):
        local_file = local_ws / rel
        if local_file.is_file():
            run_cmd([colab_bin, "upload", "-s", session, str(local_file), f"{remote_ws}/{rel}"], check=True)


def upload_referenced_drive_assets(colab_bin: str, session: str, job_path: Path, local_ws: Path, remote_ws: str) -> None:
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
    upload_drive_rel_paths(colab_bin, session, uris, local_ws, remote_ws)


def main() -> int:
    parser = argparse.ArgumentParser(description="通过 colab CLI 免浏览器全无头启动 A100 渲染并归档至 Google Drive")
    parser.add_argument("--job", type=Path, required=True, nargs="+", help="待渲染的 .wangp.json 任务单路径")
    parser.add_argument("--session", "-s", default=f"h3-{int(time.time()) % 100000}", help="Colab CLI 会话名")
    parser.add_argument("--gpu", default="A100", choices=["A100", "L4", "G4", "H100", "T4"], help="GPU 型号（默认 A100）")
    parser.add_argument(
        "--high-mem-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="申请 A100 时优先尝试 --high-mem (80GB)，拉不到自动回退 40GB A100（默认开启）",
    )
    parser.add_argument("--extra-asset", action="append", default=[], help="额外需要同步至云端工作区的 drive:// 素材 URI")
    parser.add_argument("--assemble-novoland", action="store_true", help="渲染完所有分镜后直接在云端合成 1080p clean + subtitled 母带")
    parser.add_argument("--poll-seconds", type=int, default=20, help="REST 状态轮询间隔秒数")
    parser.add_argument("--workspace-name", default="MiniMaxH3_Workspace", help="Google Drive 工作区目录名")
    parser.add_argument("--keep-alive", action="store_true", help="跑完后保留实例不销毁（默认立即销毁省 CU）")
    args = parser.parse_args()

    colab_bin, py_bin = find_colab_bins()
    drive_root = sync_drive.find_local_mydrive()
    local_ws = drive_root / args.workspace_name
    sync_drive.ensure_workspace_layout(local_ws)

    remote_ws = "/content/MiniMaxH3_Workspace"
    session_created = False

    try:
        create_colab_session(colab_bin, args.session, args.gpu, high_mem_first=args.high_mem_first)
        session_created = True

        extra_rels: set[str] = set()
        for ea in args.extra_asset:
            if isinstance(ea, str) and ea.startswith("drive://"):
                extra_rels.add(ea[len("drive://"):].lstrip("/"))
        if extra_rels:
            upload_drive_rel_paths(colab_bin, args.session, extra_rels, local_ws, remote_ws)

        remote_jobs: list[str] = []
        for j in args.job:
            j_path = j.resolve()
            if not j_path.is_file():
                raise FileNotFoundError(f"任务单不存在: {j_path}")
            upload_referenced_drive_assets(colab_bin, args.session, j_path, local_ws, remote_ws)
            remote_j = f"/content/{j_path.name}"
            run_cmd([colab_bin, "upload", "-s", args.session, str(j_path), remote_j], check=True)
            remote_jobs.append(remote_j)

        for rel_sync in (
            "configs/presets.json",
            "scripts/colab_bootstrap.sh",
            "scripts/colab_runner.py",
            "scripts/assemble_master.py",
        ):
            local_sync = REPO_ROOT / rel_sync
            if local_sync.is_file():
                run_cmd([colab_bin, "upload", "-s", args.session, str(local_sync), f"/content/local_sync/{rel_sync}"], check=True)

        jobs_arg = " ".join(f'"{rj}"' for rj in remote_jobs)
        assemble_block = ""
        if args.assemble_novoland:
            assemble_block = f"""
if [ $RUN_RC -eq 0 ]; then
  python3 /content/MiniMaxH3_pipeline/scripts/assemble_master.py --workspace-dir "{remote_ws}"
  RUN_RC=$?
fi
"""

        worker_script = f"""#!/usr/bin/env bash
set -uo pipefail
echo "RUNNING" > /content/h3_worker.status
exec > /content/h3_worker.log 2>&1

if [ ! -d /content/MiniMaxH3_pipeline/.git ]; then
  git clone --depth 1 https://github.com/flynn-os/MiniMaxH3_pipeline.git /content/MiniMaxH3_pipeline
else
  git -C /content/MiniMaxH3_pipeline pull --ff-only || true
fi

if [ -d /content/local_sync ]; then
  cp -rf /content/local_sync/* /content/MiniMaxH3_pipeline/ || true
fi

WORKSPACE_DRIVE="{remote_ws}" WANGP_DIR="/content/WanGP" \\
  bash /content/MiniMaxH3_pipeline/scripts/colab_bootstrap.sh
BOOT_RC=$?
if [ $BOOT_RC -ne 0 ]; then
  echo "FAILED_BOOTSTRAP:${{BOOT_RC}}" > /content/h3_worker.status
  exit $BOOT_RC
fi

python3 /content/MiniMaxH3_pipeline/scripts/colab_runner.py \\
  --mode batch \\
  --workspace-dir "{remote_ws}" \\
  --wangp-dir "/content/WanGP" \\
  --no-auto-disconnect \\
  --job {jobs_arg}
RUN_RC=$?
{assemble_block}
ls -1 "{remote_ws}/outputs" > /content/h3_outputs.list 2>/dev/null || true
if [ $RUN_RC -eq 0 ]; then
  echo "SUCCESS" > /content/h3_worker.status
else
  echo "FAILED_RUNNER:${{RUN_RC}}" > /content/h3_worker.status
fi
exit $RUN_RC
"""
        with tempfile.TemporaryDirectory(prefix="h3_cli_") as tmp:
            tmp_dir = Path(tmp)
            local_sh = tmp_dir / "run_h3_worker.sh"
            local_sh.write_text(worker_script, encoding="utf-8")
            run_cmd([colab_bin, "upload", "-s", args.session, str(local_sh), "/content/run_h3_worker.sh"], check=True)
            spawn_detached_remote_worker(py_bin, args.session, "/content/run_h3_worker.sh")

            status_file = tmp_dir / "status.txt"
            log_file = tmp_dir / "worker.log"
            seen_lines = 0

            while True:
                time.sleep(max(5, args.poll_seconds))
                subprocess.run(
                    [colab_bin, "download", "-s", args.session, "/content/h3_worker.status", str(status_file)],
                    env=clean_colab_env(),
                    capture_output=True,
                    check=False,
                )
                subprocess.run(
                    [colab_bin, "download", "-s", args.session, "/content/h3_worker.log", str(log_file)],
                    env=clean_colab_env(),
                    capture_output=True,
                    check=False,
                )
                if log_file.is_file():
                    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    for ln in lines[seen_lines:]:
                        print(f"   [A100] {ln}", flush=True)
                    seen_lines = len(lines)

                state_txt = status_file.read_text(encoding="utf-8").strip() if status_file.is_file() else "RUNNING"
                if state_txt != "RUNNING":
                    if state_txt != "SUCCESS":
                        print(f"❌ 云端执行失败: {state_txt}", file=sys.stderr)
                        return 1
                    break

            out_list_file = tmp_dir / "outputs.list"
            run_cmd([colab_bin, "download", "-s", args.session, "/content/h3_outputs.list", str(out_list_file)], check=True)
            for fname in out_list_file.read_text(encoding="utf-8").splitlines():
                fname = fname.strip()
                if not fname:
                    continue
                sub = "reports" if fname.startswith("run_manifest_") else "outputs"
                dst = local_ws / sub / fname
                run_cmd([colab_bin, "download", "-s", args.session, f"{remote_ws}/outputs/{fname}", str(dst)], check=True)
                if sub == "reports":
                    local_rep = REPO_ROOT / "jobs" / "reports"
                    local_rep.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dst, local_rep / fname)

            sync_drive.evict_cloud_items([local_ws / "outputs", local_ws / "assets"])
            print(f"\n🎉 渲染完成并已归档至 Google Drive: {local_ws / 'outputs'}")
            return 0
    finally:
        if session_created and not args.keep_alive:
            print(f"\n🔌 正在通过 colab stop 释放云端 GPU 会话 '{args.session}'，停止计费...")
            run_cmd([colab_bin, "stop", "-s", args.session], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
