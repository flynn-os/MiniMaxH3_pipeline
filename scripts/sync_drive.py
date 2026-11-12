#!/usr/bin/env python3
"""
sync_drive.py — Google Drive 云端优先 (Zero-Local-Disk) 工作区管理与同步工具

核心设计：
  - 重资产（模型权重 cache/、参考图音视频 assets/、生成成片 outputs/）100% 存储在 Google Drive。
  - 利用 macOS FileProvider Stream 模式，云端数 GB 视频在本机磁盘物理占用保持 0B，支持直接按需流式预览。
  - `pull` 默认仅拉取 ~1KB 的 `run_manifest_*.json` 执行战报，绝不把大体积 MP4 复制回本地仓库占用 SSD！
  - `stage-asset --move` 支持把本地图片/视频/音频迁入 Google Drive `assets/` 并立即释放本机空间。
  - `evict` 支持一键驱逐本地残留缓存块，让本机物理磁盘占用归零。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def fmt_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} TB"


def find_local_mydrive() -> Path:
    env_root = os.environ.get("MINIMAX_H3_DRIVE_ROOT")
    if env_root:
        p = Path(env_root).expanduser()
        if p.is_dir():
            return p

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if cloud_storage.is_dir():
        for gd in sorted(cloud_storage.glob("GoogleDrive-*")):
            for sub in ("My Drive", "我的云端硬盘", "マイドライブ"):
                cand = gd / sub
                if cand.is_dir():
                    return cand

    raise FileNotFoundError(
        "未检测到 Google Drive 挂载目录，请设置环境变量 MINIMAX_H3_DRIVE_ROOT 或传入 --drive-root。"
    )


def ensure_workspace_layout(workspace: Path) -> None:
    for sub in (
        "pipeline",
        "inbox/processed",
        "inbox/failed",
        "assets/images",
        "assets/videos",
        "assets/audio",
        "outputs",
        "reports",
        "cache/ckpts",
        "cache/loras",
        "cache/hf",
        "logs",
    ):
        (workspace / sub).mkdir(parents=True, exist_ok=True)

    link_path = REPO_ROOT / "drive_workspace"
    if link_path.is_symlink():
        if link_path.resolve() != workspace.resolve():
            link_path.unlink()
            link_path.symlink_to(workspace)
    elif not link_path.exists():
        link_path.symlink_to(workspace)


def folder_usage(path: Path) -> tuple[int, int, int]:
    """返回 (文件数, 云端逻辑总字节 st_size, 本机物理占用字节 st_blocks*512)，仅读 inode 不触发云端下载。"""
    if not path.is_dir():
        return 0, 0, 0
    count = 0
    logical = 0
    physical = 0
    for root, _, files in os.walk(path):
        for fname in files:
            if fname == ".DS_Store":
                continue
            fp = os.path.join(root, fname)
            try:
                st = os.lstat(fp)
                count += 1
                logical += st.st_size
                physical += getattr(st, "st_blocks", 0) * 512
            except OSError:
                pass
    return count, logical, physical


def evict_cloud_items(paths: list[Path]) -> int:
    """调用 macOS Foundation API 驱逐 FileProvider 本地缓存副本，仅保留云端 0B 在线占位符。"""
    if sys.platform != "darwin" or not shutil.which("swift"):
        return 0
    targets = [str(p) for p in paths if p.exists()]
    if not targets:
        return 0
    swift_code = (
        "import Foundation\n"
        "let fm = FileManager.default\n"
        "var evicted = 0\n"
        "for arg in CommandLine.arguments.dropFirst() {\n"
        "  let url = URL(fileURLWithPath: arg)\n"
        "  if let e = fm.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey]) {\n"
        "    for case let fileURL as URL in e {\n"
        "      if (try? fm.evictUbiquitousItem(at: fileURL)) != nil { evicted += 1 }\n"
        "    }\n"
        "  } else {\n"
        "    if (try? fm.evictUbiquitousItem(at: url)) != nil { evicted += 1 }\n"
        "  }\n"
        "}\n"
        "print(evicted)\n"
    )
    res = subprocess.run(
        ["swift", "-e", swift_code, *targets],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip().isdigit():
        return int(res.stdout.strip())
    return 0


def cmd_status(workspace: Path, workspace_name: str) -> int:
    ensure_workspace_layout(workspace)
    print(f"☁️  云端工作区映射 : ./drive_workspace -> {workspace}")
    print(f"🖥️  Colab 侧挂载路径: /content/drive/MyDrive/{workspace_name}\n")
    print(f"{'目录':16s} | {'文件数':>6s} | {'Google Drive 云端大小':>20s} | {'Mac 本地磁盘实际占用':>20s}")
    print("-" * 72)
    for sub in ("assets", "inbox", "outputs", "reports", "cache"):
        cnt, logical, physical = folder_usage(workspace / sub)
        print(f"{sub:16s} | {cnt:6d} | {fmt_bytes(logical):>20s} | {fmt_bytes(physical):>20s}")
    return 0


def cmd_push(workspace: Path, jobs: list[Path]) -> int:
    ensure_workspace_layout(workspace)
    pipeline_dst = workspace / "pipeline"

    for rel_dir in ("configs", "scripts", "notebooks", "jobs"):
        src_dir = REPO_ROOT / rel_dir
        if src_dir.is_dir():
            dst_dir = pipeline_dst / rel_dir
            shutil.copytree(
                src_dir,
                dst_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("reports", "*.mp4", "*.wav", "*.bundle.zip", ".DS_Store"),
            )

    for rel_file in ("README.md", "AGENTS.md", "LICENSE"):
        src_f = REPO_ROOT / rel_file
        if src_f.is_file():
            shutil.copy2(src_f, pipeline_dst / rel_file)

    print(f"✅ 已同步轻量工程与 Notebook 至 Google Drive: {pipeline_dst}")

    inbox_dst = workspace / "inbox"
    for j in jobs:
        if not j.is_file():
            print(f"❌ 找不到任务单文件: {j}", file=sys.stderr)
            return 1
        target = inbox_dst / j.name
        shutil.copy2(j, target)
        print(f"📥 已投递待渲染任务单 -> {target}")

    return 0


def infer_asset_subdir(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        return "images"
    if ext in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        return "videos"
    if ext in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
        return "audio"
    return "images"


def cmd_stage_asset(workspace: Path, files: list[Path], subdir: str | None, move: bool) -> int:
    ensure_workspace_layout(workspace)
    if not files:
        print("❌ 请通过 --file 指定至少一个要上云的素材文件。", file=sys.stderr)
        return 1

    staged_paths: list[Path] = []
    for f in files:
        if not f.is_file():
            print(f"❌ 素材文件不存在: {f}", file=sys.stderr)
            return 1
        sub = subdir or infer_asset_subdir(f)
        dest_dir = workspace / "assets" / sub
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.name
        if move:
            shutil.move(str(f), str(dest))
            action_label = "已迁入云盘并删除本地副本"
        else:
            shutil.copy2(f, dest)
            action_label = "已上传至云盘"
        staged_paths.append(dest)
        print(f"☁️  {action_label}: {dest}")
        print(f"   👉 Storyboard 引用 URI: drive://assets/{sub}/{f.name}")

    return 0


def cmd_pull(workspace: Path, download_media: bool) -> int:
    ensure_workspace_layout(workspace)
    reports_src = workspace / "reports"
    outputs_src = workspace / "outputs"
    local_reports = REPO_ROOT / "jobs" / "reports"
    local_reports.mkdir(parents=True, exist_ok=True)

    synced_reports = 0
    for src_dir in (reports_src, outputs_src):
        if not src_dir.is_dir():
            continue
        for item in sorted(src_dir.glob("run_manifest_*.json")):
            dst = local_reports / item.name
            if not dst.exists() or item.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(item, dst)
                synced_reports += 1
                print(f"📊 已同步轻量战报: {dst}")

    media_items = sorted(
        p for p in outputs_src.iterdir()
        if p.is_file() and p.suffix.lower() in (".mp4", ".wav", ".png")
    ) if outputs_src.is_dir() else []

    if not download_media:
        print(f"\n🎬 云端成片目录 (共 {len(media_items)} 个文件，保留在 Google Drive 不占本机空间):")
        for m in media_items[-10:]:
            st = os.lstat(m)
            local_used = getattr(st, "st_blocks", 0) * 512
            print(f"   - ./drive_workspace/outputs/{m.name}  (云端 {fmt_bytes(st.st_size)} | 本地占用 {fmt_bytes(local_used)})")
        if media_items:
            print("💡 提示: 可直接用 QuickTime / IINA / Finder 空格键流式播放上述路径；若确需拷贝到本机磁盘再加 --download-media。")
        return 0

    local_out = REPO_ROOT / "outputs"
    local_out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in media_items:
        dst = local_out / item.name
        if not dst.exists() or item.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(item, dst)
            copied += 1
            print(f"⬇️  已下载至本机磁盘: {dst}")
    print(f"✅ 共下载 {copied} 个媒体文件至 {local_out}")
    return 0


def cmd_evict(workspace: Path) -> int:
    ensure_workspace_layout(workspace)
    reclaimed_local = 0
    local_out = REPO_ROOT / "outputs"
    if local_out.is_dir():
        for f in local_out.rglob("*"):
            if f.is_file():
                reclaimed_local += f.stat().st_size
        shutil.rmtree(local_out)
        print(f"🧹 已清理本地仓库 outputs/ 副本，释放 {fmt_bytes(reclaimed_local)}")

    for z in REPO_ROOT.glob("jobs/*.bundle.zip"):
        reclaimed_local += z.stat().st_size
        z.unlink()
        print(f"🧹 已清理本地离线压缩包: {z.name}")

    evicted_count = evict_cloud_items([
        workspace / "assets",
        workspace / "outputs",
        workspace / "cache",
    ])
    print(f"☁️  已向 macOS FileProvider 请求驱逐云盘本地缓存项 ({evicted_count} 项)，所有重资产仅保留在 Google Drive 云端。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Google Drive 云端优先 (Zero-Local-Disk) 同步与空间管理工具")
    parser.add_argument(
        "action",
        choices=["status", "push", "pull", "stage-asset", "evict"],
        help="status=查看云端与本地空间占用; push=同步轻量代码与任务单; pull=仅拉取JSON战报并列出云端流式视频; stage-asset=重素材上云; evict=释放本机缓存",
    )
    parser.add_argument("--job", type=Path, nargs="*", default=[], help="push 时投递进 inbox/ 的 .wangp.json")
    parser.add_argument("--file", type=Path, nargs="*", default=[], help="stage-asset 时要上云的本地图片/视频/音频")
    parser.add_argument("--subdir", choices=["images", "videos", "audio"], default=None, help="stage-asset 目标分类子目录")
    parser.add_argument("--move", action="store_true", help="stage-asset 上传后立即删除本地源文件以省空间")
    parser.add_argument("--download-media", action="store_true", help="pull 时强制将 MP4 拷贝回本机磁盘（默认关闭以节省 Mac 空间）")
    parser.add_argument("--drive-root", type=Path, default=None, help="手动指定 Google Drive 根目录")
    parser.add_argument("--workspace-name", default="MiniMaxH3_Workspace", help="Drive 下的工作区目录名")
    args = parser.parse_args()

    drive_root = args.drive_root or find_local_mydrive()
    workspace = drive_root / args.workspace_name

    if args.action == "status":
        return cmd_status(workspace, args.workspace_name)
    if args.action == "push":
        return cmd_push(workspace, args.job)
    if args.action == "stage-asset":
        return cmd_stage_asset(workspace, args.file, args.subdir, args.move)
    if args.action == "evict":
        return cmd_evict(workspace)
    return cmd_pull(workspace, args.download_media)


if __name__ == "__main__":
    raise SystemExit(main())
