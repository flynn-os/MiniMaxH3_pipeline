# AGENTS.md — MiniMax-H3 AI-Automated Pipeline Rules

## Architecture & Role Split
- **Local Control Plane (macOS + AI Agent, 0 CU & ~0 MB Local Disk)**: Storyboarding, multimodal H3 prompt engineering, resolution/frame-grid validation, and lightweight JSON queue compilation (`scripts/build_queue.py`).
- **Google Drive Cloud Storage Plane (`MiniMaxH3_Workspace/`, Stream Mode)**: Holds all heavy files—30GB+ INT8 checkpoints & LoRAs (`cache/`), reference images/videos/audio (`assets/`), pending queues (`inbox/`), and rendered MP4 outputs (`outputs/`).
- **Cloud Compute Plane (Google Colab `A100 40GB`, ~5.37 CU/hr)**: Headless batch execution via `deepbeepmeep/WanGP` (`shared.api` in `scripts/colab_runner.py`) and automatic `runtime.unassign()` upon completion.

## Mandatory Conventions for Agents
1. **Zero Local Media Hoarding**: Never store heavy reference images, videos, audio, or rendered `.mp4` files inside the local repository. Stage input media into Google Drive via `python3 scripts/sync_drive.py stage-asset <file> --move` and reference them via `drive://assets/...` URIs. Use `python3 scripts/sync_drive.py pull` (which fetches only tiny `run_manifest_*.json` reports by default) and stream outputs directly from `./drive_workspace/outputs/` without copying `.mp4` files onto the local SSD.
2. **Prompt Formatting**: Never feed raw one-line user prompts directly into MiniMax-H3 tasks. Always follow `.agents/skills/h3-prompt-writing/SKILL.md` (`base-en.txt` for `fl2va_*`, `ref-en.txt` for `ref2va_*`) so visual description, `overall_soundscape`, `non_diegetic_music`, and `<d>[Language] ...</d>` lip-sync dialogue are structured properly.
3. **Cloud Prompt Enhancer Off**: Keep `"prompt_enhancer": ""` in compiled tasks unless the user explicitly asks for cloud-side LLM rewriting—this avoids loading an extra 27B GGUF LLM into Colab RAM/VRAM.
4. **Strict Compilation**: Always compile storyboards via `python3 scripts/build_queue.py <storyboard.json> --strict-prompt` so 32-pixel resolution multiples and `17*n + 5` frame counts (`124..362`) are guaranteed valid before touching paid GPU compute.
5. **Colab Compute Hygiene**: Default `--auto-disconnect` must stay enabled in batch mode so Colab never sits idle draining Compute Units.
