---
name: minimax-h3-director
description: Orchestrate end-to-end MiniMax-H3 audiovisual video generation workflows from natural-language ideas into validated WanGP batch queues for Google Colab A100 (40GB). Activate whenever the user asks to generate videos, plan shots/storyboards, prepare reference images/audio, or compile MiniMax-H3 job queues in this project.
---

# MiniMax-H3 Automated Director Skill

Use this workflow to turn a short creative brief into ready-to-render `*.wangp.json` tasks for the cloud Colab A100 runner with zero manual parameter tweaking and near-zero local disk usage.

## Step-by-Step Workflow

1. **Choose the right task & preset** (defined in `configs/presets.json`):
   - **Text / First-Frame / First-Last-Frame (`T2VA`, `I2VA`, `FL2VA`)**:
     - Final render: `fl2va_8step_quality` (`minimax_h3_fl2va_pruned` + LightX2V 8-step LoRA)
     - Quick draft: `fl2va_4step_fast` (DMAD 4-step LoRA)
   - **Identity / Video / Voice Reference (`Ref2VA`)**:
     - Standard 8-step: `ref2va_8step_quality`
     - High-detail 2-phase 2K IC finish: `ref2va_2k_ic` (keep shot duration 5–8s on 40GB A100)

2. **Stage heavy reference media on Google Drive (Zero Local Disk)**:
   - Never copy or store large images, videos, or audio inside the local git repository.
   - Move reference media directly into Google Drive (`MiniMaxH3_Workspace/assets/`):
     ```bash
     python3 scripts/sync_drive.py stage-asset /path/to/hero.png --subdir images --move
     ```
   - Reference them in storyboards using `drive://assets/images/hero.png` so `build_queue.py` validates metadata without downloading or zipping heavy files locally.

3. **Write native H3 audiovisual prompts**:
   - Read `.agents/skills/h3-prompt-writing/SKILL.md` and its `references/base-en.txt` (for `fl2va_*`) or `references/ref-en.txt` (for `ref2va_*`).
   - Always write descriptive sections in **English** while keeping spoken dialogue (`(S1) <d>[Chinese] ...</d>`), lyrics, and visible text in their **original language**.
   - Ensure the described action & dialogue comfortably fit the target `duration_seconds` (`5` to `15` seconds).

4. **Save Storyboard & Compile Queue**:
   - Write the structured storyboard to `jobs/<slug>.storyboard.json` (follow `jobs/demo_cyberpunk_ramen.storyboard.json`).
   - Run the compiler with strict validation:
     ```bash
     python3 scripts/build_queue.py jobs/<slug>.storyboard.json --strict-prompt
     ```

5. **Cloud Sync, Execution & Space Hygiene**:
   - Push the compiled lightweight `.wangp.json` into Google Drive `MiniMaxH3_Workspace/inbox/`:
     ```bash
     python3 scripts/sync_drive.py push --job jobs/<slug>.wangp.json
     ```
   - Once Colab finishes rendering, run `python3 scripts/sync_drive.py pull` to fetch lightweight `run_manifest_*.json` reports only, stream outputs directly from `./drive_workspace/outputs/`, and run `python3 scripts/sync_drive.py evict` whenever local CloudStorage cache needs to be reclaimed back to `0B`.
