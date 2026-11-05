# WeChat Jump Helper

**A macOS automation project for the WeChat “Jump Jump” mini-game, built with classical computer vision and native mouse input.**

The program captures the game window, detects the player and a plausible next platform, calculates a jump using a calibrated geometric model, and sends a timed mouse press. After each jump it checks scene stability and landing evidence before attempting another. If the evidence is insufficient or ambiguous, it **stops rather than guessing**.

This is an **unofficial, experimental personal project**. It is not affiliated with WeChat or Tencent, and it is not a neural-network or deep-learning project.

![OpenCV platform detection and landing-point projection](assets/detection-demo.png)

*Example screenshot from the included computer-vision regression fixtures.*

[简体中文说明 / Chinese quick start](README.zh-CN.md)

## Highlights

- **Automatic target detection:** OpenCV/NumPy edge, color-region, material, shadow, shape, and geometric analysis; supports diamond-shaped and elliptical platform tops.
- **Calibrated aiming:** projects a target onto left-up/right-up rays at a configurable angle, then computes a hold duration based on normalized horizontal distance.
- **Safety-first decisions:** requires a unique plausible target and a valid landing area. Uncertain recognition, occlusion, stale frames, or an unsafe window/focus state prevent input.
- **Temporal verification:** compares consecutive frames, estimates camera movement, and checks whether the player landed on the previous target.
- **Native macOS input:** sends mouse down/up events through Quartz; keeps image processing in a separate process from the input sender.
- **Reproducible tests:** includes unit tests and representative game-frame fixtures for visual edge cases.

In local play sessions, the project has achieved **100+ consecutive automated jumps**. This is an observed result, **not a guarantee** across game versions, skins, window sizes, or machines.

## Requirements

- **macOS** with the desktop WeChat client and the “Jump Jump” mini-game available.
- **Python 3.11+ recommended** (a current CPython installation, `python3`, and `pip`).
- **Screen Recording** and **Accessibility** permission for the Terminal app used to run the script (macOS System Settings → Privacy & Security). The exact permission label varies by macOS version.
- A visible, unobstructed game window on your main display. Keep its size fixed during a run.

The project uses `opencv-python`, `numpy`, `PyAutoGUI`, and macOS `pyobjc` bindings; see [`requirements.txt`](requirements.txt). **It is macOS-specific** and is not currently validated on Windows or Linux.

## Installation

Clone the repository (or download it as a ZIP from GitHub), open Terminal in the project folder, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp settings.example.json settings.json
```

`settings.json` is personal/local configuration and is deliberately excluded from Git. The example is a starting point, not a calibration guarantee.

## Getting started

1. Open WeChat's Jump Jump mini-game and leave it on a ready-to-jump screen.
2. Grant Screen Recording and Accessibility permissions to your terminal application. macOS may require you to restart the app after granting them.
3. Check the detected window and permissions without sending mouse events:

   ```bash
   python jump.py --diagnose
   ```

4. Optionally preview automatic recognition **without sending a jump**:

   ```bash
   python jump.py --auto --dry-run
   ```

5. Run the automated controller:

   ```bash
   python jump.py --auto
   ```

During the startup countdown, bring the game window to the foreground, move the pointer to a **blank area inside the game canvas**, release the mouse, and leave it stationary. The program then attempts jumps autonomously.

**To stop:** press **Esc**, move the mouse, or switch away from the game window. Other uncertainty conditions also stop automatically. The program does **not** restart the game after a failure.

### Useful options

```bash
python jump.py --auto --max-jumps 3       # Stop after at most three jumps
python jump.py --auto --settle 1.8        # Wait longer after each jump
python jump.py --auto --log-profile full  # Retain more debugging evidence
python jump.py --loop                     # Assisted mode: click each target yourself
python jump.py --diagnose                 # Permissions/window info only
```

On macOS you can also run `./start-auto.command` after creating the virtual environment. If necessary, first run `chmod +x start-auto.command`. The Terminal commands above are the recommended route for first-time setup.

### Calibration

Edit your local `settings.json`. The sample values come from one tested environment and may need adjustment:

```json
{
  "coefficient": 1700,
  "angle_degrees": 30,
  "auto_player": true,
  "screenshot_delay": 2,
  "focus_delay": 4
}
```

The aiming model projects onto a ray at `angle_degrees` above the horizontal; for each jump:

```text
hold_ms ≈ coefficient × projected_horizontal_distance / captured_game_width
```

The coefficient is **not** universal: game window geometry and rendering can affect calibration. Use `--dry-run` for safe checks before live input. Explicit command-line parameters override the local settings.

## How it works

```text
Game window screenshot
        ↓
Player + platform candidate generation
        ↓
Shared geometric/material validation
        ↓
Unique target + safe landing projection
        ↓
Consecutive-frame stability + fresh-frame check
        ↓
Quartz timed mouse hold
        ↓
Camera-motion tracking + landing verification
        ↳ next jump, or stop safely
```

### Code map

| Component | Responsibility |
| --- | --- |
| `jump.py`, `auto_runner.py`, `auto_worker.py` | CLI, automatic run orchestration, recognition worker |
| `vision.py` | Ordered candidate-detection pipeline and public recognition API |
| `vision_pixels.py`, `vision_geometry.py`, `vision_validation.py` | Shared pixel, geometric, and candidate validation operations |
| `vision_edges.py`, `vision_materials.py`, `vision_small.py`, `vision_occlusion.py` | Platform candidates from boundaries, materials, small surfaces, and partial occlusion |
| `vision_tracking.py`, `vision_landing.py` | Frame stability, camera translation, and post-jump landing checks |
| `aiming.py`, `mac_input.py` | Fixed-angle aiming and guarded Quartz mouse input |
| `flight_recorder.py`, `log_store.py`, `data_manager.py` | Optional local diagnostics, run retention, and archival tools |
| `test_*.py`, `testdata/` | Regression assertions and captured sample frames |

The ordered detectors intentionally **do not** skip ambiguous results to pick a more convenient result from a later detector. This is a conservative safety contract, not just an implementation detail.

## Tests

With your macOS virtual environment active, from the repository root:

```bash
python -m unittest discover -s . -p 'test_*.py'
```

Tests cover aiming, recognition, material and shape failures, temporal/landing behavior, and simulated input safety. Many tests use captured game-frame fixtures in `testdata/`; they do not intentionally send real clicks. The full suite requires a working macOS/Quartz environment.

`regression_runner.py` also supports private, frozen historical replays. Its historical catalog/archive contains local run data and **is not distributed in this public repository**. The standalone unit-test command above is the public verification path.

## Privacy, safety, and limitations

- Runs locally. The project does not require an external image-recognition API and does not upload screenshots.
- Run screenshots and diagnostic logs are written under `logs/` and ignored by Git. Review them before sharing.
- Never disable window/focus, scene-stability, or landing checks just to force a jump.
- Unknown platform designs, changing visual effects, partial occlusions, overlapping windows, and ambiguous camera motion can still cause safe stops or mistakes.
- Test fixtures and recorded local success do not prove universal reliability.
- Use only where you are authorized, and respect the platform's applicable rules.

Screenshots in `testdata/` depict third-party game artwork and are included as functional computer-vision test fixtures. Their inclusion does not imply ownership of the game's branding or artwork.

## Acknowledgments

General ideas were informed by public projects such as [wangshub/wechat_jump_game](https://github.com/wangshub/wechat_jump_game) and [B5DX/pc_wechat_jump_auto](https://github.com/B5DX/pc_wechat_jump_auto). This project uses its own macOS-specific recognition and input pipeline; these references do not imply affiliation or code reuse.

## Usage rights

No open-source license is granted for this repository. All rights in the original source code and documentation are reserved by their respective authors. Third-party software and game artwork retain their respective rights.
