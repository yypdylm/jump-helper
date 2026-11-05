"""半自动跳一跳：截图选点 + 窗口核验 + 单次长按。"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import mac_input as mi
import aiming


def save_log(data, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"诊断记录：{path.resolve()}", flush=True)


def countdown(seconds, message):
    print(message, flush=True)
    for remaining in range(seconds, 0, -1):
        print(f"{remaining}…", flush=True)
        time.sleep(1)


def identify(items, point, window_id):
    target = mi.window_at(items, point)
    if target is None or target.layer != 0:
        raise RuntimeError("该位置不是普通应用窗口，请重新选择。")
    if window_id is not None:
        if target.id != window_id:
            raise RuntimeError(f"坐标处窗口为 {target.id}，不是指定的 {window_id}。")
    elif not any(s in (target.owner + " " + target.title).lower()
                 for s in ("wechat", "weixin", "微信", "跳一跳")):
        raise RuntimeError(f"该位置属于 {target.owner}，不是可识别的微信窗口。用 --diagnose 查看窗口。")
    print(f"目标窗口：{target.owner} / {target.title!r}，PID={target.pid}，窗口编号={target.id}", flush=True)
    return target


def select_points(game, angle, auto_player=True):
    import cv2
    origin = aiming.find_player(game) if auto_player else None
    automatic = origin is not None
    hovered = None
    chosen = None
    finished_at = None
    name = "Fixed-angle aim: click target | R: reset player | Esc: cancel"

    def on_mouse(event, x, y, flags, userdata):
        nonlocal origin, hovered, chosen, finished_at, automatic
        hovered = (x, y)
        if event == cv2.EVENT_RBUTTONUP:
            origin, chosen, finished_at, automatic = None, None, None, False
        if event == cv2.EVENT_LBUTTONUP:
            if origin is None:
                origin = (float(x), float(y))
                automatic = False
            elif chosen is None:
                try:
                    projected = aiming.project_target(origin, hovered, angle)
                    px, py = projected["point"]
                    if not (0 <= px < game.shape[1] and 0 <= py < game.shape[0]):
                        return
                    chosen = (hovered, projected)
                    finished_at = time.monotonic()
                except ValueError:
                    pass

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, on_mouse)
    print("已标记棋子底部，只需点击目标。R 或右键可改选棋子。" if automatic else
          "请点棋子底部，再点目标。目标将吸附到固定角度线。", flush=True)
    try:
        while True:
            display, _ = aiming.draw_guides(game, origin, chosen[0] if chosen else hovered, angle)
            cv2.imshow(name, display)
            key = cv2.waitKey(20) & 0xff
            if key == 27 or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("已取消。")
            if key in (ord("r"), ord("R")):
                origin, chosen, finished_at, automatic = None, None, None, False
            if chosen is not None and time.monotonic() - finished_at >= 0.35:
                return origin, chosen[0], chosen[1], automatic
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def select_jump(args):
    import cv2
    import numpy as np
    import pyautogui
    mi.require_permissions()
    countdown(args.screenshot_delay, "请露出主屏幕上的跳一跳窗口，即将自动截图。")
    before = mi.windows()
    target = aiming.game_window(before, args.window_id)
    screen_w, screen_h = pyautogui.size()
    if target.x < 0 or target.y < 0 or target.x + target.width > screen_w or target.y + target.height > screen_h:
        raise RuntimeError("请把完整游戏窗口放在主屏幕内。")
    for point in ((target.x + 10, target.y + 60),
                  (target.x + target.width - 10, target.y + target.height - 10),
                  (target.x + target.width / 2, target.y + target.height / 2)):
        top = mi.window_at(before, point)
        if top is None or top.id != target.id or top.pid != target.pid:
            raise RuntimeError("游戏窗口被遮挡，请先露出完整游戏画面。")
    screenshot = pyautogui.screenshot()
    sx, sy = screenshot.width / screen_w, screenshot.height / screen_h
    if abs(sx - sy) > 0.02:
        raise RuntimeError("截图比例异常，请把游戏放到主屏幕。")
    box = (round(target.x * sx), round(target.y * sy),
           round((target.x + target.width) * sx), round((target.y + target.height) * sy))
    raw = cv2.cvtColor(np.array(screenshot.crop(box)), cv2.COLOR_RGB2BGR)
    current = next((w for w in mi.windows() if w.id == target.id and w.pid == target.pid), None)
    if current is None or (current.x, current.y, current.width, current.height) != (target.x, target.y, target.width, target.height):
        raise RuntimeError("截图时窗口正在移动，请保持稳定后重试。")
    scale = min(1.0, 520 / raw.shape[1], min(780, screen_h - 120) / raw.shape[0])
    game = cv2.resize(raw, (round(raw.shape[1] * scale), round(raw.shape[0] * scale)))
    origin, clicked, projected, automatic = select_points(game, args.angle, args.auto_player)
    w, h = game.shape[1], game.shape[0]
    dx = projected["horizontal"]
    computed = args.coefficient * dx / w
    ms = mi.duration_ms(args.press_ms if args.press_ms is not None else computed)
    point = (target.x + target.width / 2, target.y + target.height * 0.75)
    print(f"系数 {args.coefficient:g}；固定角度 {args.angle:g}°；投影水平距离 {dx:.1f}/{w}；执行 {ms} ms。", flush=True)
    details = {"points": [origin, projected["point"]], "clicked_target": clicked,
               "projection": projected, "auto_player_used": automatic,
               "preview_size": [w, h], "game_pixel_size": [raw.shape[1], raw.shape[0]],
               "screenshot_size": [screenshot.width, screenshot.height],
               "screen_points": [screen_w, screen_h], "coefficient": args.coefficient,
               "computed_ms": computed, "region_mode": "game-window"}
    if args._select_output:
        directory = Path(args._select_output).parent
        marked, _ = aiming.draw_guides(game, origin, clicked, args.angle)
        for name, pixels in (("before.png", game), ("selection.png", marked)):
            if not cv2.imwrite(str(directory / name), pixels):
                raise RuntimeError("无法保存游戏选点截图，未执行跳跃。")
    return point, target, ms, details


def selection_worker(args):
    """Only this short-lived process may import OpenCV/create selection windows."""
    path = Path(args._select_output)
    try:
        point, target, ms, details = select_jump(args)
        payload = {"point": point, "target": asdict(target), "ms": ms,
                   "details": details, "selection_pid": os.getpid()}
        status = 0
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        payload = {"error": str(exc) or "用户取消选点。"}
        status = 1
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return status


def select_in_process(args):
    # Wait for process EXIT, not merely destroyAllWindows. The sender retains
    # the same no-OpenCV environment as the successful --probe path.
    with tempfile.TemporaryDirectory(prefix="jump-selection-") as tmp:
        output = Path(tmp) / "selection.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--_select-output", str(output),
                   "--coefficient", str(args.coefficient), "--delay", str(args.delay),
                   "--angle", str(args.angle), "--screenshot-delay", str(args.screenshot_delay),
                   "--auto-player" if args.auto_player else "--no-auto-player"]
        if args.press_ms is not None:
            command += ["--press-ms", str(args.press_ms)]
        if args.window_id is not None:
            command += ["--window-id", str(args.window_id)]
        completed = subprocess.run(command, check=False)
        if not output.exists():
            raise RuntimeError(f"截图选点进程未返回结果（退出码 {completed.returncode}）。未执行跳跃。")
        payload = json.loads(output.read_text(encoding="utf-8"))
        if completed.returncode or "error" in payload:
            raise RuntimeError(payload.get("error", "截图选点进程失败，未执行跳跃。"))
        # Keep the visual evidence after the temporary GUI handoff is cleaned up.
        files = [Path(tmp) / name for name in ("before.png", "selection.png")]
        if any(path.exists() for path in files):
            capture_dir = Path(args.log_dir).resolve() / datetime.now().strftime("%Y%m%d-%H%M%S-%f-images")
            capture_dir.mkdir(parents=True, exist_ok=False)
            for path in files:
                if path.exists():
                    shutil.copy2(path, capture_dir / path.name)
            payload["details"]["image_directory"] = str(capture_dir)
    details = payload["details"]
    details["selection_pid"] = payload["selection_pid"]
    details["selection_process_exited"] = True
    return tuple(payload["point"]), mi.Window(**payload["target"]), mi.duration_ms(payload["ms"]), details


def capture_after(target, details):
    if "image_directory" not in details:
        return {"status": "没有选点截图目录，跳过。"}
    path = Path(details["image_directory"]) / "after.png"
    # Screenshot dependencies remain outside the sender, as do GUI dependencies.
    time.sleep(1)
    command = [sys.executable, str(Path(__file__).with_name("capture_game.py")),
               str(target.id), str(target.pid), str(path)]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        if result.returncode:
            return {"error": result.stderr.strip() or result.stdout.strip(), "jump_already_sent": True}
        return {"image": str(path), "metadata": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc), "jump_already_sent": True}


def choose_press_point(mode, automatic, target, items, front_pid, current_cursor):
    if mode == "auto":
        return automatic
    # Match --probe: use the user's real pointer inside the game. Never
    # silently fall back to another window or to the previous automatic point.
    mi.validate_target(target, current_cursor, items, front_pid)
    # Exclude window chrome and the resize border. The user still chooses a
    # blank area within the actual game canvas, not a game button.
    x, y = current_cursor
    if not (target.x + 8 <= x < target.x + target.width - 8 and
            target.y + 60 <= y < target.y + target.height - 8):
        raise RuntimeError("鼠标还在标题栏或窗口边缘。请在倒计时内移到游戏画面的空白处，松开并保持不动。")
    return current_cursor


def load_settings(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name("settings.json")
    defaults = {"coefficient": 1700, "angle_degrees": 30, "auto_player": True,
                "screenshot_delay": 2, "focus_delay": 4}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or set(data) - set(defaults):
                raise ValueError("配置项名称不正确")
            defaults.update(data)
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取 {path}：{exc}") from exc
    for key in ("coefficient", "angle_degrees", "screenshot_delay", "focus_delay"):
        value = defaults[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"配置 {key} 必须是有限数字。")
    if defaults["coefficient"] <= 0 or not 5 <= defaults["angle_degrees"] <= 75:
        raise ValueError("coefficient 须为正数，angle_degrees 须在 5–75 度。")
    for key in ("screenshot_delay", "focus_delay"):
        if not isinstance(defaults[key], int) or not 1 <= defaults[key] <= 60:
            raise ValueError(f"{key} 须为 1–60 的整数。")
    if not isinstance(defaults["auto_player"], bool):
        raise ValueError("auto_player 须为 true 或 false。")
    return defaults


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    try:
        settings = load_settings()
    except ValueError as exc:
        p.error(str(exc))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--diagnose", action="store_true", help="只列出权限与窗口，不发送事件")
    mode.add_argument("--probe", type=mi.duration_ms, metavar="MS", help="倒计时后在鼠标所在处测试一次长按")
    mode.add_argument("--record", type=int, metavar="SECONDS", help="只监听鼠标事件，记录真实/模拟长按")
    mode.add_argument("--auto", action="store_true", help="全自动识别并连续跳跃，只需启动时切回游戏一次")
    p.add_argument("--backend", choices=("quartz", "cliclick"), default="quartz")
    p.add_argument("--press-ms", type=mi.duration_ms, help="截图模式固定按压时间")
    p.add_argument("--press-at", choices=("cursor", "auto"), default="cursor",
                   help="截图模式按压位置：cursor 原地长按（默认）；auto 使用截图换算位置")
    p.add_argument("--_select-output", help=argparse.SUPPRESS)
    p.add_argument("--coefficient", type=float, default=settings["coefficient"], help="距离系数，数值越大按压越久")
    p.add_argument("--angle", type=float, default=settings["angle_degrees"], help="向左上/右上延伸，与水平线的夹角")
    p.add_argument("--auto-player", action=argparse.BooleanOptionalAction, default=settings["auto_player"],
                   help="尝试自动定位棋子底部；识别不可靠时手动补点")
    p.add_argument("--screenshot-delay", type=int, default=settings["screenshot_delay"])
    p.add_argument("--delay", type=int, default=settings["focus_delay"], help="选点后的切回游戏时间")
    p.add_argument("--window-id", type=int, help="限定目标窗口；也可显式选择浏览器进行对照")
    p.add_argument("--dry-run", action="store_true", help="核验窗口和坐标，但不发送鼠标事件")
    p.add_argument("--loop", action="store_true", help="每跳后自动截下一张图；每次仍需点目标，Esc 退出")
    p.add_argument("--max-jumps", type=int, default=0, help="自动模式最多跳几次；0 为不限次数")
    p.add_argument("--settle", type=float, default=1.2, help="自动模式每跳后至少等待的秒数")
    p.add_argument("--log-profile", choices=("compact", "full"), default="compact",
                   help="自动模式日志：compact 保留最近 5 跳、定期样本和新类别；full 保留全部诊断")
    p.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not 1 <= args.delay <= 60:
        p.error("--delay 须在 1–60 秒之间")
    if not 1 <= args.screenshot_delay <= 60:
        p.error("--screenshot-delay 须在 1–60 秒之间")
    if not math.isfinite(args.angle) or not 5 <= args.angle <= 75:
        p.error("--angle 须为 5–75 度")
    if not math.isfinite(args.coefficient) or args.coefficient <= 0:
        p.error("--coefficient 须是正有限数")
    if args.record is not None and not 1 <= args.record <= 60:
        p.error("--record 须在 1–60 秒之间")
    if args.record is not None and args.dry_run:
        p.error("--record 不发送事件，无需 --dry-run")
    if args.loop and (args.diagnose or args.probe is not None or args.record is not None or args.dry_run):
        p.error("--loop 只用于正常截图选点模式")
    if not math.isfinite(args.settle) or not 0.8 <= args.settle <= 30:
        p.error("--settle 须为 0.8–30 秒")
    if not 0 <= args.max_jumps <= 10000:
        p.error("--max-jumps 须为 0–10000")
    if args.auto and (args.loop or args.press_ms is not None or not args.auto_player):
        p.error("--auto 已连续运行并自动计算时长，不能搭配 --loop、--press-ms 或 --no-auto-player")
    if args._select_output:
        return selection_worker(args)
    if args.auto:
        from auto_runner import run_auto
        return run_auto(args)
    report = {"permissions": mi.permissions(), "backend": args.backend,
              "sender_pid": os.getpid()}
    status = 0
    try:
        if args.diagnose:
            report.update(frontmost=mi.frontmost(), windows=[asdict(w) for w in mi.windows()])
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        elif args.record is not None:
            print(f"开始监听 {args.record} 秒，仅记录鼠标按下、拖动、松开。请在游戏里测试长按。", flush=True)
            report.update(mi.record_mouse(args.record))
        else:
            mi.require_permissions()
            if args.probe is not None:
                countdown(args.delay, "请点击游戏标题栏，再把鼠标移到游戏内容区，松开并保持不动。")
                point = mi.cursor()
                target = identify(mi.windows(), point, args.window_id)
                ms, details = args.probe, {}
            else:
                point, target, ms, details = select_in_process(args)
                message = ("选点进程已退出。请点游戏标题栏，然后把鼠标移入游戏画面的空白处；松开并保持不动。"
                           if args.press_at == "cursor" else
                           "选点进程已退出。请点游戏标题栏，松开鼠标，保持窗口稳定。")
                countdown(args.delay, message)
            report["sender_has_cv2"] = "cv2" in sys.modules
            report.update(target=asdict(target), point=point, requested_ms=ms, selection=details,
                          frontmost_before=mi.frontmost())
            current_windows = mi.windows()
            current = next((w for w in current_windows if w.id == target.id and w.pid == target.pid), None)
            report["current_target"] = None if current is None else asdict(current)
            report["target_before_relocation"] = asdict(target)
            report["point_before_relocation"] = point
            f = report["frontmost_before"]
            target, point = mi.relocate_target(target, point, current_windows, None if f is None else f["pid"])
            old = report["target_before_relocation"]
            translation = (target.x - old["x"], target.y - old["y"])
            report.update(target=asdict(target), point=point, window_translation=translation)
            if translation != (0, 0):
                print(f"游戏窗口平移了 {translation}，已重新定位按压点至 {point}，按压时间不变。", flush=True)
            report["automatic_point"] = point
            report["cursor_before_press"] = mi.cursor()
            mode = "cursor" if args.probe is not None else args.press_at
            report["press_point_mode"] = mode
            point = choose_press_point(mode, point, target, current_windows,
                                       None if f is None else f["pid"], report["cursor_before_press"])
            report["point"] = point
            print(f"在游戏内 ({point[0]:.1f}, {point[1]:.1f}) 长按 {ms} ms。", flush=True)
            mi.guard(target, point)
            if args.dry_run:
                report["result"] = "dry-run：目标校验通过，未发送鼠标事件。"
            else:
                report["result"] = mi.send_hold(point, ms, target, args.backend)
            report["frontmost_after"] = mi.frontmost()
            print(json.dumps(report["result"], ensure_ascii=False, indent=2), flush=True)
            if args.probe is None and not args.dry_run:
                print("跳跃已发送，正在保存游戏画面；请暂时不要重新开始或切换窗口。", flush=True)
                report["after_capture"] = capture_after(target, details)
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        status = 1
        report["error"] = str(exc) or "用户中断"
        print("停止：" + report["error"], file=sys.stderr, flush=True)
    finally:
        save_log(report, args.log_dir)
    return status


if __name__ == "__main__":
    while True:
        status = main()
        if status or "--loop" not in sys.argv:
            raise SystemExit(status)
