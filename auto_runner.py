"""Automatic controller. No image/GUI imports in this input-sending process."""
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import aiming
import mac_input as mi
import vision
from log_store import RunLogStore


class AutoStopped(RuntimeError):
    pass


def abort_if_requested(anchor):
    if mi.Q.CGEventSourceKeyState(mi.Q.kCGEventSourceStateCombinedSessionState, 53):
        raise AutoStopped("已按 Esc 停止。")
    point = mi.cursor()
    if point[0] <= 2 and point[1] <= 2:
        raise AutoStopped("鼠标移到左上角，已停止。")
    if math.dist(point, anchor) > 12:
        raise AutoStopped("检测到鼠标移动，已停止自动操作。")


def wait_guarded(seconds, check):
    end = time.monotonic() + seconds
    while True:
        check()
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def recognize(target, point, angle, directory, step, check):
    request = directory / f"{step:04d}-request.json"
    output = directory / f"{step:04d}-scene.json"
    parameters = {"window": asdict(target), "press_point": point, "angle": angle}
    if step > 1:
        parameters["previous_scene"] = str(directory / f"{step - 1:04d}-scene.json")
    request.write_text(json.dumps(parameters), encoding="utf-8")
    command = [sys.executable, str(Path(__file__).with_name("auto_worker.py")), str(request), str(output)]
    with (directory / f"{step:04d}-worker.txt").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 23
            waiting_notice = False
            recovery_notice = False
            while process.poll() is None:
                check()
                if time.monotonic() > deadline:
                    raise AutoStopped("自动识别超时，已停止。")
                if not waiting_notice and time.monotonic() > deadline - 21:
                    print("正在等待可识别的稳定画面，无需重新启动…", flush=True)
                    waiting_notice = True
                if not recovery_notice and time.monotonic() > deadline - 15:
                    print("正在自动重新核对当前局面，确认站稳和目标后继续…", flush=True)
                    recovery_notice = True
                time.sleep(0.05)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if not output.exists():
        raise AutoStopped("自动识别进程未返回结果，未发送跳跃。")
    payload = json.loads(output.read_text(encoding="utf-8"))
    if process.returncode or "error" in payload:
        raise AutoStopped(payload.get("error", "自动识别失败，未发送跳跃。"))
    if payload.get("recovery"):
        print("重新核对通过，继续自动跳跃。", flush=True)
    return payload


def validate_decision(payload, previous, coefficient, angle):
    if time.monotonic() - payload["captured_at"] > 1.0:
        raise AutoStopped("识别画面已经过时，未发送跳跃。")
    scene = payload["scene"]
    if vision.unchanged_scene(previous, scene):
        # Similar consecutive layouts can have nearly identical masked
        # signatures. A verified landing plus tracked camera motion is stronger
        # evidence of progress; a missing/failed/zero-motion check cannot bypass.
        landing = payload.get("landing_check", {})
        offset = landing.get("camera_offset", ())
        progressed = (landing.get("verified") is True and len(offset) == 2 and
                      all(math.isfinite(v) for v in offset) and
                      math.hypot(*offset) > scene["image_size"][0] * 0.02)
        if not progressed:
            raise AutoStopped("上一跳后画面没有明显变化，已停止，避免重复按压。")
    projection = aiming.project_target(scene.get("aim_origin", scene["origin"]), scene["target"], angle)
    return mi.duration_ms(coefficient * projection["horizontal"] / scene["image_size"][0])


def settle_with_evidence(directory, step, ms, seconds, check):
    """Capture short-jump outcomes only after mouse-up, without GUI imports here."""
    if ms >= 300:
        wait_guarded(seconds, check)
        return
    released_at = time.monotonic()
    output = directory / f"{step:04d}-flight.json"
    command = [sys.executable, str(Path(__file__).with_name("flight_recorder.py")),
               str(directory / f"{step:04d}-request.json"), str(output), str(released_at)]
    process = None
    try:
        with (directory / f"{step:04d}-flight-worker.txt").open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            except OSError as exc:
                output.write_text(json.dumps({"error": str(exc), "frames": []}), encoding="utf-8")
            # Finish/stop recording before recognition or another jump starts.
            wait_guarded(max(seconds, 1.6), check)
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_auto(args):
    directory = Path(args.log_dir).resolve() / datetime.now().strftime("auto-%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)
    report = {"mode": "auto", "coefficient": args.coefficient, "angle": args.angle,
              "sender_has_cv2": "cv2" in sys.modules, "jumps": []}

    log_store = RunLogStore(directory, getattr(args, "log_profile", "compact"))
    report["log_profile"] = log_store.profile

    def save():
        (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        mi.require_permissions()
        print("自动模式：请点跳一跳标题栏，将鼠标移入游戏空白处并松开。之后无需操作。", flush=True)
        print("按 Esc、移动鼠标或切换窗口可停止。", flush=True)
        for remaining in range(args.delay, 0, -1):
            print(f"{remaining}…", flush=True)
            time.sleep(1)
        target = aiming.game_window(mi.windows(), args.window_id)
        anchor = mi.cursor()
        if not (target.x + 8 <= anchor[0] < target.x + target.width - 8 and
                target.y + 60 <= anchor[1] < target.y + target.height - 8):
            raise AutoStopped("鼠标不在游戏内容区，请重新启动并移入游戏空白处。")
        mi.guard(target, anchor)
        report.update(window=asdict(target), press_point=anchor)
        last_window_check = [0.0]

        def check():
            abort_if_requested(anchor)
            now = time.monotonic()
            if now - last_window_check[0] > 0.1:
                mi.guard(target, anchor)
                last_window_check[0] = now

        previous = None
        step = 1
        while True:
            check()
            payload = recognize(target, anchor, args.angle, directory, step, check)
            check()
            ms = validate_decision(payload, previous, args.coefficient, args.angle)
            record = {"step": step, "requested_ms": ms, "scene_file": f"{step:04d}-scene.json"}
            report["jumps"].append(record)
            save()
            if args.dry_run:
                record["result"] = "只识别，未发送鼠标事件。"
                print(f"识别完成：计划按压 {ms} ms；标记图：{payload['annotated']}", flush=True)
                report["stop_reason"] = "自动识别预览完成。"
                break
            print(f"第 {step} 跳：{payload['scene']['platform']['shape']} 平台，按压 {ms} ms。", flush=True)
            record["result"] = mi.send_hold(anchor, ms, target, args.backend,
                                             abort_check=lambda: abort_if_requested(anchor))
            previous = payload["scene"]
            save()
            settle_with_evidence(directory, step, ms, args.settle, check)
            # Logging maintenance runs only after mouse-up/settling, never between
            # the fresh-frame check and input. Failures cannot disable stop guards.
            try:
                log_store.after_step(step, payload)
            except Exception as exc:
                report.setdefault("log_warnings", []).append(str(exc))
            if args.max_jumps and step >= args.max_jumps:
                report["stop_reason"] = f"已达到 {args.max_jumps} 跳限制。"
                break
            step += 1
    except (RuntimeError, ValueError, OSError, KeyboardInterrupt) as exc:
        report["stop_reason"] = str(exc) or "用户中断。"
    finally:
        save()
        print("已停止：" + report.get("stop_reason", "未完成。"), flush=True)
        print(f"自动模式记录：{directory}", flush=True)
    return 0
