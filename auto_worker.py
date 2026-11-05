"""Capture stable game frames and recognize a target. This process never sends input."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import aiming
import mac_input as mi
import vision

STABILITY_TIMEOUT = 8.0
RECOVERY_TIMEOUT = 8.0
RECOVERY_STABLE_SECONDS = 1.0


def capture(target, point):
    import cv2
    import numpy as np
    import pyautogui
    mi.guard(target, point)
    sw, sh = pyautogui.size()
    if target.x < 0 or target.y < 0 or target.x + target.width > sw or target.y + target.height > sh:
        raise RuntimeError("请把完整游戏窗口放在主屏幕内。")
    for probe in ((target.x + 10, target.y + 60),
                  (target.x + target.width - 10, target.y + target.height - 10),
                  (target.x + target.width / 2, target.y + target.height / 2)):
        mi.guard(target, probe)
    shot = pyautogui.screenshot()
    sx, sy = shot.width / sw, shot.height / sh
    if abs(sx - sy) > 0.02:
        raise RuntimeError("截图比例异常。")
    box = (round(target.x * sx), round(target.y * sy),
           round((target.x + target.width) * sx), round((target.y + target.height) * sy))
    image = cv2.cvtColor(np.array(shot.crop(box)), cv2.COLOR_RGB2BGR)
    mi.guard(target, point)
    return cv2.resize(image, (400, round(image.shape[0] * 400 / image.shape[1])))


def main():
    import cv2
    request_path = Path(sys.argv[1])
    output = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    target = mi.Window(**request["window"])
    point = request["press_point"]
    angle = request["angle"]
    previous = None
    reason = "画面尚未稳定。"
    recognition_error = None
    last_jump = None
    if request.get("previous_scene"):
        last_path = Path(request["previous_scene"])
        last_scene = json.loads(last_path.read_text(encoding="utf-8"))["scene"]
        last_image = cv2.imread(str(last_path.with_suffix(".png")))
        if last_image is None:
            raise RuntimeError("缺少上一跳画面，无法核对落点。")
        last_jump = last_image, last_scene
    mi.require_permissions()
    started = time.monotonic()
    attempt = 0
    recovering = False
    stable_anchor = None
    stable_count = 0
    initial_failure = None
    while True:
        if not recovering and time.monotonic() - started >= STABILITY_TIMEOUT:
            recovering = True
            initial_failure = reason
            previous = None
            stable_anchor = None
            stable_count = 0
        attempt += 1
        image = capture(target, point)
        captured_at = time.monotonic()
        if not cv2.imwrite(str(output.with_suffix(".png")), image):
            raise RuntimeError("无法保存自动识别画面。")
        try:
            scene = vision.detect_scene(image, angle)
            recognition_error = None
            landing = vision.check_landing(*last_jump, image, scene) if last_jump else None
            stable = (previous is not None and
                      vision.scenes_stable(previous[0], previous[1], image, scene))
            if (not stable or stable_anchor is None or
                    not vision.scenes_stable(stable_anchor[0], stable_anchor[1], image, scene)):
                stable_anchor = image, scene, captured_at
                stable_count = 1
            else:
                stable_count += 1
            body_verified = False
            sustained = (stable_count >= 3 and
                         captured_at - stable_anchor[2] >= RECOVERY_STABLE_SECONDS)
            if sustained and landing is not None and not landing["verified"]:
                recovered = vision.check_landing_recovery(*last_jump, image, scene)
                if recovered["verified"]:
                    initial_failure = initial_failure or landing["reason"]
                    landing = recovered
                    body_verified = True
                else:
                    landing = {**landing, "reason": landing["reason"] + "；侧面复核：" + recovered["reason"]}
            if recovering:
                stable = sustained
            landed = landing is None or landing["verified"]
            if landed and stable:
                scene["signature"] = vision.scene_signature(image, scene)
                payload = {"scene": scene, "window": asdict(target), "captured_at": captured_at,
                           "stable_frames": stable_count, "attempts": attempt,
                           "waited_seconds": round(time.monotonic() - started, 3),
                           "frame": str(output.with_suffix(".png"))}
                if recovering or body_verified:
                    payload["recovery"] = {"method": "fresh-stable-observation",
                                           "initial_failure": initial_failure,
                                           "stable_seconds": round(captured_at - stable_anchor[2], 3)}
                if landing is not None:
                    payload["landing_check"] = landing
                annotated = output.with_name(output.stem + "-aim.png")
                if not cv2.imwrite(str(annotated), vision.annotate(image, scene, angle)):
                    raise RuntimeError("无法保存自动识别标记图。")
                payload["annotated"] = str(annotated)
                output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return 0
            if previous is not None:
                cv2.imwrite(str(output.with_name(output.stem + "-previous.png")), previous[0])
            previous = image, scene
            reason = ("棋子或平台还在移动，未取得连续稳定画面。" if landed
                      else "上一跳落点未确认：" + landing["reason"])
        except vision.RecognitionError as exc:
            recognition_error = exc
            previous = None
            stable_anchor = None
            stable_count = 0
            reason = str(exc)
        remaining = STABILITY_TIMEOUT + RECOVERY_TIMEOUT - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(0.16, remaining))
    if recognition_error is not None:
        raise recognition_error
    raise RuntimeError(f"等待并自动重新核对 {STABILITY_TIMEOUT + RECOVERY_TIMEOUT:g} 秒后仍未确认：{reason}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        payload = {"error": str(exc)}
        if isinstance(exc, vision.RecognitionError):
            payload.update(candidates=exc.candidates, origin=exc.origin)
        Path(sys.argv[2]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(1)
