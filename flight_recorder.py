"""Best-effort post-release evidence; a separate process, never sends input."""
import json
from pathlib import Path
import sys
import time


def record(request_path, output, released_at):
    import cv2
    from auto_worker import capture
    import mac_input as mi
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    target = mi.Window(**request["window"])
    output = Path(output)
    result = {"released_at": released_at, "frames": [],
              "note": "松手后采样；启动和截图耗时可能错过飞行，不代表完整录像。"}

    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    try:
        for index in range(6):
            # Do not capture stale evidence after a slow process startup.
            if time.monotonic() - released_at > 1.4:
                break
            image = capture(target, request["press_point"])
            elapsed = time.monotonic() - released_at
            path = output.with_name(f"{output.stem}-{index:02d}.png")
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("无法保存跳后采样画面")
            result["frames"].append({"file": path.name, "after_release_ms": round(elapsed*1000, 1)})
            save()
            time.sleep(0.08)
    except (RuntimeError, OSError) as exc:
        result["error"] = str(exc)
    finally:
        result["finished"] = True
        save()


if __name__ == "__main__":
    record(sys.argv[1], sys.argv[2], float(sys.argv[3]))
