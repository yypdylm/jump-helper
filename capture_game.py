"""Read-only game-window screenshot helper; never imports OpenCV or posts input."""
import json
from pathlib import Path
import sys

import mac_input as m


def main():
    import pyautogui
    window_id, pid = map(int, sys.argv[1:3])
    path = Path(sys.argv[3])
    m.require_permissions()
    items = m.windows()
    target = next((w for w in items if w.id == window_id and w.pid == pid), None)
    if target is None:
        raise RuntimeError("游戏窗口已关闭或隐藏，无法保存跳跃后的画面。")
    center = (target.x + target.width / 2, target.y + target.height / 2)
    m.guard(target, center)
    width, height = pyautogui.size()
    if target.x < 0 or target.y < 0 or target.x + target.width > width or target.y + target.height > height:
        raise RuntimeError("游戏窗口不完全位于主屏幕，未保存跳跃后截图。")
    image = pyautogui.screenshot()
    sx, sy = image.width / width, image.height / height
    box = (round(target.x * sx), round(target.y * sy),
           round((target.x + target.width) * sx), round((target.y + target.height) * sy))
    m.guard(target, center)
    image.crop(box).save(path)
    print(json.dumps({"window": m.asdict(target), "pixels_per_point": [sx, sy]}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
