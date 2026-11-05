"""macOS input and window diagnostics. Importing this module sends no events."""
import ctypes
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass

import Quartz as Q
from AppKit import NSWorkspace


@dataclass(frozen=True)
class Window:
    id: int
    pid: int
    owner: str
    title: str
    x: float
    y: float
    width: float
    height: float
    layer: int = 0

    def contains(self, point):
        x, y = point
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


def windows():
    result = Q.CGWindowListCopyWindowInfo(
        Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements,
        Q.kCGNullWindowID,
    )
    if result is None:
        raise RuntimeError("无法读取桌面窗口；请在已授权的 Terminal 中运行。")
    items = []
    for item in result:
        b = item["kCGWindowBounds"]
        if float(item.get("kCGWindowAlpha", 1)) <= 0:
            continue
        items.append(Window(int(item["kCGWindowNumber"]), int(item["kCGWindowOwnerPID"]),
                            str(item.get("kCGWindowOwnerName", "")), str(item.get("kCGWindowName", "")),
                            b["X"], b["Y"], b["Width"], b["Height"], int(item["kCGWindowLayer"])))
    return items


def window_at(items, point):
    # CGWindowList is front to back. Keep nonzero layers so floating overlays block input.
    return next((w for w in items if w.contains(point)), None)


def frontmost():
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return None if app is None else {"pid": int(app.processIdentifier()), "name": str(app.localizedName())}


def permissions():
    lib = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
    lib.AXIsProcessTrusted.restype = ctypes.c_bool
    return {"accessibility": bool(lib.AXIsProcessTrusted()),
            "screen_capture": bool(Q.CGPreflightScreenCaptureAccess())}


def require_permissions():
    missing = [name for name, allowed in permissions().items() if not allowed]
    if missing:
        raise RuntimeError("缺少权限：" + ", ".join(missing) +
                           "。请在系统设置的隐私与安全性中为运行本程序的 Terminal 启用辅助功能、屏幕录制，并重开 Terminal。")


def cursor():
    p = Q.CGEventGetLocation(Q.CGEventCreate(None))
    return float(p.x), float(p.y)


def geometry(window):
    return f"位置 ({window.x:g}, {window.y:g})，尺寸 {window.width:g}×{window.height:g}"


def relocate_target(target, point, items, front_pid):
    """Rebase a screenshot point after translation, once before pressing.

    Preserve window identity and dimensions. During the hold, validate_target
    remains strict: we never chase a moving window with the mouse held down.
    """
    if not target.contains(point):
        raise RuntimeError("原按压位置不在截图时的游戏窗口内。")
    current = next((w for w in items if w.id == target.id and w.pid == target.pid), None)
    if current is None:
        raise RuntimeError("所选游戏窗口已经关闭、隐藏或切换到其他桌面。")
    if (current.width, current.height) != (target.width, target.height):
        raise RuntimeError(f"游戏窗口尺寸已改变，请重新截图。原来：{geometry(target)}；现在：{geometry(current)}")
    relocated = (point[0] + current.x - target.x, point[1] + current.y - target.y)
    validate_target(current, relocated, items, front_pid)
    return current, relocated


def validate_target(target, point, items, front_pid):
    current = next((w for w in items if w.id == target.id and w.pid == target.pid), None)
    if current is None:
        raise RuntimeError("所选游戏窗口已经关闭、隐藏或切换到其他桌面。")
    if (current.x, current.y, current.width, current.height) != (target.x, target.y, target.width, target.height):
        raise RuntimeError(f"游戏窗口位置或大小已改变，请保持窗口稳定后重试。原来：{geometry(target)}；现在：{geometry(current)}")
    if not current.contains(point):
        raise RuntimeError("按压位置不在所选窗口内。")
    top = window_at(items, point)
    if top is None or top.id != target.id or top.pid != target.pid:
        raise RuntimeError("按压位置被其他窗口遮挡；取消发送。")
    if front_pid != target.pid:
        raise RuntimeError("前台进程不是所选游戏窗口的进程；请先点游戏标题栏。")


def guard(target, point):
    f = frontmost()
    validate_target(target, point, windows(), None if f is None else f["pid"])


def duration_ms(value):
    value = float(value)
    if not math.isfinite(value) or not 100 <= value <= 3000:
        raise ValueError("按压时间须为 100–3000 ms。")
    return round(value)


def timed_hold(ms, down, up, check=lambda: None, clock=time.perf_counter, sleep=time.sleep):
    """One down, one up even on interruption; no repeated down/drag or successful double-up."""
    ms = duration_ms(ms)
    check()
    started = clock()
    try:
        down()
        deadline = started + ms / 1000
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                break
            sleep(min(remaining, 0.02))
            check()
    finally:
        up()
    return (clock() - started) * 1000


def quartz_hold(point, ms, target, abort_check=None):
    source = Q.CGEventSourceCreate(Q.kCGEventSourceStateHIDSystemState)
    if source is None:
        raise RuntimeError("无法创建鼠标事件源。")

    def post(kind, pressure, click_count):
        event = Q.CGEventCreateMouseEvent(source, kind, point, Q.kCGMouseButtonLeft)
        if event is None:
            raise RuntimeError("无法创建鼠标事件。")
        Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventClickState, click_count)
        Q.CGEventSetDoubleValueField(event, Q.kCGMouseEventPressure, pressure)
        Q.CGEventPost(Q.kCGHIDEventTap, event)

    guard(target, point)
    if abort_check is not None:
        abort_check()
    post(Q.kCGEventMouseMoved, 0, 0)
    time.sleep(0.15)
    last_check = [0.0]

    def check():
        if abort_check is not None:
            abort_check()
        if cursor()[0] <= 2 and cursor()[1] <= 2:
            raise RuntimeError("鼠标移到主屏幕左上角，已停止并松开。")
        now = time.perf_counter()
        if now - last_check[0] >= 0.1:
            guard(target, point)
            last_check[0] = now

    elapsed = timed_hold(ms, lambda: post(Q.kCGEventLeftMouseDown, 1, 1),
                         lambda: post(Q.kCGEventLeftMouseUp, 0, 1), check)
    return {"backend": "quartz", "requested_ms": ms, "post_interval_ms": round(elapsed, 2),
            "note": "发送端计时，不代表游戏已经收到或识别。"}


def cliclick_command(binary, point, ms):
    coords = ",".join("=" + str(round(v)) for v in point)
    return [binary, "-m", "verbose", f"m:{coords}", f"dd:{coords}",
            f"w:{duration_ms(ms)}", f"du:{coords}"]


def cliclick_hold(point, ms, target, abort_check=None):
    binary = shutil.which("cliclick")
    if binary is None:
        raise RuntimeError("找不到 cliclick。")
    guard(target, point)
    if abort_check is not None:
        abort_check()
    command = cliclick_command(binary, point, ms)
    start = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        while proc.poll() is None:
            if abort_check is not None:
                abort_check()
            if time.perf_counter() - start > ms / 1000 + 5:
                raise RuntimeError("cliclick 执行超时。")
            guard(target, point)
            if cursor()[0] <= 2 and cursor()[1] <= 2:
                raise RuntimeError("鼠标已移到左上角，停止。")
            time.sleep(0.05)
        out, err = proc.communicate()
        if proc.returncode or "privileges not enabled" in err:
            raise RuntimeError(err or out or "cliclick 执行失败。")
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # Emergency release only; successful execution already supplied its own up.
        subprocess.run([binary, command[-1]], timeout=2, check=False, capture_output=True)
        raise
    return {"backend": "cliclick", "requested_ms": ms,
            "process_elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            "stdout": out, "stderr": err, "command": command,
            "note": "含启动、移动及 cliclick 内置等待；不是游戏接收时长。"}


def send_hold(point, ms, target, backend="quartz", abort_check=None):
    require_permissions()
    if Q.CGEventSourceButtonState(Q.kCGEventSourceStateCombinedSessionState, Q.kCGMouseButtonLeft):
        raise RuntimeError("请先松开真实鼠标左键。")
    return (quartz_hold if backend == "quartz" else cliclick_hold)(
        point, duration_ms(ms), target, abort_check=abort_check)


def record_mouse(seconds):
    """Listen only to left mouse events; never record keys or modify events."""
    events = []
    kinds = {Q.kCGEventLeftMouseDown: "down", Q.kCGEventLeftMouseUp: "up",
             Q.kCGEventLeftMouseDragged: "drag"}
    mask = sum(1 << kind for kind in kinds)
    tap = None

    def callback(proxy, kind, event, context):
        if kind in (Q.kCGEventTapDisabledByTimeout, Q.kCGEventTapDisabledByUserInput):
            events.append({"warning": "event tap disabled; trace may be incomplete"})
            Q.CGEventTapEnable(tap, True)
        elif kind in kinds:
            p = Q.CGEventGetLocation(event)
            events.append({"type": kinds[kind], "timestamp_ns": int(Q.CGEventGetTimestamp(event)),
                           "x": p.x, "y": p.y,
                           "source_pid": int(Q.CGEventGetIntegerValueField(event, Q.kCGEventSourceUnixProcessID)),
                           "target_pid_field": int(Q.CGEventGetIntegerValueField(event, Q.kCGEventTargetUnixProcessID)),
                           "click_state": int(Q.CGEventGetIntegerValueField(event, Q.kCGMouseEventClickState)),
                           "pressure": Q.CGEventGetDoubleValueField(event, Q.kCGMouseEventPressure)})
        return event

    tap = Q.CGEventTapCreate(Q.kCGSessionEventTap, Q.kCGTailAppendEventTap,
                            Q.kCGEventTapOptionListenOnly, mask, callback, None)
    if tap is None:
        raise RuntimeError("无法监听鼠标。请检查运行终端的输入监控/辅助功能权限。")
    loop = Q.CFRunLoopGetCurrent()
    source = Q.CFMachPortCreateRunLoopSource(None, tap, 0)
    Q.CFRunLoopAddSource(loop, source, Q.kCFRunLoopCommonModes)
    try:
        Q.CGEventTapEnable(tap, True)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            Q.CFRunLoopRunInMode(Q.kCFRunLoopDefaultMode, min(0.1, deadline - time.monotonic()), False)
    finally:
        Q.CGEventTapEnable(tap, False)
        Q.CFRunLoopRemoveSource(loop, source, Q.kCFRunLoopCommonModes)
        Q.CFMachPortInvalidate(tap)
    return {"events": events, "note": "系统事件流证据，不是微信 touch 回调；target_pid_field 可能为零。"}
