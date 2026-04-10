import atexit
import time

from colorama import init

from analyzer.terminal_ui import ConsoleManager

init()

_CONSOLE = ConsoleManager()
atexit.register(_CONSOLE.finish)


def configure_console(
    footer_enabled: bool = True,
    refresh_interval: float = 0.1,
    footer_height: int = 3,
):
    _CONSOLE.configure(
        footer_enabled=footer_enabled,
        refresh_interval=refresh_interval,
        footer_height=footer_height,
    )


def get_console_manager() -> ConsoleManager:
    return _CONSOLE


def finish_console():
    _CONSOLE.finish()


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = seconds % 60
    return f"{minutes}m {rest:.1f}s"


def log_info(msg: str):
    _CONSOLE.log("info", msg)


def log_step(msg: str):
    _CONSOLE.set_step(msg)
    _CONSOLE.log("step", msg)


def log_ok(msg: str):
    _CONSOLE.log("ok", msg)


def log_warn(msg: str):
    _CONSOLE.log("warn", msg)


def log_error(msg: str):
    _CONSOLE.log("error", msg)


def timed_step(label: str, func, *args, **kwargs):
    _CONSOLE.set_step(label)
    log_step(label)
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        _CONSOLE.set_detail(f"{label} fehlgeschlagen: {exc}", level="error")
        raise
    dt = time.time() - t0
    log_ok(f"{label} fertig in {fmt_seconds(dt)}")
    return result
