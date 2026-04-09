import shutil
import sys
import time
from colorama import init, Fore, Style

init(autoreset=True)


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = seconds % 60
    return f"{minutes}m {rest:.1f}s"


def log_info(msg: str):
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {msg}")


def log_step(msg: str):
    print(f"{Fore.BLUE}[STEP]{Style.RESET_ALL} {msg}")


def log_ok(msg: str):
    print(f"{Fore.GREEN}[ OK ]{Style.RESET_ALL} {msg}")


def log_warn(msg: str):
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {msg}")


def log_error(msg: str):
    print(f"{Fore.RED}[ERR ]{Style.RESET_ALL} {msg}")


def timed_step(label: str, func, *args, **kwargs):
    log_step(label)
    t0 = time.time()
    result = func(*args, **kwargs)
    dt = time.time() - t0
    log_ok(f"{label} fertig in {fmt_seconds(dt)}")
    return result


def draw_bottom_panel(lines: list[str]):
    if not lines:
        return
    rows, cols = shutil.get_terminal_size((24, 100))
    visible_lines = lines[-max(1, rows):]
    start_row = max(1, rows - len(visible_lines) + 1)

    sys.stdout.write("\033[s")
    for offset, line in enumerate(visible_lines):
        row = start_row + offset
        sys.stdout.write(f"\033[{row};1H\033[2K{line[:cols]}")
    sys.stdout.write("\033[u")
    sys.stdout.flush()
