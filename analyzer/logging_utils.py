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
