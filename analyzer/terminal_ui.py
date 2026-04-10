from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, TextIO

from colorama import Fore, Style

_SPINNER_FRAMES = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
_ASCII_SPINNER_FRAMES = ("-", "\\", "|", "/")
_MIN_INTERACTIVE_COLUMNS = 72
_MIN_INTERACTIVE_LINES = 8


@dataclass
class FooterSnapshot:
    processed_count: int = 0
    total_videos: int = 0
    percent: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    formula: str = ""


def _strip_ansi(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\x1b":
            i += 1
            if i < len(text) and text[i] == "[":
                i += 1
                while i < len(text) and text[i] not in "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~":
                    i += 1
                i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _truncate_middle(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def _coerce_snapshot(snapshot: FooterSnapshot | dict | None) -> FooterSnapshot:
    if snapshot is None:
        return FooterSnapshot()
    if isinstance(snapshot, FooterSnapshot):
        return snapshot
    if all(hasattr(snapshot, key) for key in ("processed_count", "total_videos", "percent", "elapsed_seconds", "eta_seconds", "formula")):
        return FooterSnapshot(
            processed_count=int(getattr(snapshot, "processed_count")),
            total_videos=int(getattr(snapshot, "total_videos")),
            percent=float(getattr(snapshot, "percent")),
            elapsed_seconds=float(getattr(snapshot, "elapsed_seconds")),
            eta_seconds=float(getattr(snapshot, "eta_seconds")),
            formula=str(getattr(snapshot, "formula")),
        )
    return FooterSnapshot(
        processed_count=int(snapshot.get("processed_count", 0)),
        total_videos=int(snapshot.get("total_videos", 0)),
        percent=float(snapshot.get("percent", 0.0)),
        elapsed_seconds=float(snapshot.get("elapsed_seconds", 0.0)),
        eta_seconds=float(snapshot.get("eta_seconds", 0.0)),
        formula=str(snapshot.get("formula", "")),
    )


def _format_clock(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _level_label(level: str) -> str:
    return {
        "ok": "OK",
        "warn": "WARN",
        "error": "ERR",
        "step": "STEP",
    }.get(level, "INFO")


def render_footer_lines(
    snapshot: FooterSnapshot | dict | None,
    *,
    width: int,
    footer_height: int = 3,
    current_video_index: Optional[int] = None,
    current_step: str = "",
    current_url: str = "",
    last_event: str = "",
    last_level: str = "info",
    spinner_index: int = 0,
    ascii_only: bool = False,
    color: bool = True,
    display_percent: Optional[float] = None,
) -> list[str]:
    snapshot = _coerce_snapshot(snapshot)
    width = max(40, int(width))
    percent = max(0.0, min(100.0, snapshot.percent if display_percent is None else float(display_percent)))
    total = max(snapshot.total_videos, 0)

    fill_char = "#" if ascii_only else "█"
    empty_char = "-" if ascii_only else "░"
    spinner_frames = _ASCII_SPINNER_FRAMES if ascii_only else _SPINNER_FRAMES
    spinner = spinner_frames[spinner_index % len(spinner_frames)]

    label = "FORTSCHRITT"
    suffix = f"{percent:5.1f}%"
    bar_width = max(10, width - len(label) - len(suffix) - 4)
    filled = min(bar_width, int(round(bar_width * percent / 100.0)))
    bar_fill = fill_char * filled
    bar_empty = empty_char * (bar_width - filled)

    if color:
        line1 = (
            f"{Style.BRIGHT}{Fore.CYAN}{label}{Style.RESET_ALL} "
            f"[{Style.BRIGHT}{Fore.GREEN}{bar_fill}{Style.DIM}{Fore.BLUE}{bar_empty}{Style.RESET_ALL}] "
            f"{Style.BRIGHT}{Fore.YELLOW}{suffix}{Style.RESET_ALL}"
        )
    else:
        line1 = f"{label} [{bar_fill}{bar_empty}] {suffix}"

    if current_video_index:
        video_text = f"Video {current_video_index}/{max(total, current_video_index)}"
    elif total > 0:
        video_text = f"Setup {snapshot.processed_count}/{total}"
    else:
        video_text = "Setup"

    stage_text = current_step or "Bereit"
    if width < 60:
        line2_plain = f"{spinner} {video_text} | {_format_clock(snapshot.elapsed_seconds)} | ETA {_format_clock(snapshot.eta_seconds)}"
        line3_plain = _truncate_middle(last_event or current_url or stage_text, width)
    else:
        line2_plain = _truncate_middle(
            f"{spinner} {video_text} | Stage: {stage_text} | Elapsed: {_format_clock(snapshot.elapsed_seconds)} | ETA: {_format_clock(snapshot.eta_seconds)}",
            width,
        )
        url_piece = f"URL: {_truncate_middle(current_url, max(12, width // 2 - 6))}" if current_url else "URL: -"
        detail_piece = f"{_level_label(last_level)}: {last_event}" if last_event else f"ETA model: {snapshot.formula or '-'}"
        line3_plain = _truncate_middle(f"{url_piece} | {detail_piece}", width)

    if color:
        level_color = {
            "ok": Fore.GREEN,
            "warn": Fore.YELLOW,
            "error": Fore.RED,
            "step": Fore.BLUE,
        }.get(last_level, Fore.CYAN)
        line2 = f"{Style.BRIGHT}{Fore.CYAN}{line2_plain}{Style.RESET_ALL}"
        line3 = f"{Style.BRIGHT}{level_color}{line3_plain}{Style.RESET_ALL}"
    else:
        line2 = line2_plain
        line3 = line3_plain

    lines = [line1, line2, line3]
    if footer_height > 3:
        lines.extend([""] * (footer_height - 3))
    return lines[:footer_height]


class ConsoleManager:
    def __init__(
        self,
        *,
        stream: Optional[TextIO] = None,
        footer_enabled: bool = True,
        refresh_interval: float = 0.1,
        footer_height: int = 3,
        interactive_override: Optional[bool] = None,
        terminal_size_provider: Optional[Callable[[], os.terminal_size]] = None,
        use_background_thread: bool = True,
    ):
        self.stream = stream or sys.stdout
        self.footer_enabled = bool(footer_enabled)
        self.refresh_interval = float(refresh_interval)
        self.footer_height = max(1, int(footer_height))
        self.interactive_override = interactive_override
        self.terminal_size_provider = terminal_size_provider or (lambda: shutil.get_terminal_size(fallback=(120, 40)))
        self.use_background_thread = use_background_thread

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._footer_active = False
        self._body_bottom = 0
        self._size = self.terminal_size_provider()
        self._ascii_only = False
        self._spinner_index = 0

        self._snapshot = FooterSnapshot()
        self.run_started_at: Optional[float] = None
        self.current_video_index: Optional[int] = None
        self.current_url = ""
        self.current_step = ""
        self.last_event = ""
        self.last_level = "info"
        self.current_video_started_at: Optional[float] = None
        self.expected_video_processing_seconds: Optional[float] = None

    def configure(
        self,
        *,
        footer_enabled: Optional[bool] = None,
        refresh_interval: Optional[float] = None,
        footer_height: Optional[int] = None,
    ):
        with self._lock:
            if footer_enabled is not None:
                self.footer_enabled = bool(footer_enabled)
            if refresh_interval is not None:
                self.refresh_interval = float(refresh_interval)
            if footer_height is not None:
                self.footer_height = max(1, int(footer_height))

    def start_run(self, total_videos: int = 0):
        with self._lock:
            now = time.time()
            self.run_started_at = now
            self._snapshot = FooterSnapshot(total_videos=max(0, int(total_videos)))
            self.current_video_index = None
            self.current_url = ""
            self.current_step = "Initialisierung"
            self.last_event = "Lauf gestartet"
            self.last_level = "info"
            self.current_video_started_at = None
            self.expected_video_processing_seconds = None
            self._spinner_index = 0
            self._stop_event.clear()
            self._enable_footer_locked()
            if self._footer_active:
                self._draw_footer_locked()
                self._start_thread_locked()

    def finish(self):
        thread = None
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.1, self.refresh_interval * 3))
        with self._lock:
            if self._footer_active:
                self._clear_footer_locked()
                self._write("\x1b[r")
                self._write("\x1b[?25h")
                self._footer_active = False
            self.current_video_index = None
            self.current_url = ""
            self.current_step = ""
            self.current_video_started_at = None
            self.expected_video_processing_seconds = None

    def refresh_now(self):
        with self._lock:
            if self._footer_active:
                self._draw_footer_locked()

    def is_footer_active(self) -> bool:
        with self._lock:
            return self._footer_active

    def set_video(
        self,
        video_idx: Optional[int],
        total_videos: Optional[int] = None,
        url: str = "",
        expected_processing_seconds: Optional[float] = None,
    ):
        with self._lock:
            self.current_video_index = int(video_idx) if video_idx else None
            if total_videos is not None:
                self._snapshot.total_videos = max(0, int(total_videos))
            self.current_url = str(url or "")
            self.current_video_started_at = time.time() if video_idx else None
            self.expected_video_processing_seconds = (
                max(0.0, float(expected_processing_seconds))
                if expected_processing_seconds is not None
                else None
            )
            if self._footer_active:
                self._draw_footer_locked()

    def set_step(self, step: str):
        with self._lock:
            self.current_step = str(step or "")
            if self._footer_active:
                self._draw_footer_locked()

    def set_progress(self, snapshot: FooterSnapshot | dict):
        with self._lock:
            new_snapshot = _coerce_snapshot(snapshot)
            self._snapshot = new_snapshot
            if self.run_started_at is None:
                self.run_started_at = time.time() - max(0.0, new_snapshot.elapsed_seconds)
            if self._footer_active:
                self._draw_footer_locked()

    def set_detail(self, detail: str, level: str = "info"):
        with self._lock:
            self.last_event = str(detail or "")
            self.last_level = level
            if self._footer_active:
                self._draw_footer_locked()

    def log(self, level: str, msg: str):
        lines = str(msg).splitlines() or [""]
        with self._lock:
            self.last_event = lines[-1]
            self.last_level = level
            if not self._footer_active:
                for line in lines:
                    self._write(self._format_log_line(level, line) + "\n")
                return

            for line in lines:
                self._write(f"\x1b[{self._body_bottom};1H")
                self._write("\x1b[2K")
                self._write(self._format_log_line(level, line))
                self._write("\n")
            self._draw_footer_locked()

    def _start_thread_locked(self):
        if not self.use_background_thread:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._refresh_loop, name="gpttest-footer", daemon=True)
        self._thread.start()

    def _refresh_loop(self):
        while not self._stop_event.wait(self.refresh_interval):
            with self._lock:
                if self._footer_active:
                    self._spinner_index += 1
                    self._draw_footer_locked()

    def _write(self, text: str):
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:
            pass

    def _format_log_line(self, level: str, msg: str) -> str:
        label = {
            "ok": "[ OK ]",
            "warn": "[WARN]",
            "error": "[ERR ]",
            "step": "[STEP]",
        }.get(level, "[INFO]")
        color = {
            "ok": Fore.GREEN,
            "warn": Fore.YELLOW,
            "error": Fore.RED,
            "step": Fore.BLUE,
        }.get(level, Fore.CYAN)
        return f"{color}{label}{Style.RESET_ALL} {msg}"

    def _interactive_requested_locked(self) -> bool:
        if not self.footer_enabled:
            return False
        if self.interactive_override is not None:
            return self.interactive_override
        is_tty = getattr(self.stream, "isatty", lambda: False)()
        return bool(is_tty)

    def _refresh_layout_locked(self):
        new_size = self.terminal_size_provider()
        self._ascii_only = "utf" not in (getattr(self.stream, "encoding", "") or "").lower()
        if (
            new_size.columns < _MIN_INTERACTIVE_COLUMNS
            or new_size.lines < self.footer_height + _MIN_INTERACTIVE_LINES
        ):
            if self._footer_active:
                self._clear_footer_locked()
                self._write("\x1b[r")
                self._write("\x1b[?25h")
            self._footer_active = False
            self._size = new_size
            return
        layout_changed = (not self._footer_active) or new_size != self._size
        self._size = new_size
        self._body_bottom = self._size.lines - self.footer_height
        if layout_changed:
            self._write("\x1b[r")
            self._write(f"\x1b[1;{self._body_bottom}r")
            self._write("\x1b[?25l")
        self._footer_active = True

    def _enable_footer_locked(self):
        if not self._interactive_requested_locked():
            self._footer_active = False
            return
        self._refresh_layout_locked()

    def _clear_footer_locked(self):
        for offset in range(self.footer_height):
            row = self._body_bottom + offset + 1
            self._write(f"\x1b[{row};1H\x1b[2K")

    def _compute_display_percent_locked(self, now: float) -> float:
        display_percent = float(self._snapshot.percent)
        if (
            self.current_video_index
            and self._snapshot.total_videos > 0
            and self.current_video_started_at is not None
            and self.expected_video_processing_seconds
            and self.expected_video_processing_seconds > 0
        ):
            start_percent = 100.0 * max(0, self.current_video_index - 1) / self._snapshot.total_videos
            end_percent = 100.0 * min(self.current_video_index, self._snapshot.total_videos) / self._snapshot.total_videos
            elapsed = max(0.0, now - self.current_video_started_at)
            fraction = min(0.97, elapsed / self.expected_video_processing_seconds)
            display_percent = max(display_percent, start_percent + (end_percent - start_percent) * fraction)
        return max(0.0, min(100.0, display_percent))

    def _live_snapshot_locked(self, now: float) -> FooterSnapshot:
        elapsed_seconds = self._snapshot.elapsed_seconds
        if self.run_started_at is not None:
            elapsed_seconds = max(elapsed_seconds, now - self.run_started_at)
        return FooterSnapshot(
            processed_count=self._snapshot.processed_count,
            total_videos=self._snapshot.total_videos,
            percent=self._snapshot.percent,
            elapsed_seconds=elapsed_seconds,
            eta_seconds=self._snapshot.eta_seconds,
            formula=self._snapshot.formula,
        )

    def _draw_footer_locked(self):
        if not self._footer_active:
            return
        self._refresh_layout_locked()
        if not self._footer_active:
            return
        now = time.time()
        snapshot = self._live_snapshot_locked(now)
        display_percent = self._compute_display_percent_locked(now)
        lines = render_footer_lines(
            snapshot,
            width=self._size.columns,
            footer_height=self.footer_height,
            current_video_index=self.current_video_index,
            current_step=self.current_step,
            current_url=self.current_url,
            last_event=self.last_event,
            last_level=self.last_level,
            spinner_index=self._spinner_index,
            ascii_only=self._ascii_only,
            display_percent=display_percent,
        )
        self._write("\x1b[s")
        for offset, line in enumerate(lines):
            row = self._body_bottom + offset + 1
            self._write(f"\x1b[{row};1H\x1b[2K{line}{Style.RESET_ALL}")
        self._write("\x1b[u")
