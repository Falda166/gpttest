from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from colorama import Fore, Style


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class RuntimeSnapshot:
    processed_count: int
    total_videos: int
    percent: float
    elapsed_seconds: float
    eta_seconds: float
    formula: str


@dataclass
class RuntimeEstimator:
    total_videos: int
    planned_durations_seconds: list[Optional[float]]
    samples: list[dict] = field(default_factory=list)

    def update(self, video_idx: int, video_seconds: float, processing_seconds: float):
        self.samples.append({
            "video_idx": int(video_idx),
            "video_minutes": max(0.0, float(video_seconds) / 60.0),
            "processing_minutes": max(0.0, float(processing_seconds) / 60.0),
            "video_seconds": max(0.0, float(video_seconds)),
            "processing_seconds": max(0.0, float(processing_seconds)),
        })

    def fit(self):
        if not self.samples:
            return 0.0, 1.0

        x = np.array([s["video_minutes"] for s in self.samples], dtype=np.float64)
        y = np.array([s["processing_minutes"] for s in self.samples], dtype=np.float64)

        if len(self.samples) < 2 or np.allclose(x, x[0]):
            ratio = float(np.mean(y / np.maximum(x, 1e-6)))
            return 0.0, max(1e-6, ratio)

        slope, intercept = np.polyfit(x, y, 1)
        return float(intercept), max(1e-6, float(slope))

    def estimate_processing_seconds_for_video(self, video_seconds: Optional[float]) -> Optional[float]:
        if video_seconds is None:
            if not self.samples:
                return None
            return float(np.mean([s["processing_seconds"] for s in self.samples]))
        intercept, slope = self.fit()
        minutes = max(0.0, float(video_seconds) / 60.0)
        return max(0.0, (intercept + slope * minutes) * 60.0)

    def estimate_remaining_seconds(self, processed_count: int) -> float:
        remaining = 0.0
        for duration in self.planned_durations_seconds[processed_count:]:
            estimate = self.estimate_processing_seconds_for_video(duration)
            if estimate is None:
                continue
            remaining += estimate
        return remaining

    def progress_percent(self, processed_count: int) -> float:
        if self.total_videos <= 0:
            return 0.0
        return min(100.0, 100.0 * processed_count / self.total_videos)

    def snapshot(self, processed_count: int, elapsed_seconds: float) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            processed_count=int(processed_count),
            total_videos=int(self.total_videos),
            percent=self.progress_percent(processed_count),
            elapsed_seconds=max(0.0, float(elapsed_seconds)),
            eta_seconds=self.estimate_remaining_seconds(processed_count),
            formula=self.formula_text(),
        )

    def render_progress_line(self, processed_count: int, elapsed_seconds: float) -> str:
        snapshot = self.snapshot(processed_count, elapsed_seconds)
        filled = int(round(snapshot.percent / 4.0))
        bar = "█" * filled + "░" * (25 - filled)
        return (
            f"[{bar}] {snapshot.percent:5.1f}% | Videos: {snapshot.processed_count}/{snapshot.total_videos} | "
            f"Elapsed: {_fmt_hms(snapshot.elapsed_seconds)} | ETA: {_fmt_hms(snapshot.eta_seconds)}"
        )

    def render_progress_panel(self, processed_count: int, elapsed_seconds: float, bar_width: int = 80) -> list[str]:
        pct = self.progress_percent(processed_count)
        width = max(20, int(bar_width))
        filled = int(round((pct / 100.0) * width))
        filled = min(width, max(0, filled))
        empty = width - filled

        bar = (
            f"{Fore.MAGENTA}{'█' * filled}"
            f"{Fore.WHITE}{'░' * empty}"
            f"{Style.RESET_ALL}"
        )
        eta = self.estimate_remaining_seconds(processed_count)

        top = f"{Fore.CYAN}{Style.BRIGHT}╔{'═' * (width + 2)}╗{Style.RESET_ALL}"
        middle = f"{Fore.CYAN}{Style.BRIGHT}║{Style.RESET_ALL} {bar} {Fore.CYAN}{Style.BRIGHT}║{Style.RESET_ALL}"
        bottom = f"{Fore.CYAN}{Style.BRIGHT}╚{'═' * (width + 2)}╝{Style.RESET_ALL}"
        status = (
            f"{Fore.YELLOW}{Style.BRIGHT}Fortschritt:{Style.RESET_ALL} "
            f"{Fore.GREEN}{pct:5.1f}%{Style.RESET_ALL} | "
            f"Videos {processed_count}/{self.total_videos} | "
            f"Elapsed {_fmt_hms(elapsed_seconds)} | "
            f"ETA {_fmt_hms(eta)}"
        )
        return [top, middle, bottom, status]

    def formula_text(self) -> str:
        intercept, slope = self.fit()
        return f"processing_minutes ≈ {intercept:.2f} + {slope:.2f} * video_minutes"

    def export(self, csv_path: Path, html_path: Path):
        if not self.samples:
            return

        df = pd.DataFrame(self.samples)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        intercept, slope = self.fit()
        x_line = np.linspace(0, max(1.0, df["video_minutes"].max() * 1.1), 80)
        y_line = intercept + slope * x_line
        line_df = pd.DataFrame({"video_minutes": x_line, "processing_minutes": y_line})

        fig = px.scatter(
            df,
            x="video_minutes",
            y="processing_minutes",
            hover_data=["video_idx"],
            title=f"Runtime-Schätzung: {self.formula_text()}",
            labels={"video_minutes": "Videolänge (min)", "processing_minutes": "Bearbeitungszeit (min)"},
        )
        fig.add_scatter(
            x=line_df["video_minutes"],
            y=line_df["processing_minutes"],
            mode="lines",
            name="Lineare Schätzung",
        )
        fig.write_html(str(html_path))
