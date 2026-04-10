import io
import os
import time
import unittest
from unittest import mock

from analyzer import logging_utils
from analyzer.progress_tracking import RuntimeEstimator
from analyzer.terminal_ui import ConsoleManager, FooterSnapshot, render_footer_lines


class TerminalUiRenderTests(unittest.TestCase):
    def test_render_footer_lines_with_unicode_bar(self):
        snapshot = FooterSnapshot(
            processed_count=2,
            total_videos=5,
            percent=40.0,
            elapsed_seconds=125.0,
            eta_seconds=310.0,
            formula="processing_minutes ≈ 0.25 + 1.20 * video_minutes",
        )

        lines = render_footer_lines(
            snapshot,
            width=100,
            current_video_index=3,
            current_step="WhisperX Alignment",
            current_url="https://www.youtube.com/watch?v=abc123",
            last_event="Match: SPEAKER_00",
            last_level="ok",
            color=False,
        )

        self.assertEqual(len(lines), 3)
        self.assertIn("FORTSCHRITT", lines[0])
        self.assertIn("40.0%", lines[0])
        self.assertIn("WhisperX Alignment", lines[1])
        self.assertIn("Match: SPEAKER_00", lines[2])
        self.assertLessEqual(len(lines[0]), 100)

    def test_render_footer_lines_ascii_fallback(self):
        snapshot = FooterSnapshot(
            processed_count=1,
            total_videos=4,
            percent=25.0,
            elapsed_seconds=12.0,
            eta_seconds=90.0,
            formula="processing_minutes ≈ 0.00 + 1.00 * video_minutes",
        )

        lines = render_footer_lines(
            snapshot,
            width=80,
            current_video_index=2,
            current_step="Audio herunterladen",
            last_event="Noch in Arbeit",
            ascii_only=True,
            color=False,
        )

        self.assertIn("#", lines[0])
        self.assertNotIn("█", lines[0])
        self.assertIn("ETA", lines[1])

    def test_render_footer_lines_for_narrow_terminal_stays_compact(self):
        snapshot = FooterSnapshot(
            processed_count=0,
            total_videos=3,
            percent=0.0,
            elapsed_seconds=5.0,
            eta_seconds=15.0,
            formula="processing_minutes ≈ 0.00 + 1.00 * video_minutes",
        )

        lines = render_footer_lines(
            snapshot,
            width=48,
            current_step="Initialisierung",
            last_event="Warte auf Modelle",
            ascii_only=True,
            color=False,
        )

        self.assertEqual(len(lines), 3)
        self.assertTrue(all(len(line) <= 48 for line in lines))


class TerminalUiConsoleTests(unittest.TestCase):
    def test_console_manager_logs_and_footer_can_share_output(self):
        stream = io.StringIO()
        manager = ConsoleManager(
            stream=stream,
            footer_enabled=True,
            refresh_interval=0.1,
            footer_height=3,
            interactive_override=True,
            terminal_size_provider=lambda: os.terminal_size((100, 20)),
            use_background_thread=False,
        )

        manager.start_run(total_videos=3)
        estimator = RuntimeEstimator(total_videos=3, planned_durations_seconds=[60.0, 60.0, 60.0])
        manager.set_progress(estimator.snapshot(processed_count=0, elapsed_seconds=0.0))
        manager.set_video(1, total_videos=3, url="https://youtu.be/demo", expected_processing_seconds=8.0)
        manager.set_step("Audio herunterladen")
        manager.log("info", "Starte Verarbeitung")
        manager.refresh_now()
        manager.finish()

        output = stream.getvalue()
        self.assertIn("\x1b[1;17r", output)
        self.assertIn("[INFO]", output)
        self.assertIn("Starte Verarbeitung", output)
        self.assertIn("Audio herunterladen", output)
        self.assertIn("\x1b[?25l", output)
        self.assertIn("\x1b[?25h", output)

    def test_timed_step_smoke_updates_console_without_real_models(self):
        stream = io.StringIO()
        manager = ConsoleManager(
            stream=stream,
            footer_enabled=True,
            refresh_interval=0.1,
            footer_height=3,
            interactive_override=True,
            terminal_size_provider=lambda: os.terminal_size((100, 20)),
            use_background_thread=False,
        )
        manager.start_run(total_videos=1)

        with mock.patch.object(logging_utils, "_CONSOLE", manager):
            result = logging_utils.timed_step("Kurzer Schritt", lambda: (time.sleep(0.01), "ok")[1])

        manager.finish()
        output = stream.getvalue()

        self.assertEqual(result, "ok")
        self.assertIn("[STEP]", output)
        self.assertIn("[ OK ]", output)
        self.assertIn("Kurzer Schritt", output)
        self.assertIn("fertig in", output)


if __name__ == "__main__":
    unittest.main()
