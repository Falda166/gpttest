import unittest
from types import SimpleNamespace

from analyzer.app_flow import load_links_from_channel, process_video_batch


class AppFlowTests(unittest.TestCase):
    def test_load_links_from_channel_uses_scraper_settings(self):
        config = SimpleNamespace(
            YOUTUBE_CHANNEL_URL="https://www.youtube.com/@papaplatte/videos",
            YOUTUBE_FETCH_ALL=False,
            YOUTUBE_MAX_LINKS=12,
            YOUTUBE_NO_PROXY=True,
            YOUTUBE_PROXY="http://proxy:8080",
        )

        calls = []

        def fake_extract(channel_url, max_links=None, proxy=None):
            calls.append((channel_url, max_links, proxy))
            return ["https://www.youtube.com/watch?v=abc"]

        links = load_links_from_channel(config, fake_extract)
        self.assertEqual(["https://www.youtube.com/watch?v=abc"], links)
        self.assertEqual(
            [("https://www.youtube.com/@papaplatte/videos", 12, "")],
            calls,
        )

    def test_load_links_from_channel_requires_channel_url(self):
        config = SimpleNamespace(
            YOUTUBE_CHANNEL_URL="",
            YOUTUBE_FETCH_ALL=False,
            YOUTUBE_MAX_LINKS=12,
            YOUTUBE_NO_PROXY=False,
            YOUTUBE_PROXY=None,
        )
        with self.assertRaises(RuntimeError):
            load_links_from_channel(config, lambda *_args, **_kwargs: [])

    def test_process_video_batch_keeps_partial_results_on_ctrl_c(self):
        calls = []
        warnings = []
        errors = []

        def fake_video_processor(idx, url):
            calls.append((idx, url))
            if idx == 2:
                raise KeyboardInterrupt()
            return {
                "words": [f"word{idx}"],
                "transcript": f"text-{idx}",
                "timed_words": [{"start": idx, "word": f"word{idx}", "speaker": "SPEAKER_00"}],
                "speaker_word_counts": {"SPEAKER_00": {f"word{idx}": 1}},
            }

        total_counter, video_texts, timeline_words, global_speaker_counts, interrupted = process_video_batch(
            ["url-1", "url-2", "url-3"],
            fake_video_processor,
            warnings.append,
            errors.append,
        )

        self.assertTrue(interrupted)
        self.assertEqual(calls, [(1, "url-1"), (2, "url-2")])
        self.assertEqual(total_counter["word1"], 1)
        self.assertEqual(video_texts["video_1"], "text-1")
        self.assertEqual(len(timeline_words), 1)
        self.assertEqual(global_speaker_counts["SPEAKER_00"]["word1"], 1)
        self.assertTrue(warnings)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
