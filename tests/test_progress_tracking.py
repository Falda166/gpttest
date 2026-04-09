import unittest

from analyzer.progress_tracking import RuntimeEstimator


class RuntimeEstimatorTests(unittest.TestCase):
    def test_snapshot_returns_structured_runtime_values(self):
        estimator = RuntimeEstimator(
            total_videos=4,
            planned_durations_seconds=[60.0, 120.0, None, 180.0],
        )
        estimator.update(video_idx=1, video_seconds=60.0, processing_seconds=90.0)

        snapshot = estimator.snapshot(processed_count=1, elapsed_seconds=95.0)

        self.assertEqual(snapshot.processed_count, 1)
        self.assertEqual(snapshot.total_videos, 4)
        self.assertAlmostEqual(snapshot.percent, 25.0)
        self.assertGreaterEqual(snapshot.eta_seconds, 0.0)
        self.assertEqual(snapshot.elapsed_seconds, 95.0)
        self.assertIn("processing_minutes", snapshot.formula)

    def test_estimate_processing_seconds_for_unknown_duration_uses_mean_sample(self):
        estimator = RuntimeEstimator(
            total_videos=2,
            planned_durations_seconds=[None, None],
        )
        estimator.update(video_idx=1, video_seconds=100.0, processing_seconds=60.0)
        estimator.update(video_idx=2, video_seconds=200.0, processing_seconds=120.0)

        estimate = estimator.estimate_processing_seconds_for_video(None)

        self.assertEqual(estimate, 90.0)


if __name__ == "__main__":
    unittest.main()
