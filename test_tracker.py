"""Небольшие воспроизводимые проверки геометрии и потери объекта."""
import unittest

import cv2
import numpy as np

from main import PlanarTracker, resize_frame, run_video


def make_template():
    rng = np.random.default_rng(7)
    image = np.full((240, 320, 3), 220, np.uint8)
    for _ in range(160):
        x, y = rng.integers(10, 230, 2)
        color = tuple(int(v) for v in rng.integers(0, 200, 3))
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 8)), color, -1)
    cv2.putText(image, "LAB 2", (35, 140), 0, 1.4, (0, 0, 0), 3)
    return image


class TrackerTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        cv2.setRNGSeed(42)
        self.image = make_template()

    def test_identity(self):
        tracker = PlanarTracker(self.image)
        corners, stats = tracker.locate(self.image)
        self.assertTrue(stats["found"])
        self.assertEqual(stats["method"], "sift")
        self.assertLess(np.max(np.abs(corners - tracker.corners)), 1)

    def test_perspective(self):
        tracker = PlanarTracker(self.image)
        expected = np.float32([[55, 35], [370, 50], [340, 285], [40, 265]]).reshape(-1, 1, 2)
        matrix = cv2.getPerspectiveTransform(tracker.corners, expected)
        frame = cv2.warpPerspective(self.image, matrix, (420, 320))
        corners, stats = tracker.locate(frame)
        self.assertTrue(stats["found"])
        self.assertLess(float(np.linalg.norm(corners - expected, axis=2).max()), 4)

    def test_flow_fallback(self):
        tracker = PlanarTracker(self.image)
        tracker.locate(self.image)
        # Запрещаем SIFT-сопоставления, чтобы проверить именно резервный LK.
        tracker.min_matches = 100000
        shifted = cv2.warpAffine(self.image, np.float32([[1, 0, 3], [0, 1, 2]]), (320, 240))
        corners, stats = tracker.locate(shifted)
        self.assertTrue(stats["found"])
        self.assertEqual(stats["method"], "flow")
        expected = tracker.corners + np.float32([3, 2])
        self.assertLess(float(np.linalg.norm(corners - expected, axis=2).max()), 3)

    def test_loss_and_recovery(self):
        tracker = PlanarTracker(self.image)
        tracker.locate(self.image)
        corners, stats = tracker.locate(np.zeros_like(self.image))
        self.assertIsNone(corners)
        self.assertFalse(stats["found"])
        corners, stats = tracker.locate(self.image)
        self.assertTrue(stats["found"])

    def test_blank_template(self):
        with self.assertRaises(ValueError):
            PlanarTracker(np.zeros_like(self.image))

    def test_resize_preserves_ratio(self):
        resized = resize_frame(self.image, 160)
        self.assertEqual(resized.shape[:2], (120, 160))
        self.assertIs(resize_frame(self.image, 640), self.image)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            run_video("__no_such_video__.avi", "output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
