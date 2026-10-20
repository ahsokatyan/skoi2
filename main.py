"""ЛР2: трекинг плоского объекта по SIFT и оптическому потоку LK."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def resize_frame(frame, max_width):
    """Уменьшаем большие кадры, сохраняя пропорции. Маленькие не увеличиваем."""
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, round(h * max_width / w)))
    return frame


class PlanarTracker:
    def __init__(self, template, ratio=0.75, min_matches=12):
        self.ratio = ratio
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50))
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(gray, None)
        if self.descriptors is None or len(self.keypoints) < min_matches:
            raise ValueError("На образце мало ключевых точек. Выберите текстурный объект.")
        h, w = gray.shape
        self.corners = np.float32([[0, 0], [w - 1, 0],
                                   [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        self.previous_gray = None
        self.previous_points = None
        self.original_points = None

    def match_template(self, gray):
        """Независимый поиск образца по SIFT: позволяет восстановиться после потери."""
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        stats = dict(keypoints=len(keypoints), matches=0, inliers=0,
                     inlier_ratio=0.0, found=False, method="lost")
        if descriptors is None or len(descriptors) < 2:
            return None, stats
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2
                and pair[0].distance < self.ratio * pair[1].distance]
        # Один дескриптор кадра не должен подтверждать несколько точек образца.
        unique = {}
        for match in sorted(good, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        good = list(unique.values())
        stats["matches"] = len(good)
        if len(good) < self.min_matches:
            return None, stats
        src = np.float32([self.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if matrix is None or mask is None:
            return None, stats
        inliers = int(mask.sum())
        stats.update(inliers=inliers, inlier_ratio=inliers / len(good))
        if inliers < 12 or stats["inlier_ratio"] < 0.5:
            return None, stats
        return matrix, stats

    def valid_corners(self, matrix, shape):
        """Проверяем геометрию рамки, прежде чем рисовать её или переносить точки."""
        corners = cv2.perspectiveTransform(self.corners, matrix)
        if not np.isfinite(corners).all():
            return None
        h, w = shape
        area = abs(cv2.contourArea(corners))
        # Отбрасываем вырожденные, самопересекающиеся и неправдоподобные рамки.
        if not cv2.isContourConvex(corners) or not 100 <= area <= 1.5 * w * h:
            return None
        xy = corners.reshape(4, 2)
        if (xy[:, 0].min() < -w or xy[:, 0].max() > 2 * w
                or xy[:, 1].min() < -h or xy[:, 1].max() > 2 * h):
            return None
        # Хотя бы часть рамки должна пересекаться с видимой областью кадра.
        screen = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        overlap, _ = cv2.intersectConvexConvex(xy, screen)
        if overlap < 50:
            return None
        return corners

    def follow_points(self, gray):
        """Если SIFT не сработал, переносим точки с предыдущего кадра методом LK."""
        if self.previous_points is None:
            return None, None
        params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        current, forward, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.previous_points, None, **params)
        if current is None:
            return None, None
        backward, reverse, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, current, None, **params)
        if backward is None:
            return None, None
        error = np.linalg.norm(backward - self.previous_points, axis=2).ravel()
        good = (forward.ravel() == 1) & (reverse.ravel() == 1) & (error < 1.5)
        count = int(good.sum())
        if count < 8:
            return None, None
        matrix, mask = cv2.findHomography(
            self.original_points[good], current[good], cv2.RANSAC, 3.0)
        if matrix is None or mask is None:
            return None, None
        inliers = int(mask.sum())
        if inliers < 8 or inliers / count < 0.6:
            return None, None
        return matrix, dict(keypoints=count, matches=count, inliers=inliers,
                            inlier_ratio=inliers / count, found=False, method="flow")

    def locate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matrix, stats = self.match_template(gray)
        corners = self.valid_corners(matrix, gray.shape) if matrix is not None else None
        method = "sift"
        if corners is None:
            matrix, flow_stats = self.follow_points(gray)
            corners = self.valid_corners(matrix, gray.shape) if matrix is not None else None
            if corners is not None:
                stats, method = flow_stats, "flow"
        self.previous_gray = gray
        self.previous_points = None
        self.original_points = None
        if corners is not None:
            stats.update(found=True, method=method)
            # Обновляем точки внутри найденного объекта, чтобы они не иссякали.
            region = np.zeros_like(gray)
            cv2.fillConvexPoly(region, np.rint(corners).astype(np.int32), 255)
            points = cv2.goodFeaturesToTrack(gray, maxCorners=300,
                                            qualityLevel=0.01, minDistance=5, mask=region)
            if points is not None and len(points) >= 8:
                try:
                    inverse = np.linalg.inv(matrix)
                    original = cv2.perspectiveTransform(points, inverse)
                    if np.isfinite(original).all():
                        self.previous_points, self.original_points = points, original
                except np.linalg.LinAlgError:
                    pass
        return corners, stats


def draw_result(frame, corners, stats, label):
    result = frame.copy()
    if corners is not None:
        polygon = np.rint(corners).astype(np.int32)
        cv2.polylines(result, [polygon], True, (0, 220, 0), 2, cv2.LINE_AA)
        x = int(np.clip(polygon[0, 0, 0], 5, max(5, frame.shape[1] - 150)))
        y = int(np.clip(polygon[0, 0, 1] - 8, 25, frame.shape[0] - 5))
        cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
    status = "FOUND" if stats["found"] else "NOT FOUND"
    text = f"{status} {stats['method']} | matches {stats['matches']} | inliers {stats['inliers']}"
    cv2.rectangle(result, (0, 0), (frame.shape[1], 23), (25, 25, 25), -1)
    cv2.putText(result, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def run_video(video, output, roi=None, select=False, show=False, label="Object",
              max_width=960, ratio=0.75, min_matches=12, snapshot_frames=()):
    """Полная обработка видео; ROI задаётся в координатах уменьшенного первого кадра."""
    if max_width < 64 or not 0 < ratio < 1 or min_matches < 4:
        raise ValueError("Нужны max_width >= 64, 0 < ratio < 1 и min_matches >= 4")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    video, output = Path(video), Path(output)
    if not video.is_file():
        raise FileNotFoundError(f"Видео не найдено: {video}")
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Не удалось прочитать первый кадр: {video}")
        frame = resize_frame(frame, max_width)
        height, width = frame.shape[:2]
        if select:
            roi = tuple(map(int, cv2.selectROI("Select object, Enter to confirm", frame,
                                              fromCenter=False, showCrosshair=True)))
            cv2.destroyAllWindows()
        if roi is None:
            roi = (0, 0, width, height)
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) < 16 or x + w > width or y + h > height:
            raise ValueError("ROI должен находиться внутри первого кадра и быть не меньше 16x16")
        template = frame[y:y + h, x:x + w].copy()
        tracker = PlanarTracker(template, ratio, min_matches)
        output.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output / "template.png"), template):
            raise OSError("Не удалось записать образец")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        writer = cv2.VideoWriter(str(output / "tracked.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise OSError("Не удалось создать MP4. Проверьте поддержку кодека mp4v в OpenCV.")
        rows = []
        index, start = 0, perf_counter()
        while True:
            tick = perf_counter()
            corners, stats = tracker.locate(frame)
            processing_ms = (perf_counter() - tick) * 1000
            result = draw_result(frame, corners, stats, label)
            writer.write(result)
            row = dict(frame=index, time_s=index / fps, **stats, processing_ms=processing_ms)
            rows.append(row)
            if index in snapshot_frames:
                cv2.imwrite(str(output / f"frame_{index:04d}.jpg"), result)
            if show:
                cv2.imshow("Tracking - Q or Esc to stop", result)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            index += 1
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_frame(frame, max_width)
            if frame.shape[:2] != (height, width):
                raise ValueError("Размер кадров меняется внутри видео")
        elapsed = perf_counter() - start
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        times = [r["processing_ms"] for r in rows]
        found = sum(r["found"] for r in rows)
        summary = dict(video=video.name, frames=len(rows), fps_input=fps,
                       width=width, height=height, roi=list(roi),
                       template_keypoints=len(tracker.keypoints),
                       found_frames=found, found_percent=100 * found / len(rows),
                       sift_frames=sum(r["method"] == "sift" for r in rows),
                       flow_frames=sum(r["method"] == "flow" for r in rows),
                       mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                       processing_fps=1000 / float(np.mean(times)),
                       total_loop_seconds=elapsed, opencv=cv2.__version__,
                       ratio=ratio, min_matches=min_matches)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show or select:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Путь к видео; образец берётся из первого кадра")
    parser.add_argument("--output", default="output", help="Папка результатов")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--select", action="store_true", help="Выбрать объект мышью")
    group.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--show", action="store_true", help="Показывать окно обработки")
    parser.add_argument("--label", default="Object", help="Подпись латиницей")
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=12)
    args = parser.parse_args()
    try:
        summary = run_video(args.video, args.output, args.roi, args.select,
                            args.show, args.label, args.max_width,
                            args.ratio, args.min_matches)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Готово: {Path(args.output) / 'tracked.mp4'}")


if __name__ == "__main__":
    main()
