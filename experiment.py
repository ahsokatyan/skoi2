"""ЛР2: обработать три видео и собрать таблицу, кадры и графики."""
import argparse
import csv
import json
import platform
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import run_video

ROOT = Path(__file__).resolve().parent
VIDEOS = ["mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit"]


def make_examples(result_dir, asset_dir, name):
    cap = cv2.VideoCapture(str(result_dir / "tracked.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [0, 100, 200, 280, 420, min(550, total - 1)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, index in zip(axes.flat, frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Не удалось прочитать результат {name}, кадр {index}")
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Кадр {index}")
        ax.axis("off")
    cap.release()
    fig.suptitle(name)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.5)
    fig.savefig(asset_dir / f"{name}.jpg", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "videos")
    parser.add_argument("--reuse", action="store_true",
                        help="Построить отчёт по уже готовым полным результатам")
    args = parser.parse_args()
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    summaries, series = [], []
    for name in VIDEOS:
        output = ROOT / "results" / name
        if args.reuse:
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        else:
            print(f"Обработка {name}...", flush=True)
            summary = run_video(args.videos_dir / f"{name}.avi", output, label="Mona_Lisa")
        with (output / "metrics.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != summary["frames"]:
            raise ValueError(f"CSV и JSON не совпадают: {name}")
        # Считаем распределение методов из CSV, в том числе при --reuse.
        summary["sift_frames"] = sum(r["method"] == "sift" for r in rows)
        summary["flow_frames"] = sum(r["method"] == "flow" for r in rows)
        summaries.append(summary)
        series.append(rows)
        make_examples(output, assets, name)
        print(f"{name}: {summary['found_frames']}/{summary['frames']} кадров с рамкой", flush=True)
    fields = ["video", "frames", "found_frames", "found_percent", "sift_frames",
              "flow_frames", "mean_ms", "median_ms", "processing_fps"]
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    environment = dict(python=platform.python_version(), opencv=cv2.__version__,
                       numpy=np.__version__, platform=platform.platform(),
                       note="Время зависит от процессора и нагрузки; разметки границ нет")
    (ROOT / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = ["Обычное", "Размытие", "Сильное размытие"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(3)
    sift = [100 * s["sift_frames"] / s["frames"] for s in summaries]
    flow = [100 * s["flow_frames"] / s["frames"] for s in summaries]
    axes[0].bar(x, sift, label="SIFT", color="#3576ad")
    axes[0].bar(x, flow, bottom=sift, label="Оптический поток LK", color="#e8a23a")
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Кадры с рамкой, %")
    axes[0].set_title("Доля кадров с принятой оценкой")
    axes[0].legend(loc="lower left")
    values = [s["mean_ms"] for s in summaries]
    axes[1].bar(x, values, color="#3576ad")
    axes[1].set_ylabel("Среднее время locate, мс/кадр")
    axes[1].set_title("Вычислительное время")
    axes[1].set_ylim(0, max(values) * 1.25)
    for i, s in enumerate(summaries):
        axes[0].text(i, s["found_percent"] + 1, f"{s['found_percent']:.1f}%", ha="center")
        axes[1].text(i, s["mean_ms"] + 1, f"{s['mean_ms']:.1f}", ha="center")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(assets / "summary.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for ax, rows, label in zip(axes, series, labels):
        time = [float(r["time_s"]) for r in rows]
        codes = [{"lost": 0, "flow": 1, "sift": 2}[r["method"]] for r in rows]
        ax.step(time, codes, where="post", linewidth=1)
        ax.set_yticks([0, 1, 2], ["Потеря", "LK", "SIFT"])
        ax.set_ylim(-0.2, 2.2)
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Время видео, с")
    fig.tight_layout()
    fig.savefig(assets / "timeline.png", dpi=150)
    plt.close(fig)
    print("Готово: results.csv, environment.json, assets/ и results/")


if __name__ == "__main__":
    main()
