#!/usr/bin/env python
"""Compare old Archer2.0 and migrated recipe/archer runs from focused W&B CSV exports.

The checker is intentionally narrow:
- validate that migrated Archer no longer behaves like native DAPO refill
- verify old-Archer-specific batch metrics are present
- check entropy stays stable instead of drifting upward
- compare best@4 against the old run at matched validation steps

Typical usage:
    python scripts/check_archer_old_alignment.py \
        --old-export-dir /tmp/archer_old_export \
        --new-export-dir /tmp/archer_new_export

You can still pass explicit CSV paths with --old-csv / --new-csv.
If an export directory contains multiple *.history.focused.csv files, also pass
--old-run / --new-run with the W&B display name.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import median


REQUIRED_OLD_ARCHER_BATCH_METRICS = [
    "batch/valid",
    "batch/solve_none",
    "batch/solve_all",
]


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_csv(label: str, csv_path: str | None, export_dir: str | None, run_name: str | None) -> Path:
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"{label} csv not found: {path}")
        return path

    if not export_dir:
        raise ValueError(f"missing {label} input: pass --{label}-csv or --{label}-export-dir")

    export_path = Path(export_dir)
    if not export_path.exists():
        raise FileNotFoundError(f"{label} export dir not found: {export_path}")

    if run_name:
        exact = export_path / f"{sanitize_filename(run_name)}.history.focused.csv"
        if exact.exists():
            return exact

    focused_files = sorted(path for path in export_path.glob("*.history.focused.csv") if path.is_file())
    if not focused_files:
        raise FileNotFoundError(f"no *.history.focused.csv files found in {export_path}")

    if run_name:
        matches: list[Path] = []
        for path in focused_files:
            try:
                rows = load_rows(path)
            except Exception:
                continue
            display_name = rows[0].get("run_display_name") if rows else None
            if display_name == run_name:
                matches.append(path)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple {label} CSVs matched run name {run_name!r} in {export_path}")
        raise FileNotFoundError(f"could not find {label} focused CSV for run {run_name!r} in {export_path}")

    if len(focused_files) == 1:
        return focused_files[0]

    raise ValueError(
        f"multiple {label} focused CSVs found in {export_path}; pass --{label}-run to choose one: "
        + ", ".join(path.name for path in focused_files)
    )


def metric_points(rows: list[dict[str, str]], metric: str, step_limit: int | None = None) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in rows:
        step = _to_int(row.get("_step"))
        value = _to_float(row.get(metric))
        if step is None or value is None:
            continue
        if step_limit is not None and step > step_limit:
            continue
        points.append((step, value))
    points.sort(key=lambda item: item[0])
    return points


def describe(points: list[tuple[int, float]]) -> dict[str, float | int | None]:
    if not points:
        return {
            "count": 0,
            "first_step": None,
            "last_step": None,
            "first_value": None,
            "last_value": None,
            "mean": None,
            "median": None,
        }
    values = [value for _, value in points]
    return {
        "count": len(points),
        "first_step": points[0][0],
        "last_step": points[-1][0],
        "first_value": points[0][1],
        "last_value": points[-1][1],
        "mean": sum(values) / len(values),
        "median": median(values),
    }


def simple_slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [float(step) for step, _ in points]
    ys = [value for _, value in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return numer / denom


def matched_step_pairs(
    old_points: list[tuple[int, float]],
    new_points: list[tuple[int, float]],
    max_step_gap: int,
) -> list[tuple[int, float, float]]:
    new_by_step = {step: value for step, value in new_points}
    pairs: list[tuple[int, float, float]] = []
    for old_step, old_value in old_points:
        if old_step in new_by_step:
            pairs.append((old_step, old_value, new_by_step[old_step]))
            continue
        candidates = [(abs(old_step - new_step), new_step, new_value) for new_step, new_value in new_points]
        if not candidates:
            continue
        distance, _, new_value = min(candidates, key=lambda item: item[0])
        if distance <= max_step_gap:
            pairs.append((old_step, old_value, new_value))
    return pairs


def format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def verdict(label: str, passed: bool, detail: str) -> str:
    status = "PASS" if passed else "WARN"
    return f"[{status}] {label}: {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether migrated recipe/archer matches old Archer behavior.")
    parser.add_argument("--old-csv", help="Focused CSV exported from the old Archer2.0 run.")
    parser.add_argument("--new-csv", help="Focused CSV exported from the migrated recipe/archer run.")
    parser.add_argument("--old-export-dir", help="Directory created by export_wandb_metrics.py for the old run.")
    parser.add_argument("--new-export-dir", help="Directory created by export_wandb_metrics.py for the new run.")
    parser.add_argument("--old-run", help="Old run display name. Needed when --old-export-dir contains multiple runs.")
    parser.add_argument("--new-run", help="New run display name. Needed when --new-export-dir contains multiple runs.")
    parser.add_argument("--step-limit", type=int, default=200, help="Only use training metrics up to this step.")
    parser.add_argument(
        "--val-step-gap",
        type=int,
        default=2,
        help="Allow matching validation points whose steps differ by at most this amount.",
    )
    parser.add_argument(
        "--report-out",
        help="Optional path to save the text report.",
    )
    parser.add_argument(
        "--num-gen-max-frac-gt1",
        type=float,
        default=0.05,
        help="Expected maximum fraction of training points with train/num_gen_batches > 1.",
    )
    parser.add_argument(
        "--entropy-max-rise",
        type=float,
        default=0.10,
        help="Expected maximum allowed increase in actor/entropy over the checked window.",
    )
    parser.add_argument(
        "--best4-max-gap",
        type=float,
        default=0.02,
        help="Expected maximum median new-minus-old best@4 gap at matched validation steps.",
    )
    args = parser.parse_args()

    old_path = resolve_csv("old", args.old_csv, args.old_export_dir, args.old_run)
    new_path = resolve_csv("new", args.new_csv, args.new_export_dir, args.new_run)
    old_rows = load_rows(old_path)
    new_rows = load_rows(new_path)

    old_name = old_rows[0].get("run_display_name", old_path.stem) if old_rows else old_path.stem
    new_name = new_rows[0].get("run_display_name", new_path.stem) if new_rows else new_path.stem

    lines: list[str] = []
    lines.append("Old Archer Alignment Report")
    lines.append(f"old_run: {old_name}")
    lines.append(f"new_run: {new_name}")
    lines.append(f"old_csv: {old_path}")
    lines.append(f"new_csv: {new_path}")
    lines.append(f"step_limit: {args.step_limit}")
    lines.append("")

    old_num_gen = metric_points(old_rows, "train/num_gen_batches", args.step_limit)
    new_num_gen = metric_points(new_rows, "train/num_gen_batches", args.step_limit)
    new_frac_gt1 = sum(1 for _, value in new_num_gen if value > 1.0) / len(new_num_gen) if new_num_gen else 1.0
    old_num_desc = describe(old_num_gen)
    new_num_desc = describe(new_num_gen)
    lines.append("Num Gen Batches")
    lines.append(
        verdict(
            "train/num_gen_batches",
            new_frac_gt1 <= args.num_gen_max_frac_gt1,
            (
                f"old mean={format_float(old_num_desc['mean'])}, new mean={format_float(new_num_desc['mean'])}, "
                f"new frac(>1)={format_float(new_frac_gt1)}"
            ),
        )
    )
    lines.append("")

    lines.append("Batch Metrics")
    for metric in REQUIRED_OLD_ARCHER_BATCH_METRICS:
        old_points = metric_points(old_rows, metric, args.step_limit)
        new_points = metric_points(new_rows, metric, args.step_limit)
        passed = bool(new_points)
        detail = (
            f"old count={len(old_points)}, new count={len(new_points)}, "
            f"new median={format_float(describe(new_points)['median'])}"
        )
        lines.append(verdict(metric, passed, detail))
    lines.append("")

    old_entropy = metric_points(old_rows, "actor/entropy", args.step_limit)
    new_entropy = metric_points(new_rows, "actor/entropy", args.step_limit)
    old_entropy_desc = describe(old_entropy)
    new_entropy_desc = describe(new_entropy)
    old_entropy_delta = None
    new_entropy_delta = None
    if old_entropy_desc["first_value"] is not None and old_entropy_desc["last_value"] is not None:
        old_entropy_delta = float(old_entropy_desc["last_value"]) - float(old_entropy_desc["first_value"])
    if new_entropy_desc["first_value"] is not None and new_entropy_desc["last_value"] is not None:
        new_entropy_delta = float(new_entropy_desc["last_value"]) - float(new_entropy_desc["first_value"])
    old_entropy_slope = simple_slope(old_entropy)
    new_entropy_slope = simple_slope(new_entropy)
    lines.append("Entropy")
    lines.append(
        verdict(
            "actor/entropy",
            new_entropy_delta is not None and new_entropy_delta <= args.entropy_max_rise,
            (
                f"old delta={format_float(old_entropy_delta)}, new delta={format_float(new_entropy_delta)}, "
                f"old slope={format_float(old_entropy_slope, 6)}, new slope={format_float(new_entropy_slope, 6)}"
            ),
        )
    )
    lines.append("")

    old_best4 = metric_points(old_rows, "val-core/livecodebench/acc/best@4/mean")
    new_best4 = metric_points(new_rows, "val-core/livecodebench/acc/best@4/mean")
    pairs = matched_step_pairs(old_best4, new_best4, args.val_step_gap)
    gaps = [new_value - old_value for _, old_value, new_value in pairs]
    median_gap = median(gaps) if gaps else None
    old_peak = max(old_best4, key=lambda item: item[1]) if old_best4 else None
    new_peak = max(new_best4, key=lambda item: item[1]) if new_best4 else None
    lines.append("Validation")
    lines.append(
        verdict(
            "best@4 matched-step gap",
            median_gap is not None and median_gap >= -args.best4_max_gap,
            (
                f"matched_points={len(pairs)}, median(new-old)={format_float(median_gap)}, "
                f"old_peak={format_float(old_peak[1]) if old_peak else 'NA'}@{old_peak[0] if old_peak else 'NA'}, "
                f"new_peak={format_float(new_peak[1]) if new_peak else 'NA'}@{new_peak[0] if new_peak else 'NA'}"
            ),
        )
    )
    if pairs:
        preview = ", ".join(
            f"step {step}: old={old_value:.4f}, new={new_value:.4f}"
            for step, old_value, new_value in pairs[:8]
        )
        lines.append(f"matched preview: {preview}")
    lines.append("")

    hard_pass = (
        new_frac_gt1 <= args.num_gen_max_frac_gt1
        and all(metric_points(new_rows, metric, args.step_limit) for metric in REQUIRED_OLD_ARCHER_BATCH_METRICS)
        and new_entropy_delta is not None
        and new_entropy_delta <= args.entropy_max_rise
    )
    lines.append("Overall")
    if hard_pass:
        lines.append("[PASS] migrated Archer matches the core old-Archer behavioral checks in this window.")
    else:
        lines.append("[WARN] migrated Archer still differs from old-Archer behavior on at least one core check.")

    report = "\n".join(lines) + "\n"
    print(report, end="")

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.write_text(report, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
