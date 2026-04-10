#!/usr/bin/env python3
"""
Export W&B run histories from the internal Weibo W&B deployment.

This script uses the GraphQL endpoint directly instead of wandb.Api because the
server schema is older than the installed client version.

Default behavior:
- export the 5 RL runs discussed in the current analysis
- write one full-history CSV per run
- write one focused-metrics CSV per run
- write one combined focused-metrics CSV across all runs
- write one summary CSV with run metadata + summary metrics

Example:
    WANDB_API_KEY=... python verl/scripts/export_wandb_metrics.py

Override project/entity/output:
    WANDB_API_KEY=... python verl/scripts/export_wandb_metrics.py \
        --entity zy9 \
        --project s_3b_old_baseline_stage1 \
        --outdir /tmp/wandb-export
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://wandb.wml.weibo.com"
DEFAULT_ENTITY = "zy9"
DEFAULT_PROJECT = "s_3b_old_baseline_stage1"
DEFAULT_RUNS = [
    "s_run_3b_rl_code16k_0.125-0.875_offpolicy_no-uly",
    "s_run_3b_rl_code24k_0.125-0.875_offpolicy_no-uly_c3",
    "s_run_3b_rl_code24k_0.125-0.875_offpolicy_no-uly_c6",
    "run_my_dapo_offpolicy_mini_c3_latest",
    "run_my_dapo_offpolicy_mini_c6_latest",
]
FOCUSED_METRICS = [
    "_step",
    "_runtime",
    "_timestamp",
    "val-core/livecodebench/acc/best@4/mean",
    "response_length/clip_ratio",
    "actor/entropy",
    "actor/grad_norm",
    "perf/throughput",
    "perf/time_per_step",
    "perf/mfu/actor",
    "train/num_gen_batches",
    "batch/valid",
    "batch/solve_none",
    "batch/solve_all",
    "batch/all_correct_filtered",
    "batch/all_incorrect_regenerate",
    "batch/mixed_kept",
    "batch/all_incorrect_filtered_max_reached",
]


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


class WandbGraphQLClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        auth = base64.b64encode(f"api:{api_key}".encode()).decode()
        self.headers = {"Authorization": f"Basic {auth}"}
        self.timeout = timeout

    def query(self, query: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/graphql",
            json={"query": query},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"GraphQL error: {payload['errors']}")
        return payload["data"]


def quote_gql_string(value: str) -> str:
    return json.dumps(value)


def build_runs_query(entity: str, project: str, first: int = 200) -> str:
    return f"""
query {{
  project(name: {quote_gql_string(project)}, entityName: {quote_gql_string(entity)}) {{
    runs(first: {first}) {{
      edges {{
        node {{
          id
          name
          displayName
          state
        }}
      }}
    }}
  }}
}}
""".strip()


def build_run_history_query(entity: str, project: str, run_name: str, samples: int) -> str:
    return f"""
query {{
  project(name: {quote_gql_string(project)}, entityName: {quote_gql_string(entity)}) {{
    run(name: {quote_gql_string(run_name)}) {{
      id
      name
      displayName
      state
      historyLineCount
      summaryMetrics
      history(samples: {samples})
    }}
  }}
}}
""".strip()


def parse_history_rows(raw_history: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw_history:
        if not item:
            continue
        if isinstance(item, str):
            rows.append(json.loads(item))
        else:
            rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fields = set()
    for row in rows:
        fields.update(row.keys())

    ordered_fields: list[str] = []
    if field_order:
        ordered_fields.extend([key for key in field_order if key in fields])
    ordered_fields.extend(sorted(fields - set(ordered_fields)))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def filter_metrics(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    filtered_rows = []
    for row in rows:
        filtered = {key: row.get(key) for key in metrics if key in row}
        if filtered:
            filtered_rows.append(filtered)
    return filtered_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export internal W&B run metrics to CSV.")
    parser.add_argument("--base-url", default=os.environ.get("WANDB_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument(
        "--outdir",
        default=str(Path.cwd() / "wandb_exports"),
        help="Directory to write CSV files into.",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        help="Display name of a run to export. Can be passed multiple times.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Missing WANDB_API_KEY. Pass --api-key or export WANDB_API_KEY.", file=sys.stderr)
        return 1

    target_runs = args.runs or DEFAULT_RUNS
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = WandbGraphQLClient(base_url=args.base_url, api_key=args.api_key)

    runs_data = client.query(build_runs_query(entity=args.entity, project=args.project))
    run_edges = runs_data["project"]["runs"]["edges"]
    display_to_name = {edge["node"]["displayName"]: edge["node"]["name"] for edge in run_edges}
    display_to_state = {edge["node"]["displayName"]: edge["node"]["state"] for edge in run_edges}

    missing = [run for run in target_runs if run not in display_to_name]
    if missing:
        print("The following runs were not found in the project:", file=sys.stderr)
        for run in missing:
            print(f"  - {run}", file=sys.stderr)
        print("\nAvailable run display names:", file=sys.stderr)
        for run in sorted(display_to_name):
            print(f"  - {run}", file=sys.stderr)
        return 2

    combined_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for display_name in target_runs:
        run_name = display_to_name[display_name]
        run_state = display_to_state[display_name]

        # First fetch line count + history in one shot using a safely high sample count.
        run_payload = client.query(
            build_run_history_query(
                entity=args.entity,
                project=args.project,
                run_name=run_name,
                samples=20000,
            )
        )["project"]["run"]

        history_rows = parse_history_rows(run_payload["history"])
        summary_metrics = json.loads(run_payload["summaryMetrics"]) if run_payload["summaryMetrics"] else {}

        run_slug = sanitize_filename(display_name)
        full_rows = []
        for row in history_rows:
            full_row = {"run_display_name": display_name, "run_name": run_name, "run_state": run_state}
            full_row.update(row)
            full_rows.append(full_row)

        focused_rows = filter_metrics(full_rows, ["run_display_name", "run_name", "run_state"] + FOCUSED_METRICS)
        combined_rows.extend(focused_rows)

        summary_row = {
            "run_display_name": display_name,
            "run_name": run_name,
            "run_state": run_state,
            "history_line_count": run_payload["historyLineCount"],
        }
        summary_row.update(summary_metrics)
        summary_rows.append(summary_row)

        write_csv(outdir / f"{run_slug}.history.full.csv", full_rows, field_order=["run_display_name", "run_name", "run_state", "_step", "_runtime", "_timestamp"])
        write_csv(outdir / f"{run_slug}.history.focused.csv", focused_rows, field_order=["run_display_name", "run_name", "run_state"] + FOCUSED_METRICS)

        print(
            f"Exported {display_name}: {len(history_rows)} history rows, "
            f"best@4={summary_metrics.get('val-core/livecodebench/acc/best@4/mean')}"
        )

    write_csv(outdir / "all_runs.focused.csv", combined_rows, field_order=["run_display_name", "run_name", "run_state"] + FOCUSED_METRICS)
    write_csv(outdir / "all_runs.summary.csv", summary_rows, field_order=["run_display_name", "run_name", "run_state", "history_line_count"])

    config = {
        "base_url": args.base_url,
        "entity": args.entity,
        "project": args.project,
        "runs": target_runs,
        "focused_metrics": FOCUSED_METRICS,
    }
    (outdir / "export_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Files written to: {outdir}")
    print("Recommended first file to inspect: all_runs.focused.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
