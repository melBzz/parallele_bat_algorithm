#!/usr/bin/env python3
"""Parse BENCH lines and compute speedup/efficiency + plots.

Usage:
  python3 tools/bench_analyze.py --input code/bench_results.txt --outdir bench_out

The program expects lines like:
  BENCH version=openmp n_bats=2000 iters=2000 procs=1 threads=4 time_s=3.890662
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

BENCH_RE = re.compile(
    r"^BENCH\s+"
    r"version=(?P<version>\S+)\s+"
    r"n_bats=(?P<n_bats>\d+)\s+"
    r"iters=(?P<iters>\d+)\s+"
    r"procs=(?P<procs>\d+)\s+"
    r"threads=(?P<threads>\d+)\s+"
    r"time_s=(?P<time_s>[0-9.]+)\s*$"
)


@dataclass(frozen=True)
class BenchRow:
    version: str
    n_bats: int
    iters: int
    procs: int
    threads: int
    time_s: float

    @property
    def p(self) -> int:
        # For OpenMP: p = threads; For MPI: p = procs; For sequential: 1
        if self.version == "openmp":
            return self.threads
        if self.version == "mpi":
            return self.procs
        return 1


def parse_lines(lines: Iterable[str]) -> List[BenchRow]:
    rows: List[BenchRow] = []
    for line in lines:
        line = line.strip()
        m = BENCH_RE.match(line)
        if not m:
            continue
        rows.append(
            BenchRow(
                version=m.group("version"),
                n_bats=int(m.group("n_bats")),
                iters=int(m.group("iters")),
                procs=int(m.group("procs")),
                threads=int(m.group("threads")),
                time_s=float(m.group("time_s")),
            )
        )
    return rows


def group_key(row: BenchRow) -> Tuple[str, int, int]:
    # Group by version + problem size
    return (row.version, row.n_bats, row.iters)


def find_baseline(rows: List[BenchRow], n_bats: int, iters: int) -> float:
    # Baseline is sequential with same problem size
    candidates = [r for r in rows if r.version == "sequential" and r.n_bats == n_bats and r.iters == iters]
    if not candidates:
        raise SystemExit(f"Missing sequential baseline for n_bats={n_bats} iters={iters}")
    # Choose min time if multiple repeats
    return min(r.time_s for r in candidates)


def _strong_metrics(rows: List[BenchRow]) -> List[Dict[str, object]]:
    """Strong scaling: fixed (n_bats, iters), baseline is sequential with same size."""
    out: List[Dict[str, object]] = []
    sizes = sorted({(r.n_bats, r.iters) for r in rows})
    # Only keep sizes that actually have a sequential baseline
    baselines: Dict[Tuple[int, int], float] = {}
    for (n, it) in sizes:
        try:
            baselines[(n, it)] = find_baseline(rows, n, it)
        except SystemExit:
            continue

    for r in rows:
        key = (r.n_bats, r.iters)
        if key not in baselines:
            continue
        t1 = baselines[key]
        p = r.p
        speedup = t1 / r.time_s if r.time_s > 0 else 0.0
        eff = speedup / p if p > 0 else 0.0
        out.append(
            {
                "mode": "strong",
                "version": r.version,
                "n_bats": r.n_bats,
                "iters": r.iters,
                "procs": r.procs,
                "threads": r.threads,
                "p": p,
                "time_s": r.time_s,
                "baseline_n_bats": r.n_bats,
                "T_base_s": t1,
                "speedup": speedup,
                "efficiency": eff,
            }
        )
    return out


def _weak_baseline(rows: List[BenchRow], iters: int, version: str) -> Optional[BenchRow]:
    """Weak scaling baseline: prefer sequential p=1; fallback to same-version p=1."""
    # Prefer sequential baseline (p=1)
    seq = [r for r in rows if r.iters == iters and r.version == "sequential" and r.p == 1]
    if seq:
        # choose smallest problem size as baseline (usually base per worker)
        return min(seq, key=lambda r: r.n_bats)

    same = [r for r in rows if r.iters == iters and r.version == version and r.p == 1]
    if same:
        return min(same, key=lambda r: r.n_bats)

    return None


def _weak_metrics(rows: List[BenchRow]) -> List[Dict[str, object]]:
    """Weak scaling: n_bats grows with p; baseline is p=1 at smallest n_bats for that iters.

    We compute weak scaling efficiency as:
      E_w(p) = T_base / T_p
    where T_base is the p=1 baseline time at the base problem size.
    """
    out: List[Dict[str, object]] = []

    iters_set = sorted({r.iters for r in rows})
    for iters in iters_set:
        for version in sorted({r.version for r in rows}):
            if version == "sequential":
                continue

            baseline = _weak_baseline(rows, iters, version)
            if baseline is None:
                continue

            t_base = baseline.time_s
            base_n = baseline.n_bats

            candidates = [r for r in rows if r.iters == iters and r.version == version]
            for r in candidates:
                p = r.p
                weak_speedup = t_base / r.time_s if r.time_s > 0 else 0.0
                # weak scaling "efficiency" is typically normalized by ideal constant time, so no /p here
                weak_eff = weak_speedup
                out.append(
                    {
                        "mode": "weak",
                        "version": r.version,
                        "n_bats": r.n_bats,
                        "iters": r.iters,
                        "procs": r.procs,
                        "threads": r.threads,
                        "p": p,
                        "time_s": r.time_s,
                        "baseline_n_bats": base_n,
                        "T_base_s": t_base,
                        "speedup": weak_speedup,
                        "efficiency": weak_eff,
                    }
                )

            # Also include the baseline row itself (useful for plotting)
            out.append(
                {
                    "mode": "weak",
                    "version": baseline.version,
                    "n_bats": baseline.n_bats,
                    "iters": baseline.iters,
                    "procs": baseline.procs,
                    "threads": baseline.threads,
                    "p": 1,
                    "time_s": baseline.time_s,
                    "baseline_n_bats": base_n,
                    "T_base_s": t_base,
                    "speedup": 1.0,
                    "efficiency": 1.0,
                }
            )

    # Deduplicate identical dicts (baseline might be added multiple times)
    uniq: Dict[Tuple, Dict[str, object]] = {}
    for m in out:
        key = (
            m["mode"],
            m["version"],
            m["n_bats"],
            m["iters"],
            m["procs"],
            m["threads"],
        )
        uniq[key] = m
    return list(uniq.values())


def compute_metrics(rows: List[BenchRow]) -> List[Dict[str, object]]:
    # We output both strong and weak metrics in one CSV.
    strong = _strong_metrics(rows)
    weak = _weak_metrics(rows)
    return strong + weak


def try_plot(metrics: List[Dict[str, object]], outdir: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping plots. Install with: pip install matplotlib")
        return

    def plot_group(mode: str, version: str, n_bats: int, iters: int, ms: List[Dict[str, object]]) -> None:
        ms = sorted(ms, key=lambda x: int(x["p"]))
        ps = [int(x["p"]) for x in ms]
        times = [float(x["time_s"]) for x in ms]
        speedups = [float(x["speedup"]) for x in ms]
        effs = [float(x["efficiency"]) for x in ms]
        t_base = float(ms[0].get("T_base_s", times[0]))

        # Time
        plt.figure()
        plt.plot(ps, times, marker="o")
        if mode == "weak":
            plt.axhline(t_base, linestyle="--", linewidth=1.0, label="ideal (constant time)")
            plt.legend()
        plt.xlabel("p (threads or MPI processes)")
        plt.ylabel("Execution time (s)")
        plt.title(f"{version} {mode} scaling: n_bats={n_bats}, iters={iters}")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, f"{version}_{mode}_time_nb{n_bats}_it{iters}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Speedup
        plt.figure()
        plt.plot(ps, speedups, marker="o", label="measured")
        if mode == "strong":
            plt.plot(ps, ps, linestyle="--", label="ideal")
        else:
            plt.plot(ps, [1.0 for _ in ps], linestyle="--", label="ideal (constant time)")
        plt.xlabel("p (threads or MPI processes)")
        plt.ylabel("Speedup")
        plt.title(f"{version} {mode} speedup: n_bats={n_bats}, iters={iters}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(outdir, f"{version}_{mode}_speedup_nb{n_bats}_it{iters}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Efficiency
        plt.figure()
        plt.plot(ps, effs, marker="o")
        plt.xlabel("p (threads or MPI processes)")
        plt.ylabel("Efficiency")
        plt.title(f"{version} {mode} efficiency: n_bats={n_bats}, iters={iters}")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, f"{version}_{mode}_efficiency_nb{n_bats}_it{iters}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Group by mode + version + (n_bats,iters)
    groups: Dict[Tuple[str, str, int, int], List[Dict[str, object]]] = {}
    for m in metrics:
        mode = str(m.get("mode", "strong"))
        version = str(m["version"])
        key = (mode, version, int(m["n_bats"]), int(m["iters"]))
        groups.setdefault(key, []).append(m)

    for (mode, version, n_bats, iters), ms in groups.items():
        # Skip pure sequential plots
        if version == "sequential":
            continue
        # For strong scaling, only plot if size is fixed and we have multiple p points
        if len({int(x["p"]) for x in ms}) < 1:
            continue
        plot_group(mode, version, n_bats, iters, ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input text file containing program output with BENCH lines")
    ap.add_argument("--outdir", default="bench_out", help="Output directory (CSV + PNG plots)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        rows = parse_lines(f)

    if not rows:
        raise SystemExit("No BENCH lines found in input.")

    metrics = compute_metrics(rows)

    # Write CSV
    csv_path = os.path.join(args.outdir, "bench_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "version",
                "n_bats",
                "iters",
                "procs",
                "threads",
                "p",
                "time_s",
                "baseline_n_bats",
                "T_base_s",
                "speedup",
                "efficiency",
            ],
        )
        w.writeheader()
        for m in metrics:
            w.writerow(m)

    print(f"Wrote {csv_path}")

    # Plots
    try_plot(metrics, args.outdir)


if __name__ == "__main__":
    main()
