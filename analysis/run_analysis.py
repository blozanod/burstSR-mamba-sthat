#!/usr/bin/env python3
"""
End-to-end post-training analysis orchestrator for MambaFusion.

Given a training config, this:
  1. Resolves the experiment folder from the config's `name:` field
     (experiments/<name>/, same convention as burstISP/utils/options.py).
  2. Finds the most recent training log for that run.
  3. Runs the logfile analyzer -> dashboard PNG + markdown summary
     (all losses / all validation metrics, discovered dynamically).
  4. Runs the checkpoint progress visualizer across every saved checkpoint
     (visualize_progress.py --source auto: in-domain SyntheticBurstVal bursts for SyntheticBurst
     configs, the RealBSR test bursts otherwise).

All outputs land under analysis/outputs/<name>/, so nothing has to be run by
hand after a training job finishes. Intended to be the single command
appended to the end of an HPC job script (see main/mamba_job.sh).

Usage:
    python analysis/run_analysis.py --config main/config.yml [--skip-progress]

Every stage is isolated: a failure in one (e.g. no GPU available for the
progress visualizer) is reported but does not stop the others, and this
script always exits 0 so it never marks the training job itself as failed.
"""

import argparse
import glob
import os
import subprocess
import sys
import traceback

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_latest_log(exp_dir, name):
    pattern = os.path.join(exp_dir, f"train_{name}_*.log")
    logs = glob.glob(pattern)
    if not logs:
        logs = glob.glob(os.path.join(exp_dir, "*.log"))
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def run_log_analysis(exp_dir, name, out_dir, config_path):
    if yaml is None:
        print("[run_analysis][WARN] PyYAML not available; skipping log analysis.")
        return
    log_path = find_latest_log(exp_dir, name)
    if log_path is None:
        print(f"[run_analysis][WARN] No log file found under {exp_dir}; skipping log analysis.")
        return
    print(f"[run_analysis] Using log: {log_path}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analyze_logfile
    analyze_logfile.run(log_path, out_dir, config_path=config_path)


def run_progress_visualization(exp_dir, out_dir, config_path, input_dir, source):
    checkpoints_dir = os.path.join(exp_dir, "models")
    if not os.path.isdir(checkpoints_dir) or not glob.glob(os.path.join(checkpoints_dir, "*.pth")):
        print(f"[run_analysis][WARN] No checkpoints found at {checkpoints_dir}; skipping progress viz.")
        return

    progress_out = os.path.join(out_dir, "progress")
    script = os.path.join(REPO_ROOT, "analysis", "visualize_progress.py")
    cmd = [
        sys.executable, script,
        "--config", config_path,
        "--checkpoints_dir", checkpoints_dir,
        "--source", source,
        "--output_path", progress_out,
    ] + (["--input_dir", input_dir] if input_dir else [])
    print(f"[run_analysis] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def print_manifest(out_dir):
    print(f"\n[run_analysis] Done. Outputs under: {out_dir}")
    for path in sorted(glob.glob(os.path.join(out_dir, "**", "*"), recursive=True)):
        if os.path.isfile(path):
            print(f"  - {os.path.relpath(path, out_dir)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "main", "config.yml"),
                         help="Path to the training YAML config (default: main/config.yml)")
    parser.add_argument("--exp-root", default=os.path.join(REPO_ROOT, "experiments"),
                         help="Root folder containing experiment run directories")
    parser.add_argument("--input-dir", default=None,
                         help="Dataset root for progress visualization (default: per source -- the config's "
                              "val root for synburst, dataset/RealBSR_RAW_testpatch for realbsr)")
    parser.add_argument("--source", default="auto", choices=["auto", "synburst", "zurich", "realbsr"],
                         help="Progress-visualization data (auto: from the config's val dataset type)")
    parser.add_argument("--skip-progress", action="store_true",
                         help="Skip the (slow, GPU-bound) checkpoint progress visualization")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)

    if yaml is None:
        print("[run_analysis][ERROR] PyYAML is not installed; cannot resolve experiment name.")
        sys.exit(0)

    if not os.path.exists(config_path):
        print(f"[run_analysis][ERROR] Config not found: {config_path}")
        sys.exit(0)

    with open(config_path, "r") as f:
        opt = yaml.safe_load(f) or {}
    name = opt.get("name")
    if not name:
        print(f"[run_analysis][ERROR] Config {config_path} has no top-level `name:` field.")
        sys.exit(0)

    exp_dir = os.path.join(args.exp_root, name)
    out_dir = os.path.join(REPO_ROOT, "analysis", "outputs", name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[run_analysis] Experiment: {name}")
    print(f"[run_analysis] Experiment dir: {exp_dir}")
    print(f"[run_analysis] Output dir: {out_dir}")

    try:
        run_log_analysis(exp_dir, name, out_dir, config_path)
    except Exception:
        print("[run_analysis][ERROR] Logfile analysis stage failed:")
        traceback.print_exc()

    if args.skip_progress:
        print("[run_analysis] Skipping progress visualization (--skip-progress).")
    else:
        try:
            run_progress_visualization(exp_dir, out_dir, config_path, args.input_dir, args.source)
        except Exception:
            print("[run_analysis][ERROR] Progress visualization stage failed:")
            traceback.print_exc()

    print_manifest(out_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
