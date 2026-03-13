"""
sweep_4dgs.py

Parameter sweep for train_4dgs_array.py
Each config runs in its own output folder.
Results summary saved as sweep_results.csv at the end.

Usage:
  python sweep_4dgs.py
"""

import os
import subprocess
import itertools
import json
import csv
import time
import numpy as np
from datetime import datetime


# ==========================================
# Base config (fixed across all sweeps)
# ==========================================
BASE = {
    "data_dir": "../imgall/PatterendCutball-close.jpg",
    "da3_depth": "../Depth-Anything-3/img/PatterendCutball-close.jpg/da3_results/npy/depth_all_views.npy",
    "focal_length": 0.6,
    "pixel_size": 0.00185,
    "te": 0.06,
    "img_w": 80,
    "img_h": 60,
    "depth_scale": 1.0,
    "z_max": 500.0,
    "save_every": 500,
    "output_dt_ms": 1.0,
    "ref_camera_idx": -1,
}

# ==========================================
# Sweep configs: each dict = one experiment
# ==========================================
SWEEP = []

# Shared defaults for v5
V5_DEFAULTS = {
    "num_points": -1,
    "iters": 5000,
    "warmup_iters": 0,  # ignored in v5 but accepted
    "integration_steps": 16,
    "views_per_step": 4,
    "lambda_ssim": 0.2,
    "lambda_deform_reg": 0.01,
    "lambda_scale_reg": 0.01,
    "lambda_opacity_reg": 0.005,
    "lambda_sharp_prior": 0.1,
    "lr_pos": 0.0005,
    "lr_color": 0.005,
    "lr_opacity": 0.01,
    "lr_scale": 0.003,
    "lr_deform": 0.0005,
    "lr_rot": 0.001,
    "early_stop_patience": 1000,
}

# --- Group A: lr_deform vs lambda_deform_reg ---
for lr_d in [0.00005, 0.0005, 0.005]:
    for lam_d in [0.001, 0.01, 0.1, 1.0]:
        SWEEP.append({
            **V5_DEFAULTS,
            "name": f"A_lrD{lr_d:.0e}_lamD{lam_d}",
            "lr_deform": lr_d,
            "lambda_deform_reg": lam_d,
        })

# --- Group B: lambda_sharp_prior vs lr_deform ---
for sharp in [0.0, 0.01, 0.1, 0.5, 1.0]:
    for lr_d in [0.0001, 0.0005, 0.002]:
        SWEEP.append({
            **V5_DEFAULTS,
            "name": f"B_sharp{sharp}_lrD{lr_d:.0e}",
            "lambda_sharp_prior": sharp,
            "lr_deform": lr_d,
        })

# --- Group C: integration_steps ---
for steps in [3, 7, 16, 31, 61]:
    SWEEP.append({
        **V5_DEFAULTS,
        "name": f"C_integ{steps}",
        "integration_steps": steps,
    })

# --- Group D: num_points ---
for np_ in [1000, 5000, 10000, -1]:
    SWEEP.append({
        **V5_DEFAULTS,
        "name": f"D_pts{np_}",
        "num_points": np_,
    })

# --- Group E: combos ---
SWEEP.extend([
    {
        **V5_DEFAULTS,
        "name": "E_max_motion",
        "integration_steps": 31,
        "lambda_deform_reg": 0.001,
        "lambda_sharp_prior": 0.5,
        "lr_deform": 0.002,
    },
    {
        **V5_DEFAULTS,
        "name": "E_conservative",
        "integration_steps": 31,
        "lambda_deform_reg": 0.5,
        "lambda_sharp_prior": 0.01,
        "lr_deform": 0.0001,
        "lr_pos": 0.0002,
    },
    {
        **V5_DEFAULTS,
        "name": "E_balanced",
        "integration_steps": 16,
        "lambda_deform_reg": 0.05,
        "lambda_sharp_prior": 0.1,
        "lr_deform": 0.0005,
    },
    {
        **V5_DEFAULTS,
        "name": "E_sharp_focus",
        "integration_steps": 31,
        "lambda_deform_reg": 0.01,
        "lambda_sharp_prior": 1.0,
        "lambda_opacity_reg": 0.02,
        "lr_deform": 0.001,
    },
    {
        **V5_DEFAULTS,
        "name": "E_fine_motion",
        "integration_steps": 61,
        "lambda_deform_reg": 0.01,
        "lambda_sharp_prior": 0.2,
        "lr_deform": 0.0005,
        "iters": 8000,
    },
])


# ==========================================
# Runner
# ==========================================
def build_command(config, out_dir):
    """Merge BASE + config ¡æ command line args."""
    merged = {**BASE, **config}
    merged.pop("name", None)
    merged["out_dir"] = out_dir

    cmd = ["python", "train_4dgs_array_v3.py"]
    for k, v in merged.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def load_final_loss(out_dir):
    """Load loss_history.npy ¡æ return last 100 avg, min, final."""
    loss_path = os.path.join(out_dir, "loss_history.npy")
    if not os.path.exists(loss_path):
        return None, None, None
    loss = np.load(loss_path)
    if len(loss) == 0:
        return None, None, None
    final = float(loss[-1])
    avg100 = float(np.mean(loss[-100:])) if len(loss) >= 100 else float(np.mean(loss))
    best = float(np.min(loss))
    return final, avg100, best


def main():
    sweep_dir = f"./sweep/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(sweep_dir, exist_ok=True)

    total = len(SWEEP)
    print(f"=" * 60)
    print(f"  4DGS Parameter Sweep")
    print(f"  Total experiments: {total}")
    print(f"  Output root: {sweep_dir}")
    print(f"=" * 60)

    # Save sweep config
    with open(os.path.join(sweep_dir, "sweep_configs.json"), "w") as f:
        json.dump(SWEEP, f, indent=2)

    results = []

    for i, config in enumerate(SWEEP):
        name = config["name"]
        out_dir = os.path.join(sweep_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  [{i+1}/{total}] {name}")
        print(f"  Output: {out_dir}")
        print(f"{'=' * 60}")

        # Save this config
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        cmd = build_command(config, out_dir)
        print(f"  CMD: {' '.join(cmd[:6])} ...")

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=open(os.path.join(out_dir, "stdout.log"), "w"),
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1hr max per experiment
            )
            elapsed = time.time() - t0
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            returncode = -1
            print(f"  TIMEOUT after {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - t0
            returncode = -2
            print(f"  ERROR: {e}")

        final, avg100, best = load_final_loss(out_dir)

        result = {
            "name": name,
            "returncode": returncode,
            "elapsed_sec": round(elapsed, 1),
            "loss_final": final,
            "loss_avg100": avg100,
            "loss_best": best,
            **{k: v for k, v in config.items() if k != "name"},
        }
        results.append(result)

        status = "OK" if returncode == 0 else f"FAIL({returncode})"
        loss_str = f"final={final:.4f} avg100={avg100:.4f} best={best:.4f}" if final else "N/A"
        print(f"  {status} | {elapsed:.0f}s | {loss_str}")

    # ==========================================
    # Summary CSV
    # ==========================================
    csv_path = os.path.join(sweep_dir, "sweep_results.csv")
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    # Print ranked results
    valid = [r for r in results if r["loss_best"] is not None]
    valid.sort(key=lambda r: r["loss_best"])

    print(f"\n{'=' * 60}")
    print(f"  SWEEP COMPLETE - {len(valid)}/{total} succeeded")
    print(f"  Results: {csv_path}")
    print(f"{'=' * 60}")
    print(f"\n  Top 10 by best loss:")
    print(f"  {'Rank':<5} {'Name':<35} {'Best':>8} {'Avg100':>8} {'Time':>6}")
    print(f"  {'-'*5} {'-'*35} {'-'*8} {'-'*8} {'-'*6}")
    for i, r in enumerate(valid[:10]):
        print(f"  {i+1:<5} {r['name']:<35} {r['loss_best']:>8.4f} "
              f"{r['loss_avg100']:>8.4f} {r['elapsed_sec']:>5.0f}s")

    print(f"\n  Bottom 5 (worst):")
    for r in valid[-5:]:
        print(f"    {r['name']:<35} best={r['loss_best']:.4f}")

    print(f"\n  All done. Check {sweep_dir}/ for detailed results.")


if __name__ == "__main__":
    main()