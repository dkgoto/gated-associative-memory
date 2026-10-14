"""Run the uploaded overlap-map workflow and save arrays plus run metadata.

The bundled dynamics module uses iid Gaussian W, as confirmed by the author
for the main figure. No original figure data are included.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s_sigmoid", type=float, required=True,
                        help="Gate steepness gamma, including 0 or inf.")
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--g", type=float, default=1.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--t-steps", type=int, default=10000)
    parser.add_argument("--num-ensemble", type=int, default=100)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1380649)
    parser.add_argument("--n-jobs", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--output-dir", type=Path, default=Path("data/overlap_vs_init_overlap"))
    parser.add_argument("--tag", default="Wiid",
                        help="Filename label only. Does not change W in the simulator.")
    parser.add_argument("--quick", action="store_true",
                        help="Small functional test, NOT a paper simulation.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if np.isnan(args.s_sigmoid) or args.s_sigmoid < 0:
        parser.error("s_sigmoid must be nonnegative (inf is allowed).")
    if not np.isfinite(args.g) or args.g <= 0:
        parser.error("g must be finite and positive.")
    if args.num_ensemble > 100:
        parser.error("The preserved seed schedule supports at most 100 runs per cell.")
    if args.seed_base < 0:
        parser.error("seed_base must be nonnegative.")
    if not args.tag or any(c in args.tag for c in '/\\'):
        parser.error("tag must be a nonempty filename label without path separators.")
    return args


def main():
    args = parse_args()
    # These grids and non-test defaults are copied from the uploaded driver.
    alpha_vals = np.linspace(0.01, 0.6, 31)
    init_ov_vals = np.linspace(0.0, 1.0, 21)
    if args.quick:
        args.N, args.t_steps, args.num_ensemble, args.record_every = 40, 100, 2, 10
        alpha_vals = np.array([0.1, 0.4])
        init_ov_vals = np.array([0.0, 0.55, 1.0])
        if args.output_dir == Path("data/overlap_vs_init_overlap"):
            args.output_dir = Path("demo_output")

    key = (f"s{args.s_sigmoid:g}_N{args.N}_g{args.g}_dt{args.dt}"
           f"_T{args.t_steps}_ens{args.num_ensemble}_{args.tag}")
    if args.quick:
        key += "_quick"
    output = args.output_dir / f"overlap_msd_{key}.npz"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}. Use --overwrite deliberately.")

    from init_overlap_vs_capacity import run_initial_overlap_vs_capacity_heatmaps

    _, _, overlap_map, msd_map = run_initial_overlap_vs_capacity_heatmaps(
        N=args.N, g=args.g, tau=args.tau, dt=args.dt,
        s_sigmoid=args.s_sigmoid, gate=True,
        alpha_list=alpha_vals, init_overlap_list=init_ov_vals,
        t_steps=args.t_steps, num_ensemble=args.num_ensemble,
        seed_base=args.seed_base, record_every=args.record_every, n_jobs=args.n_jobs,
    )
    if not np.isfinite(overlap_map).all() or not np.isfinite(msd_map).all():
        raise RuntimeError("Nonfinite observables detected. No output was written.")

    import hopfield_time_dynamics as dynamics
    import scipy
    core_path = Path(dynamics.__file__)
    metadata = {
        "N": args.N, "g": args.g, "tau": args.tau, "dt": args.dt,
        "s_sigmoid": "inf" if np.isinf(args.s_sigmoid) else args.s_sigmoid,
        "gate": True, "Tp": 0.0, "Ta": 0.0,
        "t_steps": args.t_steps, "integration_duration": args.t_steps * args.dt,
        "record_every": args.record_every,
        "last_recorded_time": (args.t_steps // args.record_every) * args.record_every * args.dt,
        "num_ensemble": args.num_ensemble, "seed_base": args.seed_base,
        "seed_rule": "seed_base + 10000*j + 100*i + k",
        "map_axes": ["alpha_index", "initial_overlap_index"],
        "integer_pattern_rule": "(alpha_vals * N).astype(int)",
        "n_jobs": args.n_jobs, "tag": args.tag,
        "quick_test": args.quick,
        "W_ensemble": "iid Gaussian",
        "W_entry_variance": 1.0 / args.N,
        "W_normalization": "1/sqrt(N) included in stored W",
        "W_setting_source": "Author confirmation for the main figure",
        "provenance_status": "Not matched to original published figure data",
        "core_file": core_path.name,
        "core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Retain the four original keys and add unambiguous metadata/effective loads.
    np.savez_compressed(
        output, alpha_vals=alpha_vals, init_ov_vals=init_ov_vals,
        overlap_map=overlap_map, msd_map=msd_map,
        P_vals=(alpha_vals * args.N).astype(int),
        alpha_effective=(alpha_vals * args.N).astype(int) / args.N,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
