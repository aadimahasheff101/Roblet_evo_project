#!/usr/bin/env python3
"""Run (or resume) CVT-MAP-Elites on the walker task.

Examples:
    python run_walker_qd.py --iterations 300 --cells 128 --out runs/walker_s1
    python run_walker_qd.py --resume runs/walker_s1/snapshot_00100.pkl
"""
import argparse
import os
import math

from roblet_evo.qd import MapElites, QDConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--cells", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--init", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = all cores")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--surface", default="paper",
                    choices=["paper", "acrylic"])
    ap.add_argument("--out", default="runs/walker")
    ap.add_argument("--resume", default=None,
                    help="path to a snapshot .pkl to continue from")
    args = ap.parse_args()

    if args.resume:
        me = MapElites.resume(args.resume)
        me.cfg.iterations = max(me.cfg.iterations, args.iterations)
    else:
        me = MapElites(QDConfig(
            n_cells=args.cells, batch_size=args.batch,
            iterations=args.iterations, n_init=args.init,
            workers=args.workers, seed=args.seed,
            surface=args.surface, out_dir=args.out))
    me.run()

    b = me.best()
    print(f"\ncoverage {me.coverage():.2f} | QD-score "
          f"{me.qd_score() * 1e3:.1f} | best {b.fitness * 1e3:.2f} mm/s "
          f"(n={b.n_modules}, freq={b.genome_dict['freq_hz']:.2f} Hz, "
          f"duty={b.genome_dict['duty']:.2f})")


if __name__ == "__main__":
    main()
