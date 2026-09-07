#!/usr/bin/env python3
"""Inspect, re-verify, and watch archive elites.

Examples (run in the roblet_evo_v2 folder):
    python view_elite.py --snapshot runs/walker/snapshot_00030.pkl
        -> ranked table of all elites

    python view_elite.py --snapshot ... --rank 0 --reeval
        -> re-evaluate the best elite with the STRICT protocol
           (5 seeds, no early stop, continuous topple detection)

    python view_elite.py --snapshot ... --rank 0 --view
        -> live MuJoCo viewer: watch it fold, then locomote
"""
import argparse
import pickle
import time

import numpy as np

from roblet_evo import params
from roblet_evo.genome import Genome
from roblet_evo.mjcf_articulated import build_articulated_xml
from roblet_evo.selffold import run_fold, fold_targets
from roblet_evo.bake_rigid import bake_rigid_xml
from roblet_evo.locomotion import load_rigid, run_episode, field_torque
from roblet_evo.field import Field


def ranked(archive):
    return sorted(archive.values(), key=lambda e: -e.fitness)


def show_table(elites):
    print(f"{'rank':>4} {'mm/s':>8} {'n':>3} {'freq':>6} {'duty':>5}  "
          f"{'pitch':>6} {'airbn':>6} {'strt':>5} {'lineage':>7} {'born':>5}")
    for i, e in enumerate(elites):
        g = e.genome_dict
        d = e.descriptors
        strt = f"{d[3]:.2f}" if len(d) > 3 else "  -- "
        print(f"{i:>4} {e.fitness*1e3:>8.2f} {e.n_modules:>3} "
              f"{g['freq_hz']:>6.2f} {g['duty']:>5.2f}  "
              f"{d[0]*45:>5.1f}d {d[1]:>6.3f} {strt:>5} {e.lineage_id:>7} "
              f"{e.birth_iter:>5}")


def reeval(genome, seeds=5):
    xml = build_articulated_xml(genome)
    fold = run_fold(xml)
    if not fold.success:
        print("FOLD FAILED on re-run:", fold.reasons)
        return
    model, data, m_eff = load_rigid(bake_rigid_xml(fold), surface="paper")
    speeds = []
    for s in range(seeds):
        ep = run_episode(model, data, m_eff, genome.freq_hz, genome.duty,
                         seed=s, early_stop_t=None)
        usable = 0.0 if (ep.toppled or ep.unstable) else ep.speed
        speeds.append(usable)
        print(f"  seed {s}: {ep.speed*1e3:7.2f} mm/s raw | "
              f"toppled={ep.toppled} unstable={ep.unstable} | "
              f"pitch {ep.pitch_amp_deg:5.1f} deg | "
              f"airborne {ep.airborne_frac:.3f} | "
              f"straight {ep.straightness:.2f} | "
              f"xtrack {ep.cross_track_rms*1e3:5.1f} mm | usable "
              f"{usable*1e3:7.2f} mm/s")
    print(f"STRICT median usable speed: {np.median(speeds)*1e3:.2f} mm/s "
          f"(archive said {'-'} — compare!)")


def view(genome):
    import mujoco
    import mujoco.viewer

    xml = build_articulated_xml(genome)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    targets = fold_targets(model)
    act_target = np.zeros(model.nu)
    for a in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        act_target[a] = targets[aname[len("act_"):]]

    dt = model.opt.timestep
    root_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY,
        f"m{int(model.numeric('root_id').data[0])}_P")

    # same symmetry-breaking tilt as the scored episodes (seed 0)
    rng = np.random.default_rng(0)
    ang = np.radians(params.TILT0_DEG) * rng.uniform(0.5, 1.0)
    ax = rng.normal(size=3)
    ax[2] = 0.0
    ax /= np.linalg.norm(ax)
    data.qpos[3:7] = np.hstack([np.cos(ang / 2), np.sin(ang / 2) * ax])

    print("phase 1: settle -> simultaneous self-fold -> phase 2: field "
          "locomotion (close window to quit)")
    t_pre, t_ramp = 0.3, params.FOLD_RAMP_TIME
    t_hold = params.FOLD_SETTLE_TIME
    fld = Field()
    period = 1.0 / genome.freq_hz

    # locomotion on the ARTICULATED model with hinges held stiff post-fold —
    # visually continuous and equivalent to the rigid bake (servo at target).
    # For the field torque, sum per-magnet contributions on their host bodies.
    from roblet_evo.selffold import magnet_records
    mags = [(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, m["body"]),
             m["dir_local"]) for m in magnet_records(model)]

    latched = False

    def latch_hinges():
        """Hardware latch: clamp every crease's joint range tightly around
        its achieved angle. Joint limits are enforced by the constraint
        solver (stable at any stiffness), so post-fold the body behaves
        rigidly — matching the baked model the fitness was scored on."""
        eps = np.radians(0.5)
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                q = float(data.qpos[model.jnt_qposadr[j]])
                model.jnt_limited[j] = 1
                model.jnt_range[j] = (q - eps, q + eps)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.xpos[root_bid]
        viewer.cam.distance = 0.15
        viewer.cam.elevation = -20
        while viewer.is_running():
            t0 = time.time()
            t = data.time
            if t < t_pre:
                data.ctrl[:] = 0.0
            elif t < t_pre + t_ramp:
                u = (t - t_pre) / t_ramp
                data.ctrl[:] = act_target * (u * u * (3 - 2 * u))
            else:
                data.ctrl[:] = act_target
            if t > t_pre + t_ramp + t_hold and not latched:
                latch_hinges()
                latched = True
                print("\nFOLD DEMO COMPLETE (body latched). This mode does "
                      "NOT show scored locomotion:\nthe articulated model "
                      "self-collides between folded modules (the baked rigid "
                      "body,\nlike real latched hardware, does not). Close "
                      "this window and run --view-rigid\nto watch the actual "
                      "scored gait.")
            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xpos[root_bid]
            viewer.sync()
            left = dt - (time.time() - t0)
            if left > 0:
                time.sleep(left)


def view_rigid(genome, seed=0):
    """EXACT replay of the scored simulation: headless fold, rigid bake,
    then the calibrated locomotion protocol (same tilt seed, same settle,
    same torque model) rendered live. What you see here is what the
    fitness number measured."""
    import mujoco
    import mujoco.viewer
    from roblet_evo.locomotion import quat_rot

    print("folding headless...")
    fold = run_fold(build_articulated_xml(genome))
    if not fold.success:
        print("FOLD FAILED:", fold.reasons)
        return
    model, data, m_eff = load_rigid(bake_rigid_xml(fold, surface="paper",
                                                   markers=True),
                                    surface="paper")
    bid = model.body("robot").id
    dt = model.opt.timestep

    rng = np.random.default_rng(seed)
    ang = np.radians(params.TILT0_DEG) * rng.uniform(0.5, 1.0)
    ax = rng.normal(size=3)
    ax[2] = 0.0
    ax /= np.linalg.norm(ax)
    data.qpos[3:7] = np.hstack([np.cos(ang / 2), np.sin(ang / 2) * ax])

    fld = Field()
    period = 1.0 / genome.freq_hz
    p0 = None
    print(f"rigid locomotion replay: seed {seed}, {genome.freq_hz:.2f} Hz, "
          f"duty {genome.duty:.2f} (grid = 10 mm; close window to quit)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.xpos[bid]
        viewer.cam.distance = 0.35
        viewer.cam.elevation = -30
        while viewer.is_running():
            t0 = time.time()
            t = data.time
            if t >= params.SETTLE_T:
                if p0 is None:
                    p0 = data.xpos[bid][:2].copy()
                Bz = fld.step(t - params.SETTLE_T, period, genome.duty, dt)
                m_world = quat_rot(data.xquat[bid], m_eff)
                data.xfrc_applied[bid, 3:6] = np.cross(
                    m_world, np.array([0.0, 0.0, Bz]))
            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xpos[bid]
            viewer.sync()
            if p0 is not None and int(t / dt) % 5000 == 0:
                d = np.linalg.norm(data.xpos[bid][:2] - p0)
                print(f"  t={t:5.1f}s  displacement {d*1e3:7.2f} mm")
            left = dt - (time.time() - t0)
            if left > 0:
                time.sleep(left)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--rank", type=int, default=None,
                    help="elite rank to act on (0 = best)")
    ap.add_argument("--reeval", action="store_true",
                    help="strict re-evaluation: 5 seeds, no early stop")
    ap.add_argument("--view", action="store_true",
                    help="live viewer: fold + locomotion (demo-style, "
                         "articulated with servo-held hinges)")
    ap.add_argument("--view-rigid", action="store_true",
                    help="live viewer: EXACT replay of the scored rigid "
                         "simulation (what the fitness measured)")
    ap.add_argument("--seed", type=int, default=0,
                    help="episode seed for --view-rigid")
    args = ap.parse_args()

    with open(args.snapshot, "rb") as f:
        payload = pickle.load(f)
    elites = ranked(payload["archive"])
    print(f"{len(elites)} elites | iteration {payload['iteration']}\n")

    if args.rank is None:
        show_table(elites)
        return
    e = elites[args.rank]
    g = Genome.from_dict(e.genome_dict)
    print(f"rank {args.rank}: {e.fitness*1e3:.2f} mm/s | n={e.n_modules} | "
          f"freq {g.freq_hz:.2f} Hz | duty {g.duty:.2f}")
    if args.reeval:
        reeval(g)
    if args.view_rigid:
        view_rigid(g, seed=args.seed)
    elif args.view:
        view(g)


if __name__ == "__main__":
    main()
