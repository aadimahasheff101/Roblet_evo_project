"""Fold stage: drive every crease simultaneously from flat to its target.

Hardware truth this emulates: uniform heating contracts the PVC film on every
crease at once (no depth staggering), the fold is thermal (field-free), and
hinges latch once folded. The stage runs the articulated model; the caller
then passes the folded state to bake_rigid for the locomotion stage.

Success gate:
  - every crease within FOLD_TOLERANCE of its target (no jam),
  - all state finite, base not flung away,
  - body upright (base z-axis still pointing up).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mujoco
import numpy as np

from . import params

_MAG_RE = re.compile(r"^mag_(\d+)_(\d+)_([PD])$")


@dataclass
class FoldResult:
    success: bool
    reasons: List[str]
    joint_error_rad: Dict[str, float]
    model: mujoco.MjModel
    data: mujoco.MjData

    @property
    def max_error_deg(self) -> float:
        if not self.joint_error_rad:
            return 0.0
        return math.degrees(max(self.joint_error_rad.values()))


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def fold_targets(model: mujoco.MjModel) -> Dict[str, float]:
    """Read per-joint fold targets from the model's custom numerics."""
    out: Dict[str, float] = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and name.startswith("fold_"):
            mid = name[len("fold_"):]
            out[name] = float(model.numeric(f"fold_target_{mid}").data[0])
    return out


def run_fold(xml: str,
             ramp_time: float = None,
             settle_time: float = None,
             pre_settle: float = 0.3) -> FoldResult:
    ramp_time = params.FOLD_RAMP_TIME if ramp_time is None else ramp_time
    settle_time = params.FOLD_SETTLE_TIME if settle_time is None else settle_time

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    targets = fold_targets(model)

    # actuator index -> target (actuators are named act_fold_{m})
    act_target = np.zeros(model.nu)
    for a in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jname = aname[len("act_"):]
        act_target[a] = targets[jname]

    dt = model.opt.timestep
    reasons: List[str] = []

    def blown_up() -> bool:
        return (not np.all(np.isfinite(data.qpos))) or abs(data.qpos[2]) > 1.0

    # field-off settle flat on the floor
    data.ctrl[:] = 0.0
    for _ in range(int(pre_settle / dt)):
        mujoco.mj_step(model, data)
    # simultaneous thermal ramp
    n_ramp = int(ramp_time / dt)
    for i in range(n_ramp):
        data.ctrl[:] = act_target * _smoothstep((i + 1) / n_ramp)
        if i % 50 == 0 and blown_up():
            reasons.append("state diverged during fold ramp")
            break
        mujoco.mj_step(model, data)
    # hold at target
    if not reasons:
        data.ctrl[:] = act_target
        for _ in range(int(settle_time / dt)):
            mujoco.mj_step(model, data)
        if blown_up():
            reasons.append("state diverged during fold settle")

    # gate ---------------------------------------------------------------
    joint_error: Dict[str, float] = {}
    if not reasons:
        for jname, tgt in targets.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            q = float(data.qpos[model.jnt_qposadr[jid]])
            err = abs(q - tgt)
            joint_error[jname] = err
            if err > params.FOLD_TOLERANCE_RAD:
                reasons.append(
                    f"{jname} jammed: {math.degrees(q):.1f} deg vs target "
                    f"{math.degrees(tgt):.1f} deg")
        # uprightness of the base body
        root_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY,
            f"m{int(model.numeric('root_id').data[0])}_P")
        zz = data.xmat[root_bid].reshape(3, 3)[2, 2]
        if zz < params.TOPPLE_Z:
            reasons.append(f"base flipped during fold (z.z = {zz:.2f})")

    return FoldResult(success=(len(reasons) == 0), reasons=reasons,
                      joint_error_rad=joint_error, model=model, data=data)


def magnet_records(model: mujoco.MjModel):
    """Parse the mag_{module}_{pad}_{half} numerics into records with the
    host body name resolved."""
    out = []
    for i in range(model.nnumeric):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, i)
        mm = _MAG_RE.match(name or "")
        if not mm:
            continue
        module, pad, half = int(mm.group(1)), int(mm.group(2)), mm.group(3)
        d = model.numeric(name).data
        out.append(dict(module=module, pad=pad, half=half,
                        body=f"m{module}_{half}",
                        pos_local=np.array(d[0:3]),
                        dir_local=np.array(d[3:6])))
    return out
