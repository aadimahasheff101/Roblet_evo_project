"""Rigid-body locomotion episode — the hardware-calibrated evaluation stage.

Port of Aaditya's v7 environment (validated against hardware video: absolute
speed, topple boundary, operating envelope), consuming baked robot.xml
strings from bake_rigid. Physics: torque-only uniform-field actuation,
tau = (R m_eff) x B(t) z_hat, applied to the single rigid free body. A
uniform field exerts no net force (F = grad(m . B) = 0), so no force term
exists here by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import mujoco
import numpy as np

from . import params
from .field import Field


def load_rigid(xml: str, surface: str = "acrylic"):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    model.geom_friction[floor] = [params.SURFACES[surface],
                                  params.FRICTION_TORSIONAL,
                                  params.FRICTION_ROLLING]
    m_eff = np.array(model.numeric("m_eff").data[:3])
    return model, data, m_eff


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ v


def field_torque(data, bid: int, m_body: np.ndarray, Bz: float) -> np.ndarray:
    m_world = quat_rot(data.xquat[bid], m_body)
    return np.cross(m_world, np.array([0.0, 0.0, Bz]))


@dataclass
class EpisodeResult:
    speed: float          # m/s
    distance: float       # m (net start-to-end displacement)
    heading_deg: float
    toppled: int
    unstable: int
    # regime fingerprints (QD descriptors)
    pitch_amp_deg: float  # mean per-cycle peak pitch
    airborne_frac: float  # fraction of steps with zero contacts
    # trajectory quality (direction-free: the field defines no XY reference,
    # heading is spontaneous symmetry breaking, so straightness is measured
    # against the robot's OWN path, not a world axis)
    straightness: float = 0.0     # net displacement / path length, in (0, 1]
    cross_track_rms: float = 0.0  # m, RMS deviation from the start-end line

    def as_dict(self) -> Dict[str, float]:
        return dict(speed=self.speed, distance=self.distance,
                    heading_deg=self.heading_deg, toppled=self.toppled,
                    unstable=self.unstable, pitch_amp_deg=self.pitch_amp_deg,
                    airborne_frac=self.airborne_frac,
                    straightness=self.straightness,
                    cross_track_rms=self.cross_track_rms)


def run_episode(model, data, m_eff: np.ndarray,
                freq_hz: float, duty: float, seed: int,
                t_sim: float = None,
                early_stop_t: float = None,
                early_stop_dist: float = 1e-3) -> EpisodeResult:
    """One locomotion episode with the calibrated v7 protocol: random tiny
    initial tilt, field-off settle, then t_sim seconds of actuation.
    Regime fingerprints are recorded alongside the task metrics.

    early_stop_t: if set, abort non-movers — displacement below
    early_stop_dist at that time ends the episode early (evolution
    screening only; leave None for benchmark/calibration runs so the
    protocol matches v7 exactly)."""
    t_sim = params.LOCOMOTION_T if t_sim is None else t_sim
    period = 1.0 / freq_hz
    rng = np.random.default_rng(seed)

    mujoco.mj_resetData(model, data)
    ang = np.radians(params.TILT0_DEG) * rng.uniform(0.5, 1.0)
    ax = rng.normal(size=3)
    ax[2] = 0.0
    ax /= np.linalg.norm(ax)
    data.qpos[3:7] = np.hstack([np.cos(ang / 2), np.sin(ang / 2) * ax])

    bid = model.body("robot").id
    dt = model.opt.timestep
    for _ in range(int(params.SETTLE_T / dt)):
        mujoco.mj_step(model, data)
    p0 = data.xpos[bid].copy()

    fld = Field()
    n = int(t_sim / dt)
    airborne = 0
    pitch_peaks = []
    cyc_peak, cyc_idx = 0.0, 0
    unstable = False
    ever_inverted = False
    traj = [p0[:2].copy()]
    traj_stride = max(1, int(0.01 / dt))   # sample the CoM path at 100 Hz
    i = 0
    for i in range(n):
        Bz = fld.step(i * dt, period, duty, dt)
        data.xfrc_applied[bid, 3:6] = field_torque(data, bid, m_eff, Bz)
        mujoco.mj_step(model, data)
        if data.ncon == 0:
            airborne += 1
        q = data.xquat[bid]
        zz = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        if zz < params.TOPPLE_Z:
            ever_inverted = True   # tumbling/rolling is not a usable gait:
                                   # any inversion marks the episode toppled,
                                   # even if the body lands upright later
        pitch = np.degrees(np.arccos(np.clip(zz, -1, 1)))
        cyc_peak = max(cyc_peak, pitch)
        if int((i * dt) / period) != cyc_idx:
            pitch_peaks.append(cyc_peak)
            cyc_peak = 0.0
            cyc_idx = int((i * dt) / period)
        if (i + 1) % traj_stride == 0:
            traj.append(data.xpos[bid][:2].copy())
        if not np.all(np.isfinite(data.qpos)) or abs(data.qpos[2]) > 1.0:
            unstable = True
            break
        if (early_stop_t is not None and (i + 1) * dt >= early_stop_t
                and np.linalg.norm(data.xpos[bid][:2] - p0[:2]) < early_stop_dist):
            t_sim = (i + 1) * dt   # score over the elapsed window
            break
    data.xfrc_applied[bid] = 0.0
    traj.append(data.xpos[bid][:2].copy())

    disp = data.xpos[bid][:2] - p0[:2]
    dist = float(np.linalg.norm(disp))

    # ---- trajectory quality (direction-free) -----------------------------
    pts = np.asarray(traj)
    seg = np.diff(pts, axis=0)
    path_len = float(np.sum(np.linalg.norm(seg, axis=1)))
    if dist > 1e-3 and path_len > 1e-9:
        straightness = min(1.0, dist / path_len)
        u = disp / dist                                  # start-end unit vector
        rel = pts - pts[0]
        cross = rel[:, 0] * u[1] - rel[:, 1] * u[0]      # signed lateral offset
        cross_track_rms = float(np.sqrt(np.mean(cross ** 2)))
    else:
        straightness = 0.0            # a robot that goes nowhere isn't straight
        cross_track_rms = 0.0

    q = data.xquat[bid]
    body_z_up = (1 - 2 * (q[1] * q[1] + q[2] * q[2])) > params.TOPPLE_Z
    return EpisodeResult(
        speed=dist / t_sim,
        distance=dist,
        heading_deg=float(np.degrees(np.arctan2(disp[1], disp[0]))),
        toppled=int((not body_z_up) or ever_inverted),
        unstable=int(unstable),
        pitch_amp_deg=float(np.mean(pitch_peaks)) if pitch_peaks else 0.0,
        airborne_frac=airborne / max(i + 1, 1),
        straightness=straightness,
        cross_track_rms=cross_track_rms,
    )
