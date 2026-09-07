"""Bake the folded articulated state into a rigid free-body robot.xml.

Hinges latch permanently in hardware, so after a successful fold the whole
assembly is one rigid body. This module fuses the folded half-disc poses into
a single free-joint body and emits an XML honouring the v7 environment
contract (body "robot", plane "floor", numerics m_eff / total_mass_kg), so
the locomotion stage runs in the hardware-calibrated environment unchanged.

Frame choice: the rigid body frame has WORLD-ALIGNED axes and its origin at
the base body's (x, y) on the floor plane. Identity root quaternion in the
baked model therefore reproduces the folded orientation exactly, and m_eff
components are world components — no hidden rotation.

MESH-RECENTERING COMPENSATION (the bug this file once had): MuJoCo's
compiler silently transforms every mesh asset to its principal frame
(centroid shift + reorientation) and composes that transform into each
geom's pos/quat. The folded world poses read from the articulated model
therefore ALREADY include the mesh transform — so the baked geoms must be
authored with the INVERSE mesh transform composed on, or the second
compilation applies it twice and scatters the halves. Authored A satisfies
A o M = F  =>  A_R = F_R M_R^T,  A_p = F_p - A_R m_p.
The geometry-preservation regression test locks this in.
"""
from __future__ import annotations

from typing import List

import mujoco
import numpy as np

from . import params
from .mjcf_articulated import _f, _mesh_vertex_string, half_disc_mesh
from .selffold import FoldResult, magnet_records


def _quat_str(R: np.ndarray) -> str:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    return " ".join(_f(v) for v in q)


def _marker_geoms() -> str:
    """Non-colliding reference posts every 5 cm on a +-0.5 m grid cross,
    so translation is visible whatever heading the robot picks."""
    lines = []
    for k in range(-10, 11):
        if k == 0:
            continue
        d = 0.05 * k
        major = (k % 4 == 0)
        h = 0.012 if major else 0.007
        c = "1 0.3 0.1 1" if major else "1 0.78 0.1 1"
        for x, y in ((d, 0.0), (0.0, d)):
            lines.append(
                f'    <geom type="cylinder" size="0.0012 {h}" '
                f'pos="{_f(x)} {_f(y)} {h}" contype="0" conaffinity="0" '
                f'rgba="{c}"/>')
    lines.append('    <geom type="cylinder" size="0.002 0.015" '
                 'pos="0 0 0.015" contype="0" conaffinity="0" '
                 'rgba="0.1 0.8 0.2 1"/>')
    return "\n".join(lines)


def bake_rigid_xml(fold: FoldResult, surface: str = "acrylic",
                   markers: bool = False) -> str:
    model, data = fold.model, fold.data
    mujoco.mj_forward(model, data)

    root_id = int(model.numeric("root_id").data[0])
    base_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"m{root_id}_P")
    base_p = data.xpos[base_bid]
    origin = np.array([base_p[0], base_p[1], 0.0])

    verts, area = half_disc_mesh()
    density = params.HALF_MASS / (area * params.MODULE_HEIGHT)
    mu = params.SURFACES[surface]

    # inverse of the compiler's mesh-recentering transform (one mesh asset)
    M_R = np.empty(9)
    mujoco.mju_quat2Mat(M_R, model.mesh_quat[0])
    M_R = M_R.reshape(3, 3)
    m_p = np.array(model.mesh_pos[0])

    geoms: List[str] = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        F_p = data.geom_xpos[g] - origin
        F_R = data.geom_xmat[g].reshape(3, 3)
        A_R = F_R @ M_R.T
        A_p = F_p - A_R @ m_p
        geoms.append(
            f'      <geom type="mesh" mesh="half" '
            f'pos="{_f(A_p[0])} {_f(A_p[1])} {_f(A_p[2])}" '
            f'quat="{_quat_str(A_R)}" density="{_f(density)}" '
            f'friction="0.5 {_f(params.FRICTION_TORSIONAL)} '
            f'{_f(params.FRICTION_ROLLING)}" '
            f'solref="{_f(params.CONTACT_SOLREF[0])} '
            f'{_f(params.CONTACT_SOLREF[1])}" '
            f'rgba="0.20 0.55 1.0 0.9"/>')

    # ---- effective dipole: rotate each magnet into the folded pose ------
    m_eff = np.zeros(3)
    mag_lines: List[str] = []
    for k, mg in enumerate(magnet_records(model)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mg["body"])
        Rb = data.xmat[bid].reshape(3, 3)
        d_world = Rb @ mg["dir_local"]
        p_world = data.xpos[bid] + Rb @ mg["pos_local"] - origin
        m_eff += d_world
        mag_lines.append(
            f'    <numeric name="magnet_{k}_m{mg["module"]}p{mg["pad"]}" '
            f'data="{_f(p_world[0])} {_f(p_world[1])} {_f(p_world[2])} '
            f'{_f(d_world[0])} {_f(d_world[1])} {_f(d_world[2])}"/>')
    m_eff *= params.MAGNET_M0

    total_mass = float(np.sum(model.body_mass))
    freq = float(model.numeric("freq_hz").data[0])
    duty = float(model.numeric("duty").data[0])
    n_modules = int(model.numeric("n_modules").data[0])

    xml = f"""<mujoco model="roblet_baked">
  <option timestep="{_f(params.TIMESTEP)}" gravity="0 0 -9.81"
          integrator="implicitfast"/>

  <custom>
    <numeric name="m_eff" data="{_f(m_eff[0])} {_f(m_eff[1])} {_f(m_eff[2])}"/>
    <numeric name="total_mass_kg" data="{_f(total_mass)}"/>
    <numeric name="n_modules" data="{n_modules}"/>
    <numeric name="magnet_m0" data="{_f(params.MAGNET_M0)}"/>
    <numeric name="n_magnets" data="{len(mag_lines)}"/>
    <numeric name="freq_hz" data="{_f(freq)}"/>
    <numeric name="duty" data="{_f(duty)}"/>
{chr(10).join(mag_lines)}
  </custom>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.90 0.90 0.90" rgb2="0.72 0.76 0.80"/>
    <material name="grid" texture="grid" texrepeat="100 100"
              texuniform="true" reflectance="0.05"/>
    <mesh name="half" vertex="{_mesh_vertex_string(verts)}"/>
  </asset>

  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 1" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid"
          friction="{_f(mu)} {_f(params.FRICTION_TORSIONAL)} {_f(params.FRICTION_ROLLING)}"
          solref="{_f(params.CONTACT_SOLREF[0])} {_f(params.CONTACT_SOLREF[1])}"/>
{_marker_geoms() if markers else ""}
    <body name="robot" pos="0 0 0">
      <freejoint name="root"/>
{chr(10).join(geoms)}
    </body>
  </worldbody>
</mujoco>
"""
    return xml
