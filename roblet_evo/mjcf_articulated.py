"""Genome -> articulated MuJoCo MJCF (the fold-stage model).

Body tree mirrors the physical kinematics derived in geometry.py:

  worldbody
    body m{root}_P  (freejoint)          # proximal half of the root
      geom half (oriented to side)
      body m{root}_D (hinge = crease)    # distal half
        geom half
        body m{child}_P ...              # children nest under the half that
                                         # physically hosts their connection
Every frame is authored in the FLAT configuration with axes aligned to the
world, so at qpos = 0 the model reproduces the 2D blueprint exactly and the
folded shape emerges purely from hinge rotations — the same convention as the
kinematic predictor, which makes the two directly comparable in tests.

Magnets are recorded as custom numerics named  mag_{module}_{pad}_{half}
with data = [pos(3), dir(3)] in the HOST HALF's body frame (sign folded into
dir). Fold targets are numerics fold_target_{module}. The rigid baker reads
both by name.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from . import params
from .blueprint import BlueprintResult, embed
from .genome import Genome
from .geometry import AssemblyLayout, build_layout

MESH_SEGMENTS = 16
JOINT_RANGE_MARGIN = math.radians(10.0)


def _f(x: float) -> str:
    return f"{x:.8g}"


def half_disc_mesh() -> Tuple[np.ndarray, float]:
    """Canonical half-disc prism: material on x >= 0, crease along the y axis,
    thickness in z. Returns (vertices Nx3, polygon area)."""
    r = params.MODULE_RADIUS * params.GEOM_SHRINK
    h = params.MODULE_HEIGHT / 2.0
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, MESH_SEGMENTS + 1)
    ring = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    # shoelace area of the closed polygon (arc + chord along the y axis)
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    bottom = np.column_stack([ring, np.full(len(ring), -h)])
    top = np.column_stack([ring, np.full(len(ring), +h)])
    return np.vstack([bottom, top]), float(area)


def _mesh_vertex_string(verts: np.ndarray) -> str:
    return " ".join(_f(v) for v in verts.reshape(-1))


def _zquat(azimuth: float) -> str:
    """Quaternion (w x y z) for a rotation of `azimuth` about +z."""
    return f"{_f(math.cos(azimuth / 2.0))} 0 0 {_f(math.sin(azimuth / 2.0))}"


def half_geom_density() -> float:
    _, area = half_disc_mesh()
    return params.HALF_MASS / (area * params.MODULE_HEIGHT)


def build_articulated_xml(genome: Genome,
                          bp: BlueprintResult = None,
                          gravity: bool = True,
                          with_floor: bool = True) -> str:
    """Compile a genome into an articulated MJCF string."""
    if bp is None:
        bp = embed(genome)
    layout = build_layout(genome, bp)
    verts, area = half_disc_mesh()
    density = params.HALF_MASS / (area * params.MODULE_HEIGHT)
    kin_children = layout.kinematic_children()
    root = genome.root_id
    lim = params.FOLD_DIHEDRAL_RAD + JOINT_RANGE_MARGIN

    numerics: List[str] = [
        f'    <numeric name="n_modules" data="{genome.size()}"/>',
        f'    <numeric name="root_id" data="{root}"/>',
        f'    <numeric name="freq_hz" data="{_f(genome.freq_hz)}"/>',
        f'    <numeric name="duty" data="{_f(genome.duty)}"/>',
        f'    <numeric name="magnet_m0" data="{_f(params.MAGNET_M0)}"/>',
    ]
    for m, ml in layout.modules.items():
        numerics.append(
            f'    <numeric name="fold_target_{m}" data="{_f(ml.hinge_target)}"/>')
        for mg in ml.magnets():
            p, d = mg["pos"], mg["dir"]
            numerics.append(
                f'    <numeric name="mag_{m}_{mg["pad_index"]}_{mg["half"]}" '
                f'data="{_f(p[0])} {_f(p[1])} {_f(p[2])} '
                f'{_f(d[0])} {_f(d[1])} {_f(d[2])}"/>')

    actuators: List[str] = []
    body_lines: List[str] = []

    def half_geom(ml, half: str, indent: str) -> str:
        n = ml.half_outward(half)
        az = math.atan2(n[1], n[0])
        return (f'{indent}<geom type="mesh" mesh="half" '
                f'quat="{_zquat(az)}" density="{_f(density)}" '
                f'friction="0.5 {_f(params.FRICTION_TORSIONAL)} '
                f'{_f(params.FRICTION_ROLLING)}" '
                f'solref="{_f(params.CONTACT_SOLREF[0])} '
                f'{_f(params.CONTACT_SOLREF[1])}" '
                f'rgba="0.20 0.55 1.0 0.9"/>')

    def emit_module(m: int, host_center: np.ndarray, depth: int) -> None:
        ml = layout.modules[m]
        ind = "    " * depth
        rel = ml.center - host_center
        name_p = f"m{m}_P"
        body_lines.append(
            f'{ind}<body name="{name_p}" '
            f'pos="{_f(rel[0])} {_f(rel[1])} {_f(rel[2])}">')
        if m == root:
            body_lines.append(f'{ind}  <freejoint name="root"/>')
        body_lines.append(half_geom(ml, 'P', ind + "  "))
        for child in kin_children.get((m, 'P'), []):
            emit_module(child, ml.center, depth + 1)

        ax = ml.crease_axis
        body_lines.append(f'{ind}  <body name="m{m}_D" pos="0 0 0">')
        body_lines.append(
            f'{ind}    <joint name="fold_{m}" type="hinge" pos="0 0 0" '
            f'axis="{_f(ax[0])} {_f(ax[1])} {_f(ax[2])}" '
            f'range="{_f(-lim)} {_f(lim)}" '
            f'damping="{_f(params.FOLD_JOINT_DAMPING)}"/>')
        body_lines.append(half_geom(ml, 'D', ind + "    "))
        for child in kin_children.get((m, 'D'), []):
            emit_module(child, ml.center, depth + 2)
        body_lines.append(f'{ind}  </body>')
        body_lines.append(f'{ind}</body>')
        actuators.append(
            f'    <position name="act_fold_{m}" joint="fold_{m}" '
            f'kp="{_f(params.FOLD_KP)}" '
            f'ctrlrange="{_f(-lim)} {_f(lim)}"/>')

    # Root body pos = its flat centre plus a hair of z clearance so the whole
    # sheet settles onto the floor. rel = center - host_center must equal
    # center + (0, 0, 2e-4), hence the negative host offset.
    emit_module(root, host_center=np.array([0.0, 0.0, -2e-4]), depth=2)

    grav = "0 0 -9.81" if gravity else "0 0 0"
    floor = ""
    if with_floor:
        floor = (
            '    <geom name="floor" type="plane" size="2 2 0.1" '
            f'material="grid" friction="0.5 {_f(params.FRICTION_TORSIONAL)} '
            f'{_f(params.FRICTION_ROLLING)}" '
            f'solref="{_f(params.CONTACT_SOLREF[0])} '
            f'{_f(params.CONTACT_SOLREF[1])}"/>\n')

    xml = f"""<mujoco model="roblet_articulated">
  <option timestep="{_f(params.TIMESTEP)}" gravity="{grav}"
          integrator="implicitfast"/>

  <custom>
{chr(10).join(numerics)}
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
{floor}{chr(10).join(body_lines)}
  </worldbody>

  <actuator>
{chr(10).join(actuators)}
  </actuator>
</mujoco>
"""
    return xml
