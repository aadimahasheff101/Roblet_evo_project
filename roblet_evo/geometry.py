"""Shared 3D geometry: from blueprint to the kinematic layout of half-discs.

Physical model (confirmed with hardware):
  - Each module is a disc that folds along ONE of its three diameters. The
    crease indexed by gene `crease = h` is the diameter PERPENDICULAR to pad h
    (so pad h sits alone on one half; the other two pads share the other half).
  - The two halves rotate relative to each other about the crease by the
    dihedral angle +-45 deg (valley = distal outer edge up). Rigid once folded.
  - Smart-glue connections between modules are rigid welds; ALL articulation
    is module creases.

Frames: everything is authored in "world-flat" coordinates — the flat sheet
laid out by the blueprint, module centres at z = H/2, all local frames axis-
aligned with the world. Folding is then pure joint rotation, both here (the
kinematic predictor) and in MuJoCo (hinge joints), so the two agree by
construction.

Naming: each module m contributes two half-bodies, (m, 'P') proximal (the
half on the path toward the root — it carries the incoming connection) and
(m, 'D') distal (the half that rotates about the crease).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import params
from .blueprint import BlueprintResult
from .genome import Genome, FLAT

Side = str  # 'A' (pad h alone) or 'B' (the other two pads)
Half = str  # 'P' proximal or 'D' distal


def unit(azimuth: float) -> np.ndarray:
    return np.array([math.cos(azimuth), math.sin(azimuth), 0.0])


@dataclass
class ModuleLayout:
    module_id: int
    center: np.ndarray                 # world-flat centre (z = H/2)
    pads: Tuple[int, int, int]         # actual hex sides (pad-list order)
    pad_azimuths: Tuple[float, float, float]
    polarity: Tuple[int, int, int]
    crease_index: int                  # h: crease is perpendicular to pad h
    fold_dir: int                      # +1 valley / -1 mountain / 0 flat
    proximal_side: Side                # side containing pad 0 (incoming)

    # ---- derived directions -------------------------------------------------
    @property
    def n_A(self) -> np.ndarray:
        """Outward unit normal (in-plane) of side A = azimuth of pad h."""
        return unit(self.pad_azimuths[self.crease_index])

    def outward(self, side: Side) -> np.ndarray:
        return self.n_A if side == 'A' else -self.n_A

    @property
    def distal_side(self) -> Side:
        return 'B' if self.proximal_side == 'A' else 'A'

    def side_of_pad(self, pad_index: int) -> Side:
        return 'A' if pad_index == self.crease_index else 'B'

    def half_of_pad(self, pad_index: int) -> Half:
        return 'P' if self.side_of_pad(pad_index) == self.proximal_side else 'D'

    def half_outward(self, half: Half) -> np.ndarray:
        side = self.proximal_side if half == 'P' else self.distal_side
        return self.outward(side)

    @property
    def crease_axis(self) -> np.ndarray:
        """Hinge axis t chosen so +theta lifts the DISTAL outer edge upward
        (valley). t = n_distal x z_hat."""
        n = self.outward(self.distal_side)
        return np.array([n[1], -n[0], 0.0])   # n x z_hat

    @property
    def hinge_target(self) -> float:
        return float(self.fold_dir) * params.FOLD_DIHEDRAL_RAD

    # ---- magnets ------------------------------------------------------------
    def magnets(self) -> List[dict]:
        """One magnet per pad: local position/direction relative to the module
        centre in world-flat axes, with polarity sign applied to direction."""
        out = []
        for i, (pad, az, s) in enumerate(
                zip(self.pads, self.pad_azimuths, self.polarity)):
            u = unit(az)
            out.append(dict(
                pad_index=i, actual_side=pad, sign=int(s),
                pos=params.MAGNET_SITE_RADIUS * u,
                dir=float(s) * u,
                half=self.half_of_pad(i),
            ))
        return out

    def dipole_flat(self) -> np.ndarray:
        """Module net dipole in the flat state = m0 * sum of signed pad units.
        (120-degree identity: all-same-sign => zero; one flipped => 2*m0.)"""
        d = np.zeros(3)
        for mg in self.magnets():
            d += mg["dir"]
        return params.MAGNET_M0 * d


@dataclass
class AssemblyLayout:
    genome: Genome
    blueprint: BlueprintResult
    modules: Dict[int, ModuleLayout]
    parent_half: Dict[int, Tuple[int, Half]]   # child -> (parent, hosting half)

    def kinematic_children(self) -> Dict[Tuple[int, Half], List[int]]:
        """Which child modules hang off each (module, half)."""
        out: Dict[Tuple[int, Half], List[int]] = {}
        for child, (parent, half) in self.parent_half.items():
            out.setdefault((parent, half), []).append(child)
        return out


def build_layout(genome: Genome, bp: BlueprintResult) -> AssemblyLayout:
    if not bp.valid:
        raise ValueError(f"blueprint invalid: {bp.reasons}")

    modules: Dict[int, ModuleLayout] = {}
    for m, gene in genome.modules.items():
        x, y = bp.xy(m)
        pads = bp.pads_actual[m]
        azs = tuple(params.port_azimuth(p) for p in pads)
        proximal = 'A' if gene.crease == 0 else 'B'  # pad 0 = incoming side
        modules[m] = ModuleLayout(
            module_id=m,
            center=np.array([x, y, params.MODULE_HEIGHT / 2.0]),
            pads=pads,
            pad_azimuths=azs,
            polarity=gene.polarity,
            crease_index=gene.crease,
            fold_dir=gene.fold_dir,
            proximal_side=proximal,
        )

    parent_half: Dict[int, Tuple[int, Half]] = {}
    for e in genome.edges:
        child_in = bp.incoming_actual[e.child_id]
        parent_side_actual = (child_in + 3) % 6
        pl = modules[e.parent_id]
        pad_index = pl.pads.index(parent_side_actual)
        parent_half[e.child_id] = (e.parent_id, pl.half_of_pad(pad_index))

    return AssemblyLayout(genome=genome, blueprint=bp,
                          modules=modules, parent_half=parent_half)


# ------------------------------------------------------------------------- #
#  Kinematic fold predictor                                                  #
# ------------------------------------------------------------------------- #

def _rot_about_line(point: np.ndarray, axis: np.ndarray,
                    theta: float) -> np.ndarray:
    """4x4 homogeneous rotation by theta about the line (point, axis)."""
    a = axis / (np.linalg.norm(axis) + 1e-15)
    x, y, z = a
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = point - R @ point
    return T


def predict_fold(layout: AssemblyLayout) -> Dict[Tuple[int, Half], np.ndarray]:
    """Closed-form world transform of every half after folding.

    Returns X[(module, half)]: a 4x4 mapping world-FLAT coordinates of that
    half's material points to world-FOLDED coordinates. The root's proximal
    half is the fixed base (identity)."""
    X: Dict[Tuple[int, Half], np.ndarray] = {}
    kin_children = layout.kinematic_children()
    root = layout.genome.root_id

    def visit(module: int, X_prox: np.ndarray) -> None:
        ml = layout.modules[module]
        X[(module, 'P')] = X_prox
        hinge = _rot_about_line(ml.center, ml.crease_axis, ml.hinge_target)
        X_dist = X_prox @ hinge
        X[(module, 'D')] = X_dist
        for child in kin_children.get((module, 'P'), []):
            visit(child, X_prox)
        for child in kin_children.get((module, 'D'), []):
            visit(child, X_dist)

    visit(root, np.eye(4))
    return X


def predicted_m_eff(layout: AssemblyLayout) -> np.ndarray:
    """Closed-form effective dipole of the folded assembly: every magnet's
    direction rotated by its host half's predicted fold transform, summed.
    Note the physics this encodes: folding rotates only HALF of each module,
    so the flat-sheet 120-degree cancellation breaks on folding — same-sign
    modules acquire a net (mostly vertical) dipole from their tilted flaps."""
    X = predict_fold(layout)
    total = np.zeros(3)
    for m, ml in layout.modules.items():
        for mg in ml.magnets():
            R = X[(m, mg["half"])][:3, :3]
            total += R @ mg["dir"]
    return params.MAGNET_M0 * total


def half_sample_points(ml: ModuleLayout, half: Half) -> np.ndarray:
    """A few representative material points of one half (world-flat coords)
    for the cheap self-intersection gate."""
    n = ml.half_outward(half)
    t = np.array([n[1], -n[0], 0.0])
    c = ml.center
    R = params.MODULE_RADIUS
    centroid = c + (4.0 * R / (3.0 * math.pi)) * n
    return np.stack([
        centroid,
        c + 0.85 * R * n,
        c + 0.55 * R * (n + t) / math.sqrt(2.0),
        c + 0.55 * R * (n - t) / math.sqrt(2.0),
    ])


def predicted_self_intersection(layout: AssemblyLayout,
                                clearance: float = None) -> List[str]:
    """Conservative-light overlap check on the predicted folded pose.
    Only flags deep interpenetration between modules that are not tree-
    adjacent; MuJoCo's fold stage remains the physical arbiter."""
    if clearance is None:
        clearance = 0.75 * params.MODULE_RADIUS
    X = predict_fold(layout)
    pts: Dict[int, np.ndarray] = {}
    for m, ml in layout.modules.items():
        chunks = []
        for half in ('P', 'D'):
            p = half_sample_points(ml, half)
            hom = np.hstack([p, np.ones((len(p), 1))])
            chunks.append((X[(m, half)] @ hom.T).T[:, :3])
        pts[m] = np.vstack(chunks)

    adjacent = set()
    for e in layout.genome.edges:
        adjacent.add((e.parent_id, e.child_id))
        adjacent.add((e.child_id, e.parent_id))

    reasons = []
    ids = sorted(pts)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if (a, b) in adjacent:
                continue
            d = np.min(np.linalg.norm(
                pts[a][:, None, :] - pts[b][None, :, :], axis=-1))
            if d < clearance:
                reasons.append(
                    f"predicted overlap: modules {a},{b} (min sample "
                    f"distance {d * 1e3:.2f} mm < {clearance * 1e3:.2f} mm)")
    return reasons
