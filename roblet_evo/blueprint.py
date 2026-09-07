"""2D blueprint embedding — the cheap validity gate.

Places the genome's tree on the triangular hex lattice using the depth-parity
port mapping (even depth uses actual sides 0/2/4, odd depth 1/3/5), and
checks the three grammar rules: every module gets a unique cell, no actual
port is used twice, and the structure is a tree. Invalid genomes are rejected
here before any physics is spent on them.

Also derives, for every module, the data the 3D stages need:
  - axial cell and cartesian centre position
  - depth (for parity)
  - the three actual hex sides its pads occupy (pad list, canonical order:
    pad 0 = incoming side for non-root modules, [0, 2, 4] for the root)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from . import params
from .genome import Genome


@dataclass
class BlueprintResult:
    valid: bool
    reasons: List[str]
    positions_axial: Dict[int, Tuple[int, int]]
    depth: Dict[int, int]
    pads_actual: Dict[int, Tuple[int, int, int]]   # 3 actual sides per module
    incoming_actual: Dict[int, int]                # actual incoming side (-1 root)

    def xy(self, module_id: int) -> Tuple[float, float]:
        q, r = self.positions_axial[module_id]
        return params.axial_to_xy(q, r)


def _actual_port(parent_depth: int, logical_port: int) -> int:
    table = (params.EVEN_DEPTH_PORTS if parent_depth % 2 == 0
             else params.ODD_DEPTH_PORTS)
    return table[logical_port]


def embed(genome: Genome) -> BlueprintResult:
    reasons: List[str] = []
    if not genome.is_tree():
        return BlueprintResult(False, ["not a rooted tree"], {}, {}, {}, {})

    root = genome.root_id
    positions: Dict[int, Tuple[int, int]] = {root: (0, 0)}
    depth: Dict[int, int] = {root: 0}
    occupied: Dict[Tuple[int, int], int] = {(0, 0): root}
    used_sides: Dict[int, set] = {m: set() for m in genome.modules}
    incoming: Dict[int, int] = {root: -1}

    ch = genome.children_map()
    edges_in = genome.parent_map()
    stack = [root]
    while stack:
        pid = stack.pop()
        pq, pr = positions[pid]
        pdepth = depth[pid]
        for cid in ch[pid]:
            e = edges_in[cid]
            side = _actual_port(pdepth, e.parent_port)
            child_in = (side + 3) % 6

            if side in used_sides[pid]:
                reasons.append(f"parent {pid} reuses actual side {side}")
            if child_in in used_sides[cid]:
                reasons.append(f"child {cid} incoming side {child_in} occupied")
            used_sides[pid].add(side)
            used_sides[cid].add(child_in)

            dq, dr = params.DIRS_AXIAL[side]
            cell = (pq + dq, pr + dr)
            if cell in occupied:
                reasons.append(
                    f"cell {cell} collision: modules {occupied[cell]} and {cid}")
            else:
                occupied[cell] = cid
            positions[cid] = cell
            depth[cid] = pdepth + 1
            incoming[cid] = child_in
            stack.append(cid)

    pads: Dict[int, Tuple[int, int, int]] = {}
    for m in genome.modules:
        if m == root:
            pads[m] = (0, 2, 4)
        else:
            a = incoming[m]
            pads[m] = (a, (a + 2) % 6, (a + 4) % 6)

    return BlueprintResult(
        valid=(len(reasons) == 0),
        reasons=reasons,
        positions_axial=positions,
        depth=depth,
        pads_actual=pads,
        incoming_actual=incoming,
    )


def random_valid_genome(rng, n_modules: int, max_tries: int = 500) -> Genome:
    """Rejection-sample a random genome that passes the blueprint gate."""
    from .genome import random_genome
    for _ in range(max_tries):
        g = random_genome(rng, n_modules)
        if g.size() == n_modules and embed(g).valid:
            return g
    raise RuntimeError(
        f"could not sample a valid {n_modules}-module genome in {max_tries} tries")
