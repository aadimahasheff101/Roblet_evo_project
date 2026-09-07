"""Genome: the evolvable description of one roblet morphology.

A genome is a rooted tree of modules plus per-module fold/magnet genes and
two global field genes. It intentionally contains no geometry — geometry is
derived deterministically by blueprint.py (2D) and geometry.py (3D).

Genes
-----
per edge   : parent_port in {0, 1, 2}      logical port on the parent
per module : crease      in {0, 1, 2}      crease perpendicular to pad[crease]
             fold_dir    in {+1, -1, 0}    valley / mountain / flat (no fold)
             polarity    (s0, s1, s2)      magnet sign per pad, each +-1
global     : freq_hz, duty                 field waveform, bounded by envelope

Polarity note: hardware may constrain the three signs to manufactured
variants (A/A'). Until that is confirmed, the genome stores all three signs
(the superset); a variant constraint later just restricts the mutation
operator, not the data model.

All stochastic operators take an explicit random.Random for reproducibility.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from . import params

VALLEY = +1
MOUNTAIN = -1
FLAT = 0

FOLD_DIRS = (VALLEY, MOUNTAIN, FLAT)
LOGICAL_PORTS = (0, 1, 2)


@dataclass(frozen=True)
class ModuleGene:
    crease: int                       # 0..2 (index into the module's pad list)
    fold_dir: int                     # +1 valley, -1 mountain, 0 flat
    polarity: Tuple[int, int, int]    # magnet sign per pad (pad-list order)

    def __post_init__(self):
        if self.crease not in (0, 1, 2):
            raise ValueError(f"crease must be 0..2, got {self.crease}")
        if self.fold_dir not in FOLD_DIRS:
            raise ValueError(f"fold_dir must be +-1 or 0, got {self.fold_dir}")
        if len(self.polarity) != 3 or any(s not in (-1, 1) for s in self.polarity):
            raise ValueError(f"polarity must be three +-1 signs, got {self.polarity}")


@dataclass(frozen=True)
class Edge:
    parent_id: int
    child_id: int
    parent_port: int   # logical 0..2

    def __post_init__(self):
        if self.parent_port not in LOGICAL_PORTS:
            raise ValueError(f"parent_port must be 0..2, got {self.parent_port}")


class Genome:
    """Mutable container for one candidate. Copy before mutating elsewhere."""

    def __init__(self, root_id: int = 0,
                 root_gene: Optional[ModuleGene] = None,
                 freq_hz: float = 6.25, duty: float = 0.5):
        self.root_id = root_id
        self.modules: Dict[int, ModuleGene] = {
            root_id: root_gene or ModuleGene(0, FLAT, (1, 1, 1))}
        self.edges: List[Edge] = []
        self.freq_hz = float(freq_hz)
        self.duty = float(duty)
        # lineage metadata (age / anti-dominance machinery)
        self.lineage_id: Optional[int] = None
        self.birth_gen: int = 0

    # ------------------------------------------------------------ structure
    def size(self) -> int:
        return len(self.modules)

    def children_map(self) -> Dict[int, List[int]]:
        ch: Dict[int, List[int]] = {mid: [] for mid in self.modules}
        for e in self.edges:
            ch[e.parent_id].append(e.child_id)
        return ch

    def parent_map(self) -> Dict[int, Edge]:
        return {e.child_id: e for e in self.edges}

    def leaves(self) -> List[int]:
        ch = self.children_map()
        return [m for m in self.modules if m != self.root_id and not ch[m]]

    def used_ports(self, parent_id: int) -> List[int]:
        return [e.parent_port for e in self.edges if e.parent_id == parent_id]

    def blocked_ports(self) -> Dict[int, Optional[int]]:
        """For every module, the logical outgoing port (if any) whose actual
        hex side coincides with the module's own INCOMING side under the
        depth-parity mapping. Using it always fails the blueprint gate, so
        growth/mutation must never offer it. Root has no incoming: None."""
        from . import params
        blocked: Dict[int, Optional[int]] = {self.root_id: None}
        depth = {self.root_id: 0}
        ch = self.children_map()
        pm = self.parent_map()
        stack = [self.root_id]
        while stack:
            n = stack.pop()
            table = (params.EVEN_DEPTH_PORTS if depth[n] % 2 == 0
                     else params.ODD_DEPTH_PORTS)
            for c in ch.get(n, []):
                e = pm[c]
                side = table[e.parent_port]
                depth[c] = depth[n] + 1
                inc = (side + 3) % 6
                ctable = (params.EVEN_DEPTH_PORTS if depth[c] % 2 == 0
                          else params.ODD_DEPTH_PORTS)
                blocked[c] = ctable.index(inc) if inc in ctable else None
                stack.append(c)
        return blocked

    def free_ports(self, parent_id: int) -> List[int]:
        """Logical ports usable for a NEW child: not already used, and not
        the port that maps onto this module's own incoming side."""
        used = set(self.used_ports(parent_id))
        bp = self.blocked_ports().get(parent_id)
        if bp is not None:
            used.add(bp)
        return [p for p in LOGICAL_PORTS if p not in used]

    def subtree_nodes(self, subroot: int) -> List[int]:
        ch = self.children_map()
        out, stack = [], [subroot]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(ch.get(n, []))
        return out

    def next_free_id(self) -> int:
        return max(self.modules) + 1

    def depth_map(self) -> Dict[int, int]:
        depth = {self.root_id: 0}
        ch = self.children_map()
        stack = [self.root_id]
        while stack:
            n = stack.pop()
            for c in ch[n]:
                depth[c] = depth[n] + 1
                stack.append(c)
        return depth

    def is_tree(self) -> bool:
        parent_count = {m: 0 for m in self.modules}
        for e in self.edges:
            if e.child_id not in parent_count or e.parent_id not in self.modules:
                return False
            parent_count[e.child_id] += 1
        for m, k in parent_count.items():
            if m == self.root_id and k != 0:
                return False
            if m != self.root_id and k != 1:
                return False
        # reachability / acyclicity
        seen, stack = set(), [self.root_id]
        ch = self.children_map()
        while stack:
            n = stack.pop()
            if n in seen:
                return False
            seen.add(n)
            stack.extend(ch[n])
        return len(seen) == len(self.modules)

    # ---------------------------------------------------------------- edit
    def add_module(self, module_id: int, parent_id: int, parent_port: int,
                   gene: ModuleGene) -> None:
        if module_id in self.modules:
            raise ValueError(f"module {module_id} already exists")
        if parent_id not in self.modules:
            raise ValueError(f"parent {parent_id} does not exist")
        if parent_port in self.used_ports(parent_id):
            raise ValueError(f"parent {parent_id} port {parent_port} in use")
        self.modules[module_id] = gene
        self.edges.append(Edge(parent_id, module_id, parent_port))

    def delete_subtree(self, subroot: int) -> None:
        doomed = set(self.subtree_nodes(subroot))
        if self.root_id in doomed:
            raise ValueError("cannot delete the root subtree")
        self.modules = {m: g for m, g in self.modules.items() if m not in doomed}
        self.edges = [e for e in self.edges
                      if e.parent_id not in doomed and e.child_id not in doomed]

    def copy(self) -> "Genome":
        g = Genome(self.root_id, self.modules[self.root_id],
                   self.freq_hz, self.duty)
        g.modules = dict(self.modules)
        g.edges = list(self.edges)
        g.lineage_id = self.lineage_id
        g.birth_gen = self.birth_gen
        return g

    # ------------------------------------------------------------ serialize
    def to_dict(self) -> dict:
        return dict(
            root_id=self.root_id,
            modules={str(m): [g.crease, g.fold_dir, list(g.polarity)]
                     for m, g in self.modules.items()},
            edges=[[e.parent_id, e.child_id, e.parent_port]
                   for e in self.edges],
            freq_hz=self.freq_hz, duty=self.duty,
            lineage_id=self.lineage_id, birth_gen=self.birth_gen,
        )

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        root = int(d["root_id"])
        rm = d["modules"][str(root)]
        g = Genome(root, ModuleGene(rm[0], rm[1], tuple(rm[2])),
                   d["freq_hz"], d["duty"])
        for m, spec in d["modules"].items():
            if int(m) != root:
                g.modules[int(m)] = ModuleGene(spec[0], spec[1], tuple(spec[2]))
        g.edges = [Edge(p, c, port) for p, c, port in d["edges"]]
        g.lineage_id = d.get("lineage_id")
        g.birth_gen = int(d.get("birth_gen", 0))
        return g

    # -------------------------------------------------------------- digest
    def canonical(self) -> tuple:
        """Hashable canonical form (for dedup / logging)."""
        mods = tuple(sorted((m, g.crease, g.fold_dir, g.polarity)
                            for m, g in self.modules.items()))
        edges = tuple(sorted((e.parent_id, e.child_id, e.parent_port)
                             for e in self.edges))
        return (self.root_id, mods, edges,
                round(self.freq_hz, 4), round(self.duty, 4))


# ------------------------------------------------------------------ random --
def random_gene(rng: random.Random) -> ModuleGene:
    return ModuleGene(
        crease=rng.randrange(3),
        fold_dir=rng.choice(FOLD_DIRS),
        polarity=tuple(rng.choice((-1, 1)) for _ in range(3)),
    )


def random_genome(rng: random.Random, n_modules: int) -> Genome:
    """Grow a random tree. Structural validity only — the blueprint gate
    (2D embedding) is applied by the caller via rejection sampling."""
    g = Genome(root_id=0, root_gene=random_gene(rng),
               freq_hz=rng.uniform(*params.FREQ_RANGE_HZ),
               duty=rng.uniform(*params.DUTY_RANGE))
    while g.size() < n_modules:
        candidates = [(m, p) for m in g.modules for p in g.free_ports(m)]
        if not candidates:
            break
        parent, port = rng.choice(candidates)
        g.add_module(g.next_free_id(), parent, port, random_gene(rng))
    return g


# --------------------------------------------------------------- mutation --
@dataclass(frozen=True)
class MutationRates:
    p_add_leaf: float = 0.18
    p_delete_leaf: float = 0.25
    p_reattach_subtree: float = 0.20
    p_change_port: float = 0.25
    p_change_crease: float = 0.25
    p_flip_fold_dir: float = 0.20
    p_flip_polarity: float = 0.25
    p_perturb_freq: float = 0.30
    p_perturb_duty: float = 0.30
    freq_sigma_hz: float = 0.8
    duty_sigma: float = 0.06


def mutate(genome: Genome, rng: random.Random,
           rates: MutationRates = MutationRates()) -> Genome:
    g = genome.copy()

    # --- structural -------------------------------------------------------
    if rng.random() < rates.p_add_leaf and g.size() < params.MAX_MODULES:
        candidates = [(m, p) for m in g.modules for p in g.free_ports(m)]
        if candidates:
            parent, port = rng.choice(candidates)
            g.add_module(g.next_free_id(), parent, port, random_gene(rng))

    if rng.random() < rates.p_delete_leaf and g.size() > params.MIN_MODULES:
        lf = g.leaves()
        if lf:
            g.delete_subtree(rng.choice(lf))

    if rng.random() < rates.p_reattach_subtree and g.size() > 2:
        movable = [m for m in g.modules if m != g.root_id]
        if movable:
            subroot = rng.choice(movable)
            doomed = set(g.subtree_nodes(subroot))
            targets = [(m, p) for m in g.modules if m not in doomed
                       for p in g.free_ports(m)]
            # the subroot's current slot frees up once detached; exclude it
            pm = g.parent_map()
            old = pm[subroot]
            targets = [(m, p) for (m, p) in targets
                       if not (m == old.parent_id and p == old.parent_port)]
            if targets:
                new_parent, new_port = rng.choice(targets)
                g.edges = [e for e in g.edges if e.child_id != subroot]
                g.edges.append(Edge(new_parent, subroot, new_port))

    if rng.random() < rates.p_change_port and g.edges:
        e = rng.choice(g.edges)
        free = [p for p in g.free_ports(e.parent_id)]
        if free:
            new_port = rng.choice(free)
            g.edges = [Edge(x.parent_id, x.child_id, new_port)
                       if x is e else x for x in g.edges]

    # --- per-module genes --------------------------------------------------
    if rng.random() < rates.p_change_crease:
        m = rng.choice(list(g.modules))
        gene = g.modules[m]
        g.modules[m] = replace(gene, crease=rng.choice(
            [c for c in (0, 1, 2) if c != gene.crease]))

    if rng.random() < rates.p_flip_fold_dir:
        m = rng.choice(list(g.modules))
        gene = g.modules[m]
        g.modules[m] = replace(gene, fold_dir=rng.choice(
            [d for d in FOLD_DIRS if d != gene.fold_dir]))

    if rng.random() < rates.p_flip_polarity:
        m = rng.choice(list(g.modules))
        gene = g.modules[m]
        i = rng.randrange(3)
        pol = list(gene.polarity)
        pol[i] = -pol[i]
        g.modules[m] = replace(gene, polarity=tuple(pol))

    # --- global field genes -------------------------------------------------
    if rng.random() < rates.p_perturb_freq:
        lo, hi = params.FREQ_RANGE_HZ
        g.freq_hz = min(hi, max(lo, g.freq_hz + rng.gauss(0.0, rates.freq_sigma_hz)))
    if rng.random() < rates.p_perturb_duty:
        lo, hi = params.DUTY_RANGE
        g.duty = min(hi, max(lo, g.duty + rng.gauss(0.0, rates.duty_sigma)))

    return g


# -------------------------------------------------------------- crossover --
def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Subtree crossover: replace a random subtree of A with one from B.
    Module genes travel with the donated subtree; IDs are remapped."""
    child = a.copy()
    a_cut_candidates = [m for m in child.modules if m != child.root_id]
    b_cut_candidates = [m for m in b.modules if m != b.root_id]
    if not a_cut_candidates or not b_cut_candidates:
        return child

    cut_a = rng.choice(a_cut_candidates)
    cut_b = rng.choice(b_cut_candidates)

    attach = child.parent_map()[cut_a]
    child.delete_subtree(cut_a)

    donated = b.subtree_nodes(cut_b)
    id_map: Dict[int, int] = {}
    next_id = child.next_free_id()
    for old in donated:
        id_map[old] = next_id
        next_id += 1

    # subtree root first (via the freed attachment slot), then interior edges
    child.modules[id_map[cut_b]] = b.modules[cut_b]
    child.edges.append(Edge(attach.parent_id, id_map[cut_b], attach.parent_port))
    b_edges = {e.child_id: e for e in b.edges}
    for old in donated:
        if old == cut_b:
            continue
        e = b_edges[old]
        child.modules[id_map[old]] = b.modules[old]
        child.edges.append(Edge(id_map[e.parent_id], id_map[old], e.parent_port))

    # inherit one parent's field genes with a coin flip per gene
    child.freq_hz = a.freq_hz if rng.random() < 0.5 else b.freq_hz
    child.duty = a.duty if rng.random() < 0.5 else b.duty
    return child


# ------------------------------------------------------------ size policy --
def clamp_size(genome: Genome, rng: random.Random) -> Genome:
    """Trim random leaves down to MAX_MODULES (never grows — growth is the
    add-leaf operator's job; forced random growth adds noise, not signal)."""
    g = genome.copy()
    while g.size() > params.MAX_MODULES:
        lf = g.leaves()
        if not lf:
            break
        g.delete_subtree(rng.choice(lf))
    return g
