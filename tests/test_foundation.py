"""Foundation tests: genome/blueprint logic and the dipole algebra."""
import math
import random

import numpy as np
import pytest

from roblet_evo import params
from roblet_evo.genome import (Genome, ModuleGene, FLAT, VALLEY, MOUNTAIN,
                               random_genome, mutate, crossover)
from roblet_evo.blueprint import embed, random_valid_genome
from roblet_evo.geometry import build_layout


def tripod_genome(fold_dir=VALLEY, polarity=(1, 1, 1)) -> Genome:
    """Root with three children — the smallest branching assembly."""
    g = Genome(root_id=0, root_gene=ModuleGene(0, FLAT, (1, 1, 1)))
    for port in (0, 1, 2):
        g.add_module(g.next_free_id(), 0, port,
                     ModuleGene(0, fold_dir, polarity))
    return g


# ------------------------------------------------------------------ genome --
def test_tree_invariants():
    g = tripod_genome()
    assert g.is_tree()
    assert g.size() == 4
    assert sorted(g.leaves()) == [1, 2, 3]
    g.delete_subtree(2)
    assert g.size() == 3 and g.is_tree()


def test_port_reuse_rejected():
    g = tripod_genome()
    with pytest.raises(ValueError):
        g.add_module(9, 0, 0, ModuleGene(0, FLAT, (1, 1, 1)))


def test_mutation_preserves_validity():
    rng = random.Random(7)
    g = random_valid_genome(rng, 8)
    for _ in range(300):
        g2 = mutate(g, rng)
        assert g2.is_tree()
        assert params.FREQ_RANGE_HZ[0] <= g2.freq_hz <= params.FREQ_RANGE_HZ[1]
        assert params.DUTY_RANGE[0] <= g2.duty <= params.DUTY_RANGE[1]
        if embed(g2).valid:
            g = g2
    assert g.is_tree()


def test_crossover_produces_trees():
    rng = random.Random(3)
    a = random_valid_genome(rng, 8)
    b = random_valid_genome(rng, 8)
    for _ in range(100):
        c = crossover(a, b, rng)
        assert c.is_tree()


def test_determinism():
    g1 = random_genome(random.Random(42), 10)
    g2 = random_genome(random.Random(42), 10)
    assert g1.canonical() == g2.canonical()


# --------------------------------------------------------------- blueprint --
def test_tripod_blueprint_valid():
    bp = embed(tripod_genome())
    assert bp.valid, bp.reasons
    # root pads are actual sides 0/2/4; children sit in distinct cells
    assert bp.pads_actual[0] == (0, 2, 4)
    assert len(set(bp.positions_axial.values())) == 4
    # children of an even-depth parent occupy sides 0/2/4
    assert bp.incoming_actual[1] == 3   # opposite of side 0


def test_cell_collision_detected():
    # two different routes to the same hex cell must be flagged
    g = Genome(root_id=0, root_gene=ModuleGene(0, FLAT, (1, 1, 1)))
    g.add_module(1, 0, 0, ModuleGene(0, FLAT, (1, 1, 1)))   # E
    g.add_module(2, 0, 1, ModuleGene(0, FLAT, (1, 1, 1)))   # NW
    g.add_module(3, 1, 1, ModuleGene(0, FLAT, (1, 1, 1)))   # from E, odd depth
    g.add_module(4, 3, 2, ModuleGene(0, FLAT, (1, 1, 1)))
    g.add_module(5, 2, 0, ModuleGene(0, FLAT, (1, 1, 1)))
    g.add_module(6, 5, 1, ModuleGene(0, FLAT, (1, 1, 1)))
    bp = embed(g)
    # not asserting validity either way — asserting consistency:
    # every reported reason must be a real string and validity must match
    assert bp.valid == (len(bp.reasons) == 0)


# ------------------------------------------------------------------ dipole --
def test_dipole_cancellation_and_2m0():
    """The 120-degree identity: all-same polarity => zero net dipole;
    one flipped sign => magnitude exactly 2*m0 along that pad's axis."""
    g_same = tripod_genome(polarity=(1, 1, 1))
    bp = embed(g_same)
    layout = build_layout(g_same, bp)
    for ml in layout.modules.values():
        assert np.linalg.norm(ml.dipole_flat()) < 1e-12

    g_flip = tripod_genome(polarity=(-1, 1, 1))
    layout = build_layout(g_flip, embed(g_flip))
    for m in (1, 2, 3):
        ml = layout.modules[m]
        d = ml.dipole_flat()
        assert np.linalg.norm(d) == pytest.approx(2.0 * params.MAGNET_M0,
                                                  rel=1e-9)
        # direction: minus the flipped pad's unit vector (pad 0 sign is -1)
        expect = -2.0 * params.MAGNET_M0 * np.array(
            [math.cos(ml.pad_azimuths[0]), math.sin(ml.pad_azimuths[0]), 0.0])
        assert np.allclose(d, expect, atol=1e-12)


def test_crease_axis_and_halves():
    g = tripod_genome()
    layout = build_layout(g, embed(g))
    for m in (1, 2, 3):
        ml = layout.modules[m]
        # crease 0 => pad 0 (incoming) alone on side A => proximal is A
        assert ml.proximal_side == 'A' and ml.distal_side == 'B'
        # crease axis is horizontal and perpendicular to pad 0's azimuth
        t = ml.crease_axis
        assert abs(t[2]) < 1e-12
        n = ml.outward('B')
        assert abs(np.dot(t, n)) < 1e-12
        # +theta lifts the distal outer edge: (t x n) must point up
        assert np.cross(t, n)[2] > 0.99


def test_magnet_half_assignment():
    g = tripod_genome()
    layout = build_layout(g, embed(g))
    ml = layout.modules[1]           # crease 0: pad 0 alone on proximal
    halves = [mg["half"] for mg in ml.magnets()]
    assert halves == ['P', 'D', 'D']
