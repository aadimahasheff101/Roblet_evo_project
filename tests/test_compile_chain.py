"""Compile-chain tests: genome -> MJCF -> fold -> bake -> locomotion.

The key correctness test is kinematic equivalence: with gravity off, the
MuJoCo fold must land every half-disc where the closed-form predictor says
(relative to the floating base) — the two implementations share only the
convention, not the code, so agreement is a real check.
"""
import math
import random

import mujoco
import numpy as np
import pytest

from roblet_evo import params
from roblet_evo.genome import Genome, ModuleGene, FLAT, VALLEY, MOUNTAIN
from roblet_evo.blueprint import embed
from roblet_evo.geometry import build_layout, predict_fold
from roblet_evo.mjcf_articulated import build_articulated_xml
from roblet_evo.selffold import run_fold, magnet_records
from roblet_evo.bake_rigid import bake_rigid_xml
from roblet_evo.locomotion import load_rigid, run_episode


def tripod(fold_dir=VALLEY, polarity=(-1, 1, 1), crease=0):
    g = Genome(root_id=0, root_gene=ModuleGene(0, FLAT, (1, 1, 1)),
               freq_hz=6.25, duty=0.4)
    for port in (0, 1, 2):
        g.add_module(g.next_free_id(), 0, port,
                     ModuleGene(crease, fold_dir, polarity))
    return g


def test_articulated_model_compiles_and_masses():
    g = tripod()
    xml = build_articulated_xml(g)
    model = mujoco.MjModel.from_xml_string(xml)
    # one free joint + one hinge per module
    assert model.nu == g.size()
    assert model.njnt == g.size() + 1
    total = float(np.sum(model.body_mass))
    assert total == pytest.approx(g.size() * params.MODULE_MASS, rel=0.02)


def test_fold_matches_kinematic_predictor():
    g = tripod(fold_dir=VALLEY)
    bp = embed(g)
    layout = build_layout(g, bp)
    X = predict_fold(layout)

    xml = build_articulated_xml(g, bp, gravity=False, with_floor=False)
    res = run_fold(xml)
    assert res.success, res.reasons

    model, data = res.model, res.data
    root = g.root_id
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"m{root}_P")
    Tb = np.eye(4)
    Tb[:3, :3] = data.xmat[base_bid].reshape(3, 3)
    Tb[:3, 3] = data.xpos[base_bid]
    Tb_inv = np.linalg.inv(Tb)

    F_root = np.eye(4)
    F_root[:3, 3] = layout.modules[root].center

    for m, ml in layout.modules.items():
        for half in ('P', 'D'):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"m{m}_{half}")
            T = np.eye(4)
            T[:3, :3] = data.xmat[bid].reshape(3, 3)
            T[:3, 3] = data.xpos[bid]
            T_rel_mj = Tb_inv @ T

            F_b = np.eye(4)
            F_b[:3, 3] = ml.center
            T_rel_pred = np.linalg.inv(F_root) @ X[(m, half)] @ F_b

            dp = np.linalg.norm(T_rel_mj[:3, 3] - T_rel_pred[:3, 3])
            assert dp < 1.5e-3, f"m{m}_{half} pos off by {dp*1e3:.2f} mm"
            R_err = T_rel_mj[:3, :3].T @ T_rel_pred[:3, :3]
            ang = math.degrees(math.acos(
                np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
            assert ang < 6.0, f"m{m}_{half} rotation off by {ang:.2f} deg"


def test_fold_on_floor_succeeds_and_bakes():
    g = tripod(fold_dir=VALLEY, polarity=(-1, 1, 1))
    xml = build_articulated_xml(g)
    res = run_fold(xml)
    assert res.success, res.reasons
    assert res.max_error_deg < math.degrees(params.FOLD_TOLERANCE_RAD)

    rigid = bake_rigid_xml(res)
    model, data, m_eff = load_rigid(rigid)
    assert float(np.sum(model.body_mass)) == pytest.approx(
        g.size() * params.MODULE_MASS, rel=0.02)
    assert np.all(np.isfinite(m_eff))
    # mixed polarity on every leg + folded rotations => non-zero net dipole
    assert np.linalg.norm(m_eff) > 0.1 * params.MAGNET_M0


def test_baked_meff_matches_analytic_prediction():
    """Folding breaks the flat 120-degree cancellation: the distal magnets
    tilt out of plane, so an all-same-polarity tripod acquires a NET VERTICAL
    dipole of 3 legs x 2 magnets x 0.5 x sin(45deg) x m0. The baked m_eff must
    match the closed-form predictor (near-vertical, correct magnitude)."""
    from roblet_evo.geometry import predicted_m_eff
    g = tripod(fold_dir=VALLEY, polarity=(1, 1, 1))
    g.modules[0] = ModuleGene(0, FLAT, (1, 1, 1))
    bp = embed(g)
    m_pred = predicted_m_eff(build_layout(g, bp))
    expected_z = 3 * 2 * 0.5 * math.sin(params.FOLD_DIHEDRAL_RAD) * params.MAGNET_M0
    assert m_pred[2] == pytest.approx(expected_z, rel=1e-9)
    assert np.linalg.norm(m_pred[:2]) < 1e-12

    res = run_fold(build_articulated_xml(g))
    assert res.success, res.reasons
    _, _, m_eff = load_rigid(bake_rigid_xml(res))
    # simulated fold reaches its target within the servo tolerance, so the
    # baked dipole tracks the analytic one to a few percent
    assert m_eff[2] == pytest.approx(m_pred[2], rel=0.06)
    assert np.linalg.norm(m_eff[:2]) < 0.02 * abs(m_pred[2])
    # near-vertical dipole under a vertical field => almost no torque:
    # all-same-polarity assemblies are close to inert, by physics.


def test_bake_preserves_folded_geometry():
    """REGRESSION (mesh-recentering double-application): the baked rigid
    body's geoms must sit exactly where the articulated fold left them.
    Compares all pairwise inter-geom distances AND orientations."""
    g = tripod(fold_dir=VALLEY, polarity=(-1, 1, 1))
    fold = run_fold(build_articulated_xml(g))
    assert fold.success, fold.reasons
    am, ad = fold.model, fold.data
    mujoco.mj_forward(am, ad)
    art_p = np.array([ad.geom_xpos[i] for i in range(am.ngeom)
                      if am.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH])
    art_R = [ad.geom_xmat[i].reshape(3, 3) for i in range(am.ngeom)
             if am.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]

    bm = mujoco.MjModel.from_xml_string(bake_rigid_xml(fold))
    bd = mujoco.MjData(bm)
    mujoco.mj_forward(bm, bd)
    bak_p = np.array([bd.geom_xpos[i] for i in range(bm.ngeom)
                      if bm.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH])
    bak_R = [bd.geom_xmat[i].reshape(3, 3) for i in range(bm.ngeom)
             if bm.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]

    assert len(art_p) == len(bak_p)
    da = np.linalg.norm(art_p[:, None] - art_p[None, :], axis=-1)
    db = np.linalg.norm(bak_p[:, None] - bak_p[None, :], axis=-1)
    assert np.max(np.abs(da - db)) < 1e-4, \
        f"baked shape distorted by {np.max(np.abs(da-db))*1e3:.3f} mm"
    # the baked frame is the articulated world frame translated only (axes
    # world-aligned), so every geom's orientation must match EXACTLY
    for Ra, Rb in zip(art_R, bak_R):
        rel = Ra.T @ Rb
        ang = math.degrees(math.acos(np.clip((np.trace(rel) - 1) / 2,
                                             -1, 1)))
        assert ang < 0.5, f"geom orientation drift {ang:.2f} deg"


def test_locomotion_smoke():
    g = tripod(fold_dir=VALLEY, polarity=(-1, 1, 1))
    res = run_fold(build_articulated_xml(g))
    assert res.success, res.reasons
    model, data, m_eff = load_rigid(bake_rigid_xml(res))
    out = run_episode(model, data, m_eff, freq_hz=6.25, duty=0.4,
                      seed=0, t_sim=2.0)
    assert np.isfinite(out.speed) and out.unstable == 0
    assert 0.0 <= out.airborne_frac <= 1.0
