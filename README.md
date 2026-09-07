# roblet_evo — v2 pipeline

Physics-in-the-loop evolution of self-folding Roblet morphologies actuated
by a global oscillating vertical magnetic field. Replaces the v1 geometric
pre-simulation fitness with MuJoCo evaluation in the hardware-calibrated
environment from the b-field optimisation work.

## Pipeline

    genome (tree + fold/polarity genes + field genes)
      -> blueprint.py     2D hex embedding, grammar gate          [no physics]
      -> geometry.py      kinematic layout + fold predictor gate  [no physics]
      -> mjcf_articulated  articulated MJCF (2 half-discs/module, hinge/crease)
      -> selffold.py      simultaneous thermal fold ramp, success gate
      -> bake_rigid.py    folded pose fused into one rigid body, m_eff summed
      -> locomotion.py    calibrated v7 episode (torque-only uniform field)
      -> evaluate.py      fitness + regime descriptors
      -> qd.py            CVT-MAP-Elites: age/immigrants, adaptive budget,
                          JSONL logging, snapshots, resumable

## Run

    pip install mujoco numpy pytest
    python -m pytest tests/            # 15 tests incl. kinematic equivalence
    python run_walker_qd.py --iterations 300 --out runs/walker_s1

## Physics model (see roblet_evo/params.py for provenance tags)

- Module: 12.5 mm dia x 5 mm, 0.44 g, folds along ONE of three diameters to
  +-45 deg (valley/mountain), rigid once folded. Connections are rigid welds.
- 3 magnets per module (F412, m0 = 6.5e-3 A m^2), radial, one per pad,
  present whether bonded or not; signs are polarity genes.
- Uniform vertical bipolar square-wave field (+-10 mT, coil lag 8 ms):
  torque only, tau = (R m_eff) x B — no net force by physics.
- Fold stage is field-free (thermal); all creases ramp simultaneously.

## Key verified facts (tests/)

- MuJoCo fold matches the closed-form kinematic predictor (<1.5 mm, <6 deg).
- Baked m_eff matches the analytic dipole prediction.
- Folding breaks the flat 120-degree magnet cancellation: same-sign modules
  acquire a net near-VERTICAL dipole (zero torque under the vertical field)
  — the polarity pattern is what makes a body actuatable.

## Still to build

- bridge env (blocks of varying height, static structural scoring)
- swimmer env (near-neutral buoyancy, MuJoCo fluid model)
- tri-walker benchmark + re-validation vs hardware video (12-magnet model)
- analysis suite (archive maps, stats, novelty distance), demo renderer
