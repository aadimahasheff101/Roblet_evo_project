"""Task evaluation: genome in, fitness + behavior descriptors out.

Walker pipeline per candidate:
  blueprint gate -> predicted-fold gate -> MuJoCo fold -> rigid bake ->
  N locomotion seeds at the genome's own (freq, duty) -> median score.

Scores: toppled or unstable episodes score zero speed (a robot that flips is
not a usable operating point — same rule as the calibrated v7 analysis).
Fold failure scores 0 with a validity flag so the archive never keeps it.

Descriptors (normalized to [0, 1] for the CVT archive):
  d0  pitch amplitude   (deg / 45, clipped)   rocking vs vibration regime
  d1  airborne fraction (already 0..1)        ground-bound vs bouncing
  d2  module count      ((n - MIN) / (MAX - MIN))  size axis
  d3  straightness      net displacement / path length  wanderers vs liners
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import params
from .genome import Genome
from .blueprint import embed
from .geometry import build_layout, predicted_self_intersection
from .mjcf_articulated import build_articulated_xml
from .selffold import run_fold
from .bake_rigid import bake_rigid_xml
from .locomotion import load_rigid, run_episode

N_DESCRIPTORS = 4
SCREEN_SEEDS = 1
FULL_SEEDS = 3


@dataclass
class EvalResult:
    ok: bool                       # made it through fold + at least one episode
    fitness: float                 # median usable speed (m/s); 0 if not ok
    descriptors: Optional[np.ndarray]
    stage: str                     # how far it got / where it died
    reasons: List[str] = field(default_factory=list)
    episodes: List[dict] = field(default_factory=list)
    fold_error_deg: float = 0.0
    n_modules: int = 0
    wall_time_s: float = 0.0

    def as_record(self, genome: Genome) -> dict:
        return dict(
            genome=genome.to_dict(),
            ok=self.ok, fitness=self.fitness,
            descriptors=(None if self.descriptors is None
                         else [float(x) for x in self.descriptors]),
            stage=self.stage, reasons=self.reasons,
            episodes=self.episodes, fold_error_deg=self.fold_error_deg,
            n_modules=self.n_modules, wall_time_s=self.wall_time_s,
        )


def _descriptors(episodes: List[dict], n_modules: int) -> np.ndarray:
    pitch = float(np.mean([e["pitch_amp_deg"] for e in episodes]))
    airborne = float(np.mean([e["airborne_frac"] for e in episodes]))
    straight = float(np.mean([e.get("straightness", 0.0) for e in episodes]))
    span = max(params.MAX_MODULES - params.MIN_MODULES, 1)
    return np.clip(np.array([
        pitch / 45.0,
        airborne,
        (n_modules - params.MIN_MODULES) / span,
        straight,
    ]), 0.0, 1.0)


def evaluate_walker(genome: Genome,
                    n_seeds: int = FULL_SEEDS,
                    surface: str = "paper",
                    t_sim: float = None,
                    early_stop: bool = True) -> EvalResult:
    t0 = time.time()
    n_mod = genome.size()

    bp = embed(genome)
    if not bp.valid:
        return EvalResult(False, 0.0, None, "blueprint", bp.reasons[:4],
                          n_modules=n_mod, wall_time_s=time.time() - t0)

    layout = build_layout(genome, bp)
    overlap = predicted_self_intersection(layout)
    if overlap:
        return EvalResult(False, 0.0, None, "fold_predict", overlap[:4],
                          n_modules=n_mod, wall_time_s=time.time() - t0)

    xml = build_articulated_xml(genome, bp)
    try:
        fold = run_fold(xml)
    except Exception as exc:                      # compile/sim blow-up
        return EvalResult(False, 0.0, None, "fold_crash", [repr(exc)],
                          n_modules=n_mod, wall_time_s=time.time() - t0)
    if not fold.success:
        return EvalResult(False, 0.0, None, "fold", fold.reasons[:4],
                          fold_error_deg=fold.max_error_deg,
                          n_modules=n_mod, wall_time_s=time.time() - t0)

    rigid = bake_rigid_xml(fold, surface=surface)
    model, data, m_eff = load_rigid(rigid, surface=surface)

    episodes: List[dict] = []
    speeds: List[float] = []
    for seed in range(n_seeds):
        ep = run_episode(model, data, m_eff, genome.freq_hz, genome.duty,
                         seed=seed, t_sim=t_sim,
                         early_stop_t=(3.0 if early_stop else None))
        episodes.append(ep.as_dict())
        speeds.append(0.0 if (ep.toppled or ep.unstable) else ep.speed)

    return EvalResult(
        ok=True,
        fitness=float(np.median(speeds)),
        descriptors=_descriptors(episodes, n_mod),
        stage="done",
        episodes=episodes,
        fold_error_deg=fold.max_error_deg,
        n_modules=n_mod,
        wall_time_s=time.time() - t0,
    )


# ---------------------------------------------------------------- workers --
def _eval_task(args) -> dict:
    """Multiprocessing entry point: (genome_dict, kwargs) -> record dict."""
    genome_dict, kwargs = args
    g = Genome.from_dict(genome_dict)
    res = evaluate_walker(g, **kwargs)
    return res.as_record(g)
