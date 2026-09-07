"""CVT-MAP-Elites with lineage age, random immigrants, and adaptive budget.

Archive: k well-spread niches in descriptor space [0,1]^d (centroidal
Voronoi tessellation — robust in any dimension, unlike a regular grid).
Each niche keeps its best-ever candidate.

Anti-dominance machinery (agreed design):
  - every elite records its lineage_id and birth iteration ("age");
  - a fraction of parent draws is biased toward YOUNG lineages;
  - every batch includes brand-new random immigrants, so fresh ancestry
    keeps entering even late in the run.

Adaptive evaluation budget:
  - phase 1 screen: 1 seed with early termination of non-movers;
  - phase 2 full:   FULL_SEEDS only for candidates that would enter the
    archive (empty niche or screen fitness beating the incumbent).

Everything evaluated is appended to a JSONL log; the archive is snapshotted
periodically; runs are resumable from a snapshot. Nothing is lost.
"""
from __future__ import annotations

import json
import pickle
import random
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import params
from .genome import Genome, mutate, crossover, MutationRates
from .blueprint import embed, random_valid_genome
from .evaluate import (_eval_task, evaluate_walker, EvalResult,
                       N_DESCRIPTORS, SCREEN_SEEDS, FULL_SEEDS)


# ---------------------------------------------------------------- CVT ------
def build_cvt(n_cells: int, dim: int = N_DESCRIPTORS,
              seed: int = 0, n_samples: int = 40000,
              iters: int = 25) -> np.ndarray:
    """Deterministic k-means CVT over the unit hypercube."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n_samples, dim))
    centroids = pts[rng.choice(n_samples, n_cells, replace=False)].copy()
    for _ in range(iters):
        d = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=-1)
        assign = np.argmin(d, axis=1)
        for c in range(n_cells):
            mask = assign == c
            if np.any(mask):
                centroids[c] = pts[mask].mean(axis=0)
    return centroids


def cell_index(centroids: np.ndarray, desc: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(centroids - desc[None, :], axis=1)))


# ---------------------------------------------------------------- elite ----
@dataclass
class Elite:
    fitness: float
    descriptors: List[float]
    genome_dict: dict
    lineage_id: int
    birth_iter: int
    n_modules: int


@dataclass
class QDConfig:
    n_cells: int = 128
    batch_size: int = 32
    iterations: int = 200
    n_init: int = 64
    immigrant_frac: float = 0.10
    p_crossover: float = 0.40
    p_young_parent: float = 0.20    # parent draw biased to youngest quartile
    young_window: int = 20          # iterations counted as "young"
    workers: int = 0                # 0 = os.cpu_count()
    seed: int = 1
    snapshot_every: int = 10
    init_modules: Tuple[int, int] = (params.MIN_MODULES, 12)
    surface: str = "paper"
    out_dir: str = "runs/walker"


class MapElites:
    def __init__(self, cfg: QDConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.centroids = build_cvt(cfg.n_cells, seed=cfg.seed)
        self.archive: Dict[int, Elite] = {}
        self.iteration = 0
        self._next_lineage = 0
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "evals.jsonl"

    # ------------------------------------------------------------ helpers
    def _new_lineage(self) -> int:
        self._next_lineage += 1
        return self._next_lineage

    def _log(self, phase: str, record: dict) -> None:
        record = dict(record)
        record["iter"] = self.iteration
        record["phase"] = phase
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _immigrant(self) -> Genome:
        """Sample a fresh random genome; if a size proves hard to embed,
        fall back to smaller sizes rather than killing the run."""
        n = self.rng.randint(*self.cfg.init_modules)
        g = None
        while g is None:
            try:
                g = random_valid_genome(self.rng, n)
            except RuntimeError:
                if n <= params.MIN_MODULES:
                    raise
                n -= 1
        g.lineage_id = self._new_lineage()
        g.birth_gen = self.iteration
        return g

    def _pick_parent(self) -> Genome:
        cells = list(self.archive)
        if self.rng.random() < self.cfg.p_young_parent:
            young = [c for c in cells
                     if self.iteration - self.archive[c].birth_iter
                     <= self.cfg.young_window]
            if young:
                cells = young
        e = self.archive[self.rng.choice(cells)]
        g = Genome.from_dict(e.genome_dict)
        g.lineage_id = e.lineage_id
        return g

    def _make_batch(self) -> List[Genome]:
        batch: List[Genome] = []
        n_imm = max(1, int(self.cfg.immigrant_frac * self.cfg.batch_size))
        while len(batch) < n_imm:
            try:
                batch.append(self._immigrant())
            except RuntimeError:
                break
        while len(batch) < self.cfg.batch_size:
            if not self.archive:
                batch.append(self._immigrant())
                continue
            p1 = self._pick_parent()
            if self.rng.random() < self.cfg.p_crossover and len(self.archive) > 1:
                child = crossover(p1, self._pick_parent(), self.rng)
            else:
                child = p1.copy()
            child = mutate(child, self.rng)
            child.lineage_id = p1.lineage_id
            child.birth_gen = self.iteration
            if embed(child).valid:      # cheap gate before spending sim time
                batch.append(child)
        return batch

    def _target_cell(self, desc: List[float]) -> int:
        return cell_index(self.centroids, np.asarray(desc))

    def _would_enter(self, fitness: float, desc: List[float]) -> bool:
        cell = self._target_cell(desc)
        return (cell not in self.archive
                or fitness > self.archive[cell].fitness)

    def _insert(self, record: dict) -> bool:
        if not record["ok"] or record["descriptors"] is None:
            return False
        cell = self._target_cell(record["descriptors"])
        incumbent = self.archive.get(cell)
        if incumbent is not None and record["fitness"] <= incumbent.fitness:
            return False
        gd = record["genome"]
        self.archive[cell] = Elite(
            fitness=record["fitness"],
            descriptors=record["descriptors"],
            genome_dict=gd,
            lineage_id=gd.get("lineage_id") or 0,
            birth_iter=self.iteration,
            n_modules=record["n_modules"],
        )
        return True

    # ------------------------------------------------------------ metrics
    def coverage(self) -> float:
        return len(self.archive) / self.cfg.n_cells

    def qd_score(self) -> float:
        return float(sum(e.fitness for e in self.archive.values()))

    def best(self) -> Optional[Elite]:
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda e: e.fitness)

    # ----------------------------------------------------------- snapshot
    def snapshot(self) -> None:
        payload = dict(cfg=self.cfg, centroids=self.centroids,
                       archive=self.archive, iteration=self.iteration,
                       rng_state=self.rng.getstate(),
                       next_lineage=self._next_lineage)
        with (self.out / f"snapshot_{self.iteration:05d}.pkl").open("wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def resume(path: str) -> "MapElites":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        me = MapElites(payload["cfg"])
        me.centroids = payload["centroids"]
        me.archive = payload["archive"]
        me.iteration = payload["iteration"]
        me.rng.setstate(payload["rng_state"])
        me._next_lineage = payload["next_lineage"]
        return me

    # ---------------------------------------------------------------- run
    def run(self, pool: Optional[Pool] = None, verbose: bool = True) -> None:
        cfg = self.cfg
        own_pool = pool is None
        if own_pool:
            import os
            pool = Pool(cfg.workers or os.cpu_count())
        try:
            if self.iteration == 0 and not self.archive:
                init = [self._immigrant() for _ in range(cfg.n_init)]
                self._evaluate_and_insert(init, pool)
                self.snapshot()
            while self.iteration < cfg.iterations:
                self.iteration += 1
                batch = self._make_batch()
                n_ins = self._evaluate_and_insert(batch, pool)
                if verbose:
                    b = self.best()
                    print(f"iter {self.iteration:04d} | archive "
                          f"{len(self.archive):4d}/{cfg.n_cells} | inserted "
                          f"{n_ins:3d} | best {0 if b is None else b.fitness * 1e3:7.2f} mm/s "
                          f"| QD {self.qd_score() * 1e3:8.1f}", flush=True)
                if self.iteration % cfg.snapshot_every == 0:
                    self.snapshot()
            self.snapshot()
        finally:
            if own_pool:
                pool.close()
                pool.join()

    def _evaluate_and_insert(self, batch: List[Genome], pool: Pool) -> int:
        screen_kwargs = dict(n_seeds=SCREEN_SEEDS, early_stop=True,
                             surface=self.cfg.surface)
        full_kwargs = dict(n_seeds=FULL_SEEDS, early_stop=True,
                           surface=self.cfg.surface)

        screen = pool.map(
            _eval_task, [(g.to_dict(), screen_kwargs) for g in batch])
        for rec in screen:
            self._log("screen", rec)

        promote = [(g, rec) for g, rec in zip(batch, screen)
                   if rec["ok"] and rec["descriptors"] is not None
                   and self._would_enter(rec["fitness"], rec["descriptors"])]
        if not promote:
            return 0
        full = pool.map(
            _eval_task, [(g.to_dict(), full_kwargs) for g, _ in promote])
        inserted = 0
        for rec in full:
            self._log("full", rec)
            if self._insert(rec):
                inserted += 1
        return inserted
