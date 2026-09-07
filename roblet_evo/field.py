"""Global field model: bipolar square wave seen through the coil's L/R lag.

Ported from the hardware-calibrated v7 environment (Aaditya's
bfield_optimisation work): B(t) targets +B for duty*T of each cycle and -B
for the rest, filtered by a first-order lag with time constant B_TAU
(coil L/R plus driver dynamics, fitted to Chen et al. IROS 2026)."""
from __future__ import annotations

from . import params


class Field:
    def __init__(self, b_mag: float = None, b_tau: float = None):
        self.b_mag = params.B_MAG if b_mag is None else b_mag
        self.b_tau = params.B_TAU if b_tau is None else b_tau
        self.Bz = 0.0

    def step(self, t: float, period: float, duty: float, dt: float) -> float:
        target = self.b_mag if (t % period) < duty * period else -self.b_mag
        if self.b_tau <= 0.0:
            self.Bz = target
        else:
            self.Bz += (dt / self.b_tau) * (target - self.Bz)
        return self.Bz
