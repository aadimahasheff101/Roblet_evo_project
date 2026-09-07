"""Physical and simulation constants — the single source of truth for v2.

Every number in the pipeline traces back to this file. Units are SI unless a
name says otherwise. Each constant is tagged with its provenance:

  [HW]      hardware-confirmed (Aaditya / datasheet / Chen et al. IROS 2026)
  [CAL]     calibrated against hardware video in the v7 b-field environment
  [ASSUME]  modelling assumption, defensible but not measured
  [TODO]    placeholder awaiting a lab measurement — flagged, one-line swap
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- module ----
MODULE_RADIUS = 6.25e-3        # m   [HW] 12.5 mm diameter disc
MODULE_HEIGHT = 5.0e-3         # m   [HW] 5 mm thick
MODULE_MASS = 0.44e-3          # kg  [HW] 0.44 g, uniform for all GA modules
HALF_MASS = MODULE_MASS / 2.0  # kg  each half-disc

# Centre-to-centre spacing of bonded neighbours in the flat sheet.
# Tangent discs: 2 R. Collision geoms are shrunk slightly (GEOM_SHRINK) so
# lattice neighbours that are NOT bonded do not register permanent contact.
CENTER_SPACING = 2.0 * MODULE_RADIUS   # m  [ASSUME] pads interlock at rim
GEOM_SHRINK = 0.98                     # [ASSUME] collision-geom radial shrink

# ------------------------------------------------------------------ fold ----
# Relative (dihedral) rotation between the two half-discs after folding.
# +45 deg = valley (outer edges up), -45 deg = mountain. Rigid once folded.
FOLD_DIHEDRAL_RAD = math.radians(45.0)   # [HW] 45 deg fold, latched
FOLD_TOLERANCE_RAD = math.radians(5.0)   # [ASSUME] fold-success gate width
FOLD_RAMP_TIME = 2.0    # s  [ASSUME] simultaneous thermal ramp duration
FOLD_SETTLE_TIME = 0.5  # s  [ASSUME] hold after ramp before gating/baking
FOLD_KP = 2.0e-2        # N m/rad [ASSUME] position-servo stiffness per crease.
                        # Sized so gravity sag of a full subtree stays < 1 deg
                        # (tau_grav ~ 2.6e-4 N m at max lever) while genuine
                        # geometric jams — contact forces — still stall the
                        # ramp and trip the fold gate.
FOLD_JOINT_DAMPING = 1.0e-4  # N m s/rad [ASSUME] crease damping during fold

# --------------------------------------------------------------- magnets ----
MAGNET_M0 = 6.5e-3       # A m^2 [HW] F412 N42 2x2 mm disc; Br~1.30 T checked
MAGNETS_PER_MODULE = 3   # [HW] one per smart-glue site, bonded or not
MAGNET_SITE_RADIUS = 5.25e-3  # m [TODO] radial offset of sites from centre
                              # (rim minus pad inset; v7's 7.5 mm exceeds the
                              # 6.25 mm module radius and cannot be right)

# ----------------------------------------------------------------- field ----
B_MAG = 10e-3            # T   [CAL] bipolar square wave amplitude
B_TAU = 8e-3             # s   [CAL] coil L/R + driver lag (first-order)
FREQ_RANGE_HZ = (3.0, 15.0)   # [CAL] operating envelope (envelope.py, v7)
DUTY_RANGE = (0.10, 0.90)     # [CAL] swept range in v7 timing experiment

# --------------------------------------------------------------- contact ----
TIMESTEP = 2.0e-4        # s   [CAL] gram-scale body needs a small step
FRICTION_TORSIONAL = 0.005    # [CAL] near-lossless contact reproduces HW
FRICTION_ROLLING = 0.0001     # [CAL]
CONTACT_SOLREF = (0.02, 1.0)  # [CAL]
SURFACES = {"acrylic": 0.6,   # [CAL] sliding mu, video-calibrated
            "paper": 0.55}    # [TODO] paper not hardware-calibrated yet

# --------------------------------------------------------------- episode ----
TILT0_DEG = 0.5          # [CAL] random initial tilt (symmetry breaking)
SETTLE_T = 0.5           # s   [CAL] field-off settling before actuation
LOCOMOTION_T = 10.0      # s   [CAL] evaluation window
TOPPLE_Z = 0.0           # body-z projection below this = toppled

# ----------------------------------------------------------------- water ----
WATER_DENSITY = 1000.0   # kg/m^3
WATER_VISCOSITY = 1.0e-3  # Pa s
# Effective body density relative to water. "Kind of floats" => just under 1.
BUOYANCY_RATIO = 0.98    # [TODO] calibrate from apparent weight in water

# ------------------------------------------------------------ population ----
MIN_MODULES = 4          # [ASSUME] smallest interesting assembly
MAX_MODULES = 20         # [ASSUME] buildability ceiling for hardware follow-up

# -------------------------------------------------------------- geometry ----
# Actual hex port directions (axial q, r), pointy-top convention. Logical
# ports 0..2 map to actual sides by tree-depth parity (even: 0/2/4, odd: 1/3/5)
# exactly as in the v1 blueprint embedding.
DIRS_AXIAL = {
    0: (+1, 0),   # E
    1: (+1, -1),  # NE
    2: (0, -1),   # NW
    3: (-1, 0),   # W
    4: (-1, +1),  # SW
    5: (0, +1),   # SE
}
EVEN_DEPTH_PORTS = (0, 2, 4)
ODD_DEPTH_PORTS = (1, 3, 5)


def port_azimuth(actual_port: int) -> float:
    """Azimuth (rad) of actual hex side k in the flat sheet: -60 deg * k."""
    return -math.radians(60.0) * (actual_port % 6)


def axial_to_xy(q: int, r: int, spacing: float = CENTER_SPACING):
    """Axial hex cell -> cartesian centre position (consistent with DIRS)."""
    x = spacing * (q + 0.5 * r)
    y = spacing * (math.sqrt(3.0) / 2.0) * r
    return x, y
