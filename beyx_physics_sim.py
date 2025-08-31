"""Physics engine for Beyblade X simulations.
Implements a simple 2D rigid body model and exposes simulate and
monte_carlo utilities used by the Streamlit front end.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Part definitions
# ---------------------------------------------------------------------------

# Blade: mass (kg), radius (m), recoil (0-1), shape_def, aero (drag)
BLADE: Dict[str, Dict[str, float]] = {
    "Knight Shield": {"mass": 0.036, "radius": 0.060, "recoil": 0.15, "shape_def": 0.90, "aero": 0.010},
    "Cobalt Ifrit": {"mass": 0.035, "radius": 0.060, "recoil": 0.25, "shape_def": 0.70, "aero": 0.012},
    "Silver Wolf": {"mass": 0.034, "radius": 0.058, "recoil": 0.20, "shape_def": 0.75, "aero": 0.011},
    "Phoenix Wing": {"mass": 0.033, "radius": 0.057, "recoil": 0.30, "shape_def": 0.60, "aero": 0.013},
    "Dran Sword": {"mass": 0.032, "radius": 0.058, "recoil": 0.35, "shape_def": 0.50, "aero": 0.014},
    "Cowle Dragon": {"mass": 0.037, "radius": 0.060, "recoil": 0.40, "shape_def": 0.55, "aero": 0.015},
}

# Ratchet: lock multiplier, height factor
RATCHET: Dict[str, Dict[str, float]] = {
    "1-60": {"lock": 0.90, "height": 0.60},
    "3-60": {"lock": 1.00, "height": 0.60},
    "4-55": {"lock": 1.10, "height": 0.55},
    "3-80": {"lock": 1.20, "height": 0.80},
    "9-60": {"lock": 1.05, "height": 0.60},
}

# Bit: mu_roll, mu_kin, traction, burst multiplier, drift factor
BIT: Dict[str, Dict[str, float]] = {
    "Wall Ball": {"mu_roll": 0.010, "mu_kin": 0.10, "traction": 0.80, "burst": 1.20, "drift": 0.00},
    "Disc Ball": {"mu_roll": 0.012, "mu_kin": 0.11, "traction": 0.75, "burst": 1.10, "drift": 0.02},
    "Gear Ball": {"mu_roll": 0.013, "mu_kin": 0.12, "traction": 0.80, "burst": 1.15, "drift": 0.01},
    "Needle": {"mu_roll": 0.008, "mu_kin": 0.08, "traction": 0.90, "burst": 1.00, "drift": 0.00},
    "Point": {"mu_roll": 0.009, "mu_kin": 0.09, "traction": 0.88, "burst": 1.00, "drift": 0.02},
    "Dot": {"mu_roll": 0.007, "mu_kin": 0.07, "traction": 0.92, "burst": 0.95, "drift": 0.00},
    "Gear Needle": {"mu_roll": 0.008, "mu_kin": 0.08, "traction": 0.90, "burst": 1.05, "drift": 0.01},
    "Spike": {"mu_roll": 0.010, "mu_kin": 0.10, "traction": 0.85, "burst": 1.00, "drift": 0.03},
    "Bound Spike": {"mu_roll": 0.011, "mu_kin": 0.11, "traction": 0.80, "burst": 1.10, "drift": 0.03},
    "Flat": {"mu_roll": 0.020, "mu_kin": 0.20, "traction": 0.50, "burst": 0.90, "drift": 0.10},
    "Accel": {"mu_roll": 0.022, "mu_kin": 0.22, "traction": 0.55, "burst": 0.90, "drift": 0.12},
    "Gear Flat": {"mu_roll": 0.021, "mu_kin": 0.21, "traction": 0.55, "burst": 0.95, "drift": 0.11},
    "Quake": {"mu_roll": 0.030, "mu_kin": 0.25, "traction": 0.40, "burst": 0.85, "drift": 0.15},
    "Cyclone": {"mu_roll": 0.025, "mu_kin": 0.23, "traction": 0.45, "burst": 0.88, "drift": 0.13},
    "Gear Point": {"mu_roll": 0.009, "mu_kin": 0.09, "traction": 0.90, "burst": 1.05, "drift": 0.03},
}

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
ARENA_R = 1.0
POCKET_ANGLES = [math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6]
POCKET_HW = 0.20  # radians
DT = 0.004
MAX_T = 30.0
STAMINA_K = 0.05
WALL_E = 0.2
BURST_K = 0.5

# ---------------------------------------------------------------------------
# Helper data structure
# ---------------------------------------------------------------------------

@dataclass
class Bey:
    blade: Dict[str, float]
    ratchet: Dict[str, float]
    bit: Dict[str, float]
    pos: np.ndarray
    vel: np.ndarray
    omega: float
    lock_hp: float

    @property
    def mass(self) -> float:
        return self.blade["mass"]

    @property
    def radius(self) -> float:
        return self.blade["radius"]


# ---------------------------------------------------------------------------
# Utility functions to expose dictionaries
# ---------------------------------------------------------------------------

def part_dicts() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return all part dictionaries."""
    return {"blade": BLADE, "ratchet": RATCHET, "bit": BIT}


def load_parts(blade_json: str | None = None,
               ratchet_json: str | None = None,
               bit_json: str | None = None) -> None:
    """Hot reload part dictionaries from JSON strings."""
    if blade_json:
        BLADE.clear()
        BLADE.update(json.loads(blade_json))
    if ratchet_json:
        RATCHET.clear()
        RATCHET.update(json.loads(ratchet_json))
    if bit_json:
        BIT.clear()
        BIT.update(json.loads(bit_json))


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _assemble(combo: Dict[str, str]) -> Bey:
    blade = BLADE[combo["blade"]]
    ratchet = RATCHET[combo["ratchet"]]
    bit = BIT[combo["bit"]]
    omega0 = 180.0
    lock_hp = 100.0 * ratchet["lock"] * bit["burst"]
    return Bey(blade, ratchet, bit, np.zeros(2), np.zeros(2), omega0, lock_hp)


def _init_state(comboA: Dict[str, str], comboB: Dict[str, str], rng: random.Random) -> Tuple[Bey, Bey]:
    a = _assemble(comboA)
    b = _assemble(comboB)

    # start slightly apart with opposite tangential velocities
    a.pos = np.array([-0.2, 0.0])
    b.pos = np.array([0.2, 0.0])
    v0 = 0.4
    a.vel = np.array([0.0, v0])
    b.vel = np.array([0.0, -v0])

    # small random variation
    a.vel += rng.uniform(-0.05, 0.05) * np.ones(2)
    b.vel += rng.uniform(-0.05, 0.05) * np.ones(2)
    return a, b


def _wall_interaction(b: Bey) -> bool:
    """Bounce on wall. Returns True if KO."""
    r = np.linalg.norm(b.pos)
    if r <= ARENA_R:
        return False

    ang = math.atan2(b.pos[1], b.pos[0]) % (2 * math.pi)
    for p in POCKET_ANGLES:
        diff = abs((ang - p + math.pi) % (2 * math.pi) - math.pi)
        if diff < POCKET_HW:
            return True

    n = b.pos / r
    b.pos = n * ARENA_R
    v_n = np.dot(b.vel, n)
    b.vel = b.vel - (1 + WALL_E) * v_n * n
    return False


def _collide(a: Bey, b: Bey) -> float:
    """Resolve collision; returns impulse magnitude."""
    delta = b.pos - a.pos
    dist = np.linalg.norm(delta)
    overlap = a.radius + b.radius - dist
    if overlap <= 0:
        return 0.0

    n = delta / dist if dist > 1e-8 else np.array([1.0, 0.0])
    # separate
    a.pos -= n * overlap / 2
    b.pos += n * overlap / 2

    rel_v = b.vel - a.vel
    vn = np.dot(rel_v, n)
    if vn > 0:
        return 0.0

    recoil = (a.blade["recoil"] + b.blade["recoil"]) / 2
    e = 0.2 + 0.6 * recoil
    j = -(1 + e) * vn / (1 / a.mass + 1 / b.mass)
    a.vel -= j * n / a.mass
    b.vel += j * n / b.mass

    dmg_a = j * b.blade["recoil"] * BURST_K
    dmg_b = j * a.blade["recoil"] * BURST_K
    a.lock_hp -= dmg_a
    b.lock_hp -= dmg_b
    return j


def simulate(comboA: Dict[str, str], comboB: Dict[str, str], seed: int | None = None) -> Dict:
    """Simulate a single battle.

    Returns a dict with result tuple and trajectory histories.
    """
    rng = random.Random(seed)
    a, b = _init_state(comboA, comboB, rng)
    histA: List[Tuple[float, float]] = []
    histB: List[Tuple[float, float]] = []

    t = 0.0
    winner = None
    reason = None
    step = 0

    while t < MAX_T:
        for bey in (a, b):
            bey.omega = max(0.0, bey.omega - DT * (bey.blade["aero"] * bey.omega + STAMINA_K * bey.bit["mu_roll"]))
            r = np.linalg.norm(bey.pos)
            if r > 1e-6:
                rad = bey.pos / r
                drift = bey.bit["drift"] * bey.omega * bey.ratchet["height"]
                bey.vel += drift * rad * DT
            bey.pos += bey.vel * DT

        if _wall_interaction(a):
            winner, reason = "B", "KO"
            break
        if _wall_interaction(b):
            winner, reason = "A", "KO"
            break

        _collide(a, b)
        if a.lock_hp <= 0 and b.lock_hp <= 0:
            winner, reason = ("A" if rng.random() < 0.5 else "B"), "Burst"
            break
        if a.lock_hp <= 0:
            winner, reason = "B", "Burst"
            break
        if b.lock_hp <= 0:
            winner, reason = "A", "Burst"
            break

        if step % 5 == 0:
            histA.append(tuple(a.pos))
            histB.append(tuple(b.pos))

        if a.omega < 0.2 and b.omega < 0.2 and np.linalg.norm(a.vel) < 0.02 and np.linalg.norm(b.vel) < 0.02:
            winner = "A" if a.omega > b.omega else "B"
            reason = "OS"
            break

        t += DT
        step += 1

    if winner is None:
        winner = "A" if a.omega > b.omega else "B"
        reason = "OS"

    return {
        "result": (winner, reason),
        "histA": histA,
        "histB": histB,
    }


def monte_carlo(comboA: Dict[str, str], comboB: Dict[str, str], N: int = 1000, seed: int | None = None) -> Dict[str, int]:
    rng = random.Random(seed)
    counts = {
        "A_OS": 0,
        "B_OS": 0,
        "A_KO": 0,
        "B_KO": 0,
        "A_Burst": 0,
        "B_Burst": 0,
    }
    for i in range(N):
        s = seed + i if seed is not None else None
        res = simulate(comboA, comboB, s)["result"]
        win, reason = res
        key = f"{win}_{reason}"
        counts[key] += 1
    return counts


__all__ = [
    "simulate",
    "monte_carlo",
    "BLADE",
    "RATCHET",
    "BIT",
    "part_dicts",
    "load_parts",
    "ARENA_R",
    "POCKET_ANGLES",
    "POCKET_HW",
]
