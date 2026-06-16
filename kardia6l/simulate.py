"""
simulate.py — Génère un ECG synthétique (Lead I & II) pour tester le pipeline
sans fichier .atc réel.

Utilise le simulateur de NeuroKit2, puis construit deux dérivations plausibles
(Lead II d'amplitude un peu supérieure à Lead I, comme souvent in vivo). Permet
d'exécuter tout le flux — dérivation des leads, R-peaks, FC, HRV, plots — de
bout en bout immédiatement.
"""

from __future__ import annotations

import numpy as np

from . import SAMPLING_RATE_HZ
from .atc_reader import AtcRecording


def simulate_recording(duration_s: int = 30,
                       heart_rate: int = 72,
                       sampling_rate: int = SAMPLING_RATE_HZ,
                       seed: int | None = 42) -> AtcRecording:
    """
    Renvoie un AtcRecording synthétique (Lead I & II en mV).

    Args:
        duration_s: durée en secondes.
        heart_rate: FC cible (bpm).
        sampling_rate: 300 Hz par défaut (comme le 6L).
        seed: graine aléatoire pour reproductibilité.
    """
    import neurokit2 as nk

    base = nk.ecg_simulate(
        duration=duration_s,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        method="ecgsyn",
        random_state=seed,
    )
    base = np.asarray(base, dtype=np.float64)

    # Lead I et Lead II : on dérive deux « projections » de l'axe cardiaque.
    # Approximation simple : Lead II ~ axe inférieur, amplitude plus grande.
    rng = np.random.default_rng(seed)
    lead_I = 0.6 * base + 0.02 * rng.standard_normal(base.size)
    lead_II = 1.0 * base + 0.02 * rng.standard_normal(base.size)

    return AtcRecording(
        leads={"I": lead_I, "II": lead_II},
        sampling_rate=sampling_rate,
        uv_per_lsb=1.0,
        metadata={"source": "simulate", "heart_rate": heart_rate},
    )
