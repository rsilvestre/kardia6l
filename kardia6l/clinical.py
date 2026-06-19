"""
clinical.py — Plages de référence adultes et classification (exploratoire).

⚠️ Repères indicatifs pour adultes, NON spécifiques au sexe/âge/contexte.
Servent uniquement à colorer un rapport ; ne constituent PAS un diagnostic.
Les seuils suivent des conventions usuelles (cf. recommandations générales
de lecture ECG) mais doivent être confirmés par un médecin sur l'ECG d'origine.
"""

from __future__ import annotations

import math

# (borne basse normale, borne haute normale, unité). None = pas de borne.
NORMAL_RANGES = {
    "HR":   (60.0, 100.0, "bpm"),
    "PR":   (120.0, 200.0, "ms"),
    "QRS":  (None, 110.0, "ms"),     # > 120 ms = élargi
    "QT":   (None, 440.0, "ms"),
    "QTc":  (None, 450.0, "ms"),     # borderline 450–470, prolongé > 470
}

# Statuts possibles.
WITHIN = "normal"
BORDERLINE = "limite"
OUTSIDE = "hors plage"
UNKNOWN = "non mesuré"


def _is_num(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def classify(measure: str, value) -> str:
    """Classe une mesure vs sa plage normale → WITHIN / BORDERLINE / OUTSIDE / UNKNOWN."""
    if not _is_num(value) or measure not in NORMAL_RANGES:
        return UNKNOWN
    low, high, _unit = NORMAL_RANGES[measure]

    # QTc : zone limite explicite entre 450 et 470 ms.
    if measure == "QTc":
        if value <= 450:
            return WITHIN
        if value <= 470:
            return BORDERLINE
        return OUTSIDE

    # QRS : normal ≤ 110, limite 110–120, élargi > 120.
    if measure == "QRS":
        if value <= 110:
            return WITHIN
        if value <= 120:
            return BORDERLINE
        return OUTSIDE

    if low is not None and value < low:
        return OUTSIDE
    if high is not None and value > high:
        return OUTSIDE
    return WITHIN


def normal_range_text(measure: str) -> str:
    """Texte lisible de la plage normale (ex. « 60–100 bpm », « ≤ 450 ms »)."""
    if measure not in NORMAL_RANGES:
        return "—"
    low, high, unit = NORMAL_RANGES[measure]
    if measure == "QTc":
        return f"≤ 450 {unit}"
    if measure == "QRS":
        return f"≤ 120 {unit}"
    if low is not None and high is not None:
        return f"{low:.0f}–{high:.0f} {unit}"
    if high is not None:
        return f"≤ {high:.0f} {unit}"
    if low is not None:
        return f"≥ {low:.0f} {unit}"
    return "—"


def interpret_axis(axis_deg) -> str:
    """Interprète un axe QRS (degrés) en catégorie d'orientation."""
    if not _is_num(axis_deg):
        return UNKNOWN
    if -30.0 <= axis_deg <= 90.0:
        return "axe normal"
    if -90.0 <= axis_deg < -30.0:
        return "déviation axiale gauche"
    if 90.0 < axis_deg <= 180.0:
        return "déviation axiale droite"
    return "axe indéterminé (nord-ouest)"


def rhythm_regularity(rr_intervals_ms, cv_threshold: float = 0.10) -> str:
    """Régularité du rythme d'après le coefficient de variation des R-R."""
    import numpy as np
    rr = np.asarray(rr_intervals_ms, dtype=float)
    if rr.size < 3:
        return UNKNOWN
    cv = rr.std(ddof=1) / rr.mean() if rr.mean() else float("inf")
    return "régulier" if cv <= cv_threshold else "irrégulier"
