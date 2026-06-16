"""
leads.py — Dérivation des 6 dérivations du KardiaMobile 6L.

Le 6L ne mesure que DEUX canaux indépendants (Lead I et Lead II). Les quatre
autres se calculent algébriquement et ne contiennent aucune information
nouvelle. On reconstruit ici les 6 dérivations « membres » standard.

Définitions (RA = pouce droit, LA = pouce gauche, LL = jambe gauche) :

  Bipolaires (Einthoven) :
      I   = LA − RA        (mesurée)
      II  = LL − RA        (mesurée)
      III = II − I         (= LL − LA, dérivée ; loi d'Einthoven II = I + III)

  Augmentées (Goldberger) :
      aVR = −(I + II) / 2
      aVL =  I − II / 2     (= (I − III) / 2)
      aVF =  II − I / 2     (= (II + III) / 2)

Invariant de contrôle : aVR + aVL + aVF = 0 (à l'arrondi numérique près).

⚠️ Avertissement axe : les contacts du 6L sont pouces + jambe, pas
poignets/chevilles. L'algèbre est identique, mais l'interprétation de l'axe
électrique peut différer d'un ECG 12-dérivations clinique standard.

# TODO(code): si tu veux gérer un montage d'électrodes alternatif (ex. variantes
#             de pose), paramétrer ici la matrice de transformation.
"""

from __future__ import annotations

import numpy as np

# Ordre canonique d'affichage des dérivations membres.
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF"]


def derive_six_leads(lead_I: np.ndarray, lead_II: np.ndarray) -> dict[str, np.ndarray]:
    """
    Reconstruit les 6 dérivations à partir des deux canaux mesurés.

    Args:
        lead_I:  Lead I en mV (np.ndarray 1D)
        lead_II: Lead II en mV (même longueur)

    Returns:
        dict {nom_dérivation: np.ndarray en mV}, dans l'ordre LEAD_ORDER.
    """
    lead_I = np.asarray(lead_I, dtype=np.float64)
    lead_II = np.asarray(lead_II, dtype=np.float64)
    if lead_I.shape != lead_II.shape:
        raise ValueError(
            f"Lead I et Lead II de tailles différentes : "
            f"{lead_I.shape} vs {lead_II.shape}"
        )

    lead_III = lead_II - lead_I
    aVR = -(lead_I + lead_II) / 2.0
    aVL = lead_I - lead_II / 2.0
    aVF = lead_II - lead_I / 2.0

    return {
        "I": lead_I,
        "II": lead_II,
        "III": lead_III,
        "aVR": aVR,
        "aVL": aVL,
        "aVF": aVF,
    }


def check_goldberger_invariant(leads: dict[str, np.ndarray], atol: float = 1e-6) -> bool:
    """Vérifie aVR + aVL + aVF ≈ 0 — garde-fou contre une erreur de dérivation."""
    residual = leads["aVR"] + leads["aVL"] + leads["aVF"]
    return bool(np.allclose(residual, 0.0, atol=atol))
