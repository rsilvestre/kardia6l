"""
plotting.py — Visualisation : grille des 6 dérivations + tachogramme R-R.

Style proche d'un papier ECG (fond quadrillé, calibration mm/mV) pour une
lecture familière. Sauvegarde en PNG dans output/ (pas d'affichage interactif
par défaut, pour rester utilisable en CLI / headless).
"""

from __future__ import annotations

import numpy as np

from . import SAMPLING_RATE_HZ
from .leads import LEAD_ORDER


def whole_second_formatter():
    """Formateur d'axe temps : n'étiquette qu'aux secondes entières."""
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(
        lambda x, _pos: f"{x:.0f}" if abs(x - round(x)) < 1e-6 else ""
    )


def apply_ecg_paper_grid(ax, time_labels: bool = True) -> None:
    """Applique le quadrillage calibré « papier ECG » (0,2 s × 0,5 mV ;
    petites cases 0,04 s × 0,1 mV) à un axe matplotlib."""
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    if time_labels:
        ax.xaxis.set_major_formatter(whole_second_formatter())
    ax.grid(which="major", color="#f4b6b6", linewidth=0.6)
    ax.grid(which="minor", color="#f9dada", linewidth=0.3)


def plot_six_leads(leads: dict[str, np.ndarray],
                   sampling_rate: int = SAMPLING_RATE_HZ,
                   rpeaks_idx: np.ndarray | None = None,
                   rpeaks_lead: str = "II",
                   device_beats_idx: np.ndarray | None = None,
                   out_path: str = "output/ecg_6leads.png",
                   title: str = "KardiaMobile 6L — dérivations") -> str:
    """Trace les dérivations disponibles, empilées. Renvoie le chemin du PNG.

    Ne suppose pas que les 6 dérivations sont présentes : on n'affiche que celles
    réellement contenues dans `leads`, dans l'ordre canonique LEAD_ORDER.
    """
    import matplotlib
    matplotlib.use("Agg")  # backend headless
    import matplotlib.pyplot as plt

    names = [name for name in LEAD_ORDER if name in leads]
    n = len(names)
    first = leads[names[0]]
    t = np.arange(len(first)) / sampling_rate

    fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, name in zip(axes, names):
        sig = leads[name]
        ax.plot(t, sig, linewidth=0.7, color="#b00020")
        apply_ecg_paper_grid(ax)
        ax.set_ylabel(name, rotation=0, ha="right", va="center",
                      fontweight="bold")
        ax.margins(x=0)
        if name == rpeaks_lead:
            has_markers = False
            # Nos pics R détectés : triangle plein bleu.
            if rpeaks_idx is not None and len(rpeaks_idx):
                ax.plot(rpeaks_idx / sampling_rate, sig[rpeaks_idx],
                        "v", color="#1565c0", markersize=5, label="R (détecté)")
                has_markers = True
            # Battements annotés par l'appareil : fines barres verticales en bas
            # de l'axe. Sous chaque pic R détecté → concordance ; une barre
            # isolée (sans triangle) = battement vu seulement par l'appareil.
            if device_beats_idx is not None and len(device_beats_idx):
                dev = np.asarray(device_beats_idx, dtype=int)
                dev = dev[(dev >= 0) & (dev < len(sig))]
                # Transform : x en coordonnées de données, y en fraction d'axe.
                ax.vlines(dev / sampling_rate, 0.0, 0.08,
                          transform=ax.get_xaxis_transform(),
                          color="#2e7d32", alpha=0.6, linewidth=1.0,
                          label="appareil (ann)")
                has_markers = True
            if has_markers:
                ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Temps (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_rr_tachogram(rr_ms: np.ndarray,
                      out_path: str = "output/rr_tachogram.png") -> str | None:
    """Trace le tachogramme (intervalles R-R successifs). Renvoie le chemin."""
    if rr_ms.size < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(np.arange(rr_ms.size), rr_ms, "-o", markersize=3,
            color="#1565c0", linewidth=0.8)
    ax.axhline(rr_ms.mean(), color="#b00020", linestyle="--", linewidth=0.8,
               label=f"moyenne {rr_ms.mean():.0f} ms")
    ax.set_xlabel("Index du battement")
    ax.set_ylabel("Intervalle R-R (ms)")
    ax.set_title("Tachogramme R-R")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
