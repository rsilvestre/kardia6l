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


def plot_six_leads(leads: dict[str, np.ndarray],
                   sampling_rate: int = SAMPLING_RATE_HZ,
                   rpeaks_idx: np.ndarray | None = None,
                   rpeaks_lead: str = "II",
                   out_path: str = "output/ecg_6leads.png",
                   title: str = "KardiaMobile 6L — dérivations") -> str:
    """Trace les dérivations disponibles, empilées. Renvoie le chemin du PNG.

    Ne suppose pas que les 6 dérivations sont présentes : on n'affiche que celles
    réellement contenues dans `leads`, dans l'ordre canonique LEAD_ORDER.
    """
    import matplotlib
    matplotlib.use("Agg")  # backend headless
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FuncFormatter

    # Le quadrillage est calibré à 0,2 s, mais on n'étiquette qu'aux secondes
    # entières (sinon 150 labels illisibles sur un tracé de 30 s).
    whole_second = FuncFormatter(
        lambda x, _pos: f"{x:.0f}" if abs(x - round(x)) < 1e-6 else ""
    )

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
        # Quadrillage calibré « papier ECG » : 0,2 s × 0,5 mV (grosses cases),
        # 0,04 s × 0,1 mV (petites cases).
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.xaxis.set_major_formatter(whole_second)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(which="major", color="#f4b6b6", linewidth=0.6)
        ax.grid(which="minor", color="#f9dada", linewidth=0.3)
        ax.set_ylabel(name, rotation=0, ha="right", va="center",
                      fontweight="bold")
        ax.margins(x=0)
        if rpeaks_idx is not None and name == rpeaks_lead:
            ax.plot(rpeaks_idx / sampling_rate, sig[rpeaks_idx],
                    "v", color="#1565c0", markersize=5, label="R")
            ax.legend(loc="upper right", fontsize=8)

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
