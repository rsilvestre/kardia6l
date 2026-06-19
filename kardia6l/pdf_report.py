"""
pdf_report.py — Rapport ECG clinique d'une page (PDF), mise en page « cardiologue ».

Compose en une page A4 : en-tête + métadonnées, bandeau NON DIAGNOSTIQUE,
tracé calibré (25 mm/s, 10 mm/mV) des dérivations disponibles, et un tableau de
mesures (FC, rythme, axe QRS, PR, QRS, QT, QTc Bazett/Fridericia) confrontées aux
repères adultes avec code couleur. Rendu via matplotlib (aucune dépendance en plus).

⚠️ Mesures automatiques, indicatives — voir clinical.py et les avertissements.
"""

from __future__ import annotations

import numpy as np

from . import clinical
from .leads import LEAD_ORDER
from .analysis import (compute_intervals, estimate_qrs_axis, qtc_fridericia,
                       signal_quality)

# Couleur du texte de statut.
_STATUS_COLOR = {
    clinical.WITHIN: "#1b7d3a",
    clinical.BORDERLINE: "#c77700",
    clinical.OUTSIDE: "#b00020",
    clinical.UNKNOWN: "#888888",
}


def _axis_status(axis_deg: float) -> str:
    category = clinical.interpret_axis(axis_deg)
    if category == clinical.UNKNOWN:
        return clinical.UNKNOWN
    if category == "axe normal":
        return clinical.WITHIN
    if "déviation" in category:
        return clinical.BORDERLINE
    return clinical.OUTSIDE


def _ms(value) -> str:
    return f"{value:.0f} ms" if value is not None and np.isfinite(value) else "—"


def _measurement_rows(recording, result, leads) -> list[tuple[str, str, str, str]]:
    """Construit les lignes (libellé, valeur, normale, statut) du tableau."""
    fs = recording.sampling_rate
    rr = result.rr_intervals_ms
    intervals = compute_intervals(result.cleaned, result.rpeaks_idx, fs)
    qt_ms = intervals.get("QT_ms", float("nan"))
    rr_s = float(np.median(rr) / 1000.0) if rr.size else float("nan")
    qtc_b = intervals.get("QTc_Bazett_ms", float("nan"))
    qtc_f = qtc_fridericia(qt_ms, rr_s)
    axis = estimate_qrs_axis(leads, result.rpeaks_idx, fs)
    regularity = clinical.rhythm_regularity(rr)

    hr = result.heart_rate_bpm
    rhythm_status = {"régulier": clinical.WITHIN,
                     "irrégulier": clinical.BORDERLINE}.get(regularity, clinical.UNKNOWN)
    axis_txt = (f"{axis:.0f}° ({clinical.interpret_axis(axis)})"
                if np.isfinite(axis) else "—")

    return [
        ("Fréquence cardiaque",
         f"{hr:.0f} bpm" if np.isfinite(hr) else "—",
         clinical.normal_range_text("HR"), clinical.classify("HR", hr)),
        ("Rythme", regularity, "—", rhythm_status),
        ("Axe QRS", axis_txt, "-30 à +90°", _axis_status(axis)),
        ("Intervalle PR", _ms(intervals.get("PR_ms")),
         clinical.normal_range_text("PR"), clinical.classify("PR", intervals.get("PR_ms"))),
        ("Durée QRS", _ms(intervals.get("QRS_ms")),
         clinical.normal_range_text("QRS"), clinical.classify("QRS", intervals.get("QRS_ms"))),
        ("Intervalle QT", _ms(qt_ms),
         clinical.normal_range_text("QT"), clinical.classify("QT", qt_ms)),
        ("QTc (Bazett)", _ms(qtc_b),
         clinical.normal_range_text("QTc"), clinical.classify("QTc", qtc_b)),
        ("QTc (Fridericia)", _ms(qtc_f),
         clinical.normal_range_text("QTc"), clinical.classify("QTc", qtc_f)),
    ]


def build_pdf(recording, result, leads: dict, source_name: str,
              analysis_lead: str, out_path: str) -> str:
    """Génère le rapport PDF d'une page. Renvoie le chemin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotting import apply_ecg_paper_grid

    fs = recording.sampling_rate
    meta = recording.metadata
    quality = signal_quality(result.cleaned, fs)

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor("white")

    # --- En-tête + bandeau ----------------------------------------------------
    fig.text(0.5, 0.975, "Rapport ECG — KardiaMobile 6L",
             ha="center", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.949,
             "⚠ Document exploratoire — NON DIAGNOSTIQUE — à confirmer par un médecin",
             ha="center", fontsize=8.5, color="#b00020", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdecec",
                       edgecolor="#b00020"))

    quality_txt = f"{quality:.2f}" if np.isfinite(quality) else "—"
    left = "\n".join([
        f"Date         : {meta.get('recorded_at', '—')}",
        f"Appareil     : {meta.get('device_fw', '—')}",
        f"N° série     : {meta.get('serial_number', '—')}",
        f"Identifiant  : {meta.get('record_uuid', '—')}",
    ])
    right = "\n".join([
        f"Durée    : {recording.duration_s:.0f} s @ {fs} Hz",
        f"Dérivations : {len(leads)} (analyse : Lead {analysis_lead})",
        f"Qualité signal : {quality_txt}",
        f"Source   : {source_name}",
    ])
    fig.text(0.06, 0.925, left, va="top", ha="left", fontsize=8, family="monospace")
    fig.text(0.54, 0.925, right, va="top", ha="left", fontsize=8, family="monospace")

    # --- Tracé calibré (10 premières secondes) -------------------------------
    names = [n for n in LEAD_ORDER if n in leads]
    n = len(names)
    band_top, band_bottom = 0.82, 0.45
    per = (band_top - band_bottom) / n
    win_s = min(10.0, recording.duration_s)
    n_win = int(win_s * fs)
    t = np.arange(n_win) / fs

    fig.text(0.08, band_top + 0.006,
             f"Tracé — {win_s:.0f} premières s (calibré 25 mm/s · 10 mm/mV)",
             fontsize=7.5, color="#555")
    for i, name in enumerate(names):
        ax = fig.add_axes([0.08, band_top - (i + 1) * per, 0.88, per * 0.9])
        sig = leads[name][:n_win]
        ax.plot(t, sig, linewidth=0.6, color="#b00020")
        apply_ecg_paper_grid(ax, time_labels=(i == n - 1))
        ax.set_xlim(0, win_s)
        ax.set_ylabel(name, rotation=0, ha="right", va="center",
                      fontweight="bold", fontsize=9)
        ax.margins(x=0)
        if i != n - 1:
            ax.set_xticklabels([])
        if name == analysis_lead and result.rpeaks_idx.size:
            rp = result.rpeaks_idx[result.rpeaks_idx < n_win]
            if rp.size:
                ax.plot(rp / fs, sig[rp], "v", color="#1565c0", markersize=4)

    # --- Tableau de mesures ---------------------------------------------------
    rows = _measurement_rows(recording, result, leads)
    fig.text(0.06, 0.432, "Mesures automatiques vs repères adultes",
             fontsize=10, fontweight="bold")

    tax = fig.add_axes([0.06, 0.12, 0.88, 0.29])
    tax.axis("off")
    cols_x = [0.00, 0.40, 0.64, 0.84]
    headers = ["Mesure", "Valeur", "Normale", "Statut"]
    n_lines = len(rows) + 1
    row_h = 1.0 / n_lines

    def _y(line_index):
        return 1.0 - row_h * (line_index + 0.5)

    for cx, header in zip(cols_x, headers):
        tax.text(cx, _y(0), header, fontsize=9, fontweight="bold",
                 transform=tax.transAxes, va="center")
    tax.plot([0, 1], [1 - row_h, 1 - row_h], transform=tax.transAxes,
             color="#bbbbbb", linewidth=0.8)

    for r, (label, value, normal, status) in enumerate(rows, start=1):
        y = _y(r)
        tax.text(cols_x[0], y, label, fontsize=8.5, transform=tax.transAxes, va="center")
        tax.text(cols_x[1], y, value, fontsize=8.5, transform=tax.transAxes, va="center")
        tax.text(cols_x[2], y, normal, fontsize=8.5, transform=tax.transAxes,
                 va="center", color="#555")
        tax.text(cols_x[3], y, status.capitalize(), fontsize=8.5,
                 transform=tax.transAxes, va="center", fontweight="bold",
                 color=_STATUS_COLOR.get(status, "#000000"))

    # --- Avertissements / pied de page ---------------------------------------
    fig.text(0.06, 0.085,
             "Mesures automatiques sur électrodes pouces / 30 s : indicatives, "
             "souvent imprécises (PR/QRS), à ne pas interpréter sans relecture.",
             fontsize=7, color="#555", style="italic", wrap=True)
    fig.text(0.5, 0.035,
             "Signal retraité (filtrage AliveCor d'origine perdu). Exploration "
             "personnelle — PAS un dispositif de diagnostic.\n"
             "Généré par kardia6l.",
             ha="center", fontsize=7, color="#777")

    fig.savefig(out_path)  # format déduit de l'extension (.pdf, .png…)
    plt.close(fig)
    return out_path
