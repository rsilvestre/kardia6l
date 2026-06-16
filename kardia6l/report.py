"""
report.py — Génère un rapport Markdown complet par enregistrement.

Rassemble TOUT ce qu'on peut tirer du signal et du conteneur .atc :
métadonnées (date, appareil, firmware, série, batterie…), propriétés du signal,
rythme (FC, R-R), HRV (temporel/fréquentiel/non linéaire), intervalles
P-QRS-T (durée QRS, PR, QT, QTc) et qualité du signal.

⚠️ Sortie exploratoire — PAS un diagnostic. Toute valeur doit être confirmée
par un médecin sur l'ECG d'origine.
"""

from __future__ import annotations

import numpy as np

from .leads import LEAD_ORDER
from .analysis import EcgAnalysis, compute_intervals, signal_quality

# Sous-ensembles HRV mis en avant (le reste va dans l'annexe).
_HRV_TIME = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD",
             "HRV_pNN50", "HRV_pNN20", "HRV_CVNN", "HRV_MinNN", "HRV_MaxNN"]
_HRV_FREQ = ["HRV_VLF", "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn",
             "HRV_HFn", "HRV_TP"]
_HRV_NONLINEAR = ["HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_S",
                  "HRV_SampEn", "HRV_ApEn", "HRV_DFA_alpha1"]


def _fmt(value) -> str:
    """Formate une valeur pour le rapport (NaN → « — »)."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "—" if not np.isfinite(value) else f"{value:.2f}"
    return str(value)


def _table(rows: list[tuple[str, str]]) -> str:
    """Rend un petit tableau Markdown à deux colonnes."""
    out = ["| Champ | Valeur |", "|---|---|"]
    out += [f"| {label} | {value} |" for label, value in rows]
    return "\n".join(out)


def _hrv_rows(hrv: dict, keys: list[str]) -> list[tuple[str, str]]:
    rows = []
    for key in keys:
        if key in hrv and np.isfinite(_as_float(hrv[key])):
            rows.append((key.replace("HRV_", ""), _fmt(hrv[key])))
    return rows


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_report(recording, result: EcgAnalysis, leads: dict,
                 source_name: str, analysis_lead: str,
                 correct_artifacts: bool = True) -> str:
    """Construit le rapport Markdown (chaîne) pour un enregistrement."""
    meta = recording.metadata
    rr = result.rr_intervals_ms
    fs = recording.sampling_rate

    lines: list[str] = []
    add = lines.append

    recorded_at = meta.get("recorded_at", "date inconnue")
    add(f"# Rapport ECG — {recorded_at}")
    add("")
    add("> ⚠️ **Exploratoire, NON diagnostique.** Signal retraité (filtrage "
        "AliveCor d'origine perdu). À confirmer par un cardiologue.")
    add("")

    # --- 1. Source & métadonnées ------------------------------------------
    add("## 1. Source & métadonnées")
    add(_table([
        ("Fichier source (id d'origine)", f"`{source_name}.atc`"),
        ("Date / heure d'enregistrement", recorded_at),
        ("UUID d'enregistrement", meta.get("record_uuid", "—")),
        ("Appareil", meta.get("device_fw", "—")),
        ("Numéro de série", meta.get("serial_number", "—")),
        ("Batterie (%)", meta.get("battery_pct", "—")),
        ("Téléphone / OS", meta.get("phone_os", "—")),
        ("Application", meta.get("app_version", "—")),
        ("Version du format .atc", _fmt(meta.get("atc_version", "—"))),
        ("Annotations appareil (battements)",
         _fmt(meta.get("n_device_annotations", 0))),
    ]))
    add("")

    # --- 2. Propriétés du signal ------------------------------------------
    add("## 2. Propriétés du signal")
    add(_table([
        ("Durée", f"{recording.duration_s:.1f} s"),
        ("Fréquence d'échantillonnage", f"{fs} Hz"),
        ("Résolution d'amplitude",
         f"{_fmt(meta.get('amplitude_nv_per_lsb', '—'))} nV/LSB"),
        ("Nombre de dérivations stockées", _fmt(len(leads))),
        ("Dérivation analysée", f"Lead {analysis_lead}"),
        ("Qualité moyenne du signal (0–1)",
         _fmt(signal_quality(result.cleaned, fs))),
    ]))
    add("")
    add("Amplitude par dérivation (mV) :")
    add("")
    amp_rows = ["| Dérivation | min | max | crête-crête | R moyen |",
                "|---|---|---|---|---|"]
    for name in LEAD_ORDER:
        if name not in leads:
            continue
        sig = leads[name]
        r_amp = (float(np.mean(sig[result.rpeaks_idx]))
                 if result.rpeaks_idx.size else float("nan"))
        amp_rows.append(
            f"| {name} | {sig.min():.2f} | {sig.max():.2f} | "
            f"{sig.max() - sig.min():.2f} | {_fmt(r_amp)} |"
        )
    add("\n".join(amp_rows))
    add("")

    # --- 3. Rythme --------------------------------------------------------
    add("## 3. Rythme")
    if rr.size:
        inst_hr = 60_000.0 / rr
        rhythm = [
            ("FC moyenne (bpm)", f"{result.heart_rate_bpm:.1f}"),
            ("FC min / max (bpm)", f"{inst_hr.min():.0f} / {inst_hr.max():.0f}"),
            ("Battements détectés (pics R)", _fmt(int(result.rpeaks_idx.size))),
            ("R-R moyen (ms)", f"{rr.mean():.0f}"),
            ("R-R min / max (ms)", f"{rr.min():.0f} / {rr.max():.0f}"),
            ("R-R écart-type (ms)", f"{rr.std(ddof=1):.0f}" if rr.size > 1 else "—"),
            ("Correction d'artéfacts R-R",
             "activée" if correct_artifacts else "désactivée (--raw-peaks)"),
        ]
    else:
        rhythm = [("Battements détectés", _fmt(int(result.rpeaks_idx.size))),
                  ("R-R", "insuffisant pour l'analyse")]
    add(_table(rhythm))
    add("")

    # --- 4. Intervalles P-QRS-T -------------------------------------------
    add("## 4. Intervalles P-QRS-T (délimitation, best-effort)")
    intervals = compute_intervals(result.cleaned, result.rpeaks_idx, fs)
    if intervals and "error" not in intervals:
        add(_table([
            ("Durée QRS (ms)", _fmt(intervals.get("QRS_ms"))),
            ("Intervalle PR (ms)", _fmt(intervals.get("PR_ms"))),
            ("Intervalle QT (ms)", _fmt(intervals.get("QT_ms"))),
            ("QTc Bazett (ms)", _fmt(intervals.get("QTc_Bazett_ms"))),
        ]))
        add("")
        add("*⚠️ Délimitation automatique sur un signal court à électrodes "
            "pouces : les bornes d'ondes sont souvent mal placées (QRS/PR "
            "fréquemment surestimés). Valeurs purement indicatives, à ne pas "
            "interpréter cliniquement.*")
    else:
        reason = intervals.get("error", "signal insuffisant")
        add(f"Délimitation indisponible ({reason}).")
    add("")

    # --- 5. HRV -----------------------------------------------------------
    add("## 5. Variabilité de la fréquence cardiaque (HRV)")
    hrv = result.hrv or {}
    if not hrv or "error" in hrv:
        reason = hrv.get("error", "pics insuffisants") if hrv else "indisponible"
        add(f"HRV indisponible ({reason}).")
    else:
        for title, keys in (("Domaine temporel", _HRV_TIME),
                            ("Domaine fréquentiel", _HRV_FREQ),
                            ("Non linéaire", _HRV_NONLINEAR)):
            rows = _hrv_rows(hrv, keys)
            if rows:
                add(f"### {title}")
                add(_table(rows))
                add("")
        # Annexe : toutes les autres métriques HRV finies.
        shown = set(_HRV_TIME + _HRV_FREQ + _HRV_NONLINEAR)
        extra = [(k.replace("HRV_", ""), _fmt(v)) for k, v in hrv.items()
                 if k not in shown and np.isfinite(_as_float(v))]
        if extra:
            add("<details><summary>Autres métriques HRV</summary>")
            add("")
            add(_table(extra))
            add("")
            add("</details>")
    add("")

    add("---")
    add("*Généré par kardia6l — traitement de signal, non diagnostique.*")
    return "\n".join(lines)


def write_report(recording, result: EcgAnalysis, leads: dict,
                 source_name: str, analysis_lead: str, out_path: str,
                 correct_artifacts: bool = True) -> str:
    """Écrit le rapport Markdown dans `out_path`. Renvoie le chemin."""
    text = build_report(recording, result, leads, source_name,
                        analysis_lead, correct_artifacts)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path
