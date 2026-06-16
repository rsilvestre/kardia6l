"""
pipeline.py — Orchestration de bout en bout + interface ligne de commande.

Flux :
    .atc (ou signal simulé)
        → atc_reader        : (Lead I, Lead II) @ 300 Hz
        → leads.derive      : 6 dérivations
        → analysis.analyze  : R-peaks, FC, R-R, HRV (sur Lead II)
        → plotting          : PNG (6 dérivations + tachogramme)
        → export CSV optionnel

Exemples :
    python -m kardia6l.pipeline --simulate --plot
    python -m kardia6l.pipeline --atc data/mon_ecg.atc --plot --export-csv
    python -m kardia6l.pipeline --atc data/mon_ecg.atc --prefer-pyatc
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .atc_reader import read_atc, AtcRecording
from .leads import derive_six_leads, check_goldberger_invariant, LEAD_ORDER
from .analysis import analyze
from . import plotting


def run(recording: AtcRecording,
        do_plot: bool = False,
        export_csv: bool = False,
        correct_artifacts: bool = True,
        out_dir: str = "output") -> dict:
    """Exécute le pipeline complet sur un AtcRecording. Renvoie un dict de résultats."""
    os.makedirs(out_dir, exist_ok=True)

    # 1) Obtention des dérivations. Le .atc 6L stocke directement les 6 canaux ;
    #    s'il n'en contient que deux (I & II), on reconstruit les 4 autres.
    stored = recording.leads
    if "I" in stored and "II" in stored and len(stored) < 6:
        leads = derive_six_leads(stored["I"], stored["II"])
        if not check_goldberger_invariant(leads):
            print("⚠️  Invariant de Goldberger non vérifié — dérivation suspecte.")
    else:
        leads = stored  # déjà 6 dérivations (ou un seul canal) — on les garde tels quels

    # 2) Analyse rythmique sur la meilleure dérivation disponible
    #    (Lead II de préférence : QRS le plus net).
    analysis_lead = "II" if "II" in leads else next(iter(leads))
    result = analyze(leads[analysis_lead], sampling_rate=recording.sampling_rate,
                     correct_artifacts=correct_artifacts)
    print(f"Durée : {recording.duration_s:.1f} s @ {recording.sampling_rate} Hz "
          f"| {len(leads)} dérivation(s) | analyse sur Lead {analysis_lead}")
    print(result.summary())

    outputs: dict = {"analysis": result, "leads": leads}

    # 3) Visualisation.
    if do_plot:
        p1 = plotting.plot_six_leads(
            leads, sampling_rate=recording.sampling_rate,
            rpeaks_idx=result.rpeaks_idx, rpeaks_lead=analysis_lead,
            out_path=os.path.join(out_dir, "ecg_6leads.png"),
        )
        p2 = plotting.plot_rr_tachogram(
            result.rr_intervals_ms,
            out_path=os.path.join(out_dir, "rr_tachogram.png"),
        )
        outputs["plots"] = [p for p in (p1, p2) if p]
        print("Figures :", ", ".join(outputs["plots"]))

    # 4) Export CSV (toutes les dérivations, alignées dans le temps).
    if export_csv:
        present = [name for name in LEAD_ORDER if name in leads]
        t = np.arange(len(leads[present[0]])) / recording.sampling_rate
        cols = np.column_stack([t] + [leads[name] for name in present])
        header = "time_s," + ",".join(present)
        csv_path = os.path.join(out_dir, "ecg_leads.csv")
        np.savetxt(csv_path, cols, delimiter=",", header=header,
                   comments="", fmt="%.6f")
        outputs["csv"] = csv_path
        print("CSV :", csv_path)

    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyse du signal ECG du KardiaMobile 6L "
                    "(traitement de signal — PAS un diagnostic)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--atc", metavar="FICHIER", help="chemin d'un fichier .atc")
    src.add_argument("--simulate", action="store_true",
                     help="utiliser un signal synthétique (sans .atc)")

    parser.add_argument("--prefer-pyatc", action="store_true",
                        help="parser avec pyATC en priorité (formats exotiques)")
    parser.add_argument("--plot", action="store_true",
                        help="générer les figures PNG")
    parser.add_argument("--export-csv", action="store_true",
                        help="exporter les 6 dérivations en CSV")
    parser.add_argument("--raw-peaks", action="store_true",
                        help="désactiver la correction d'artéfacts R-R "
                             "(pics bruts — utile si arythmie suspectée)")
    parser.add_argument("--out-dir", default="output",
                        help="dossier de sortie (défaut: output/)")
    parser.add_argument("--duration", type=int, default=30,
                        help="[simulate] durée en s (défaut: 30)")
    parser.add_argument("--hr", type=int, default=72,
                        help="[simulate] FC cible en bpm (défaut: 72)")
    args = parser.parse_args(argv)

    if args.simulate:
        from .simulate import simulate_recording
        recording = simulate_recording(duration_s=args.duration,
                                        heart_rate=args.hr)
    else:
        recording = read_atc(args.atc, prefer_pyatc=args.prefer_pyatc)

    run(recording, do_plot=args.plot, export_csv=args.export_csv,
        correct_artifacts=not args.raw_peaks, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
