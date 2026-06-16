"""
batch.py — Traitement par lot : analyse automatiquement les .atc de data/.

Parcourt un dossier de données, et pour chaque enregistrement .atc produit
tracés + CSV dans output/<nom_du_fichier>/. Par défaut, ne retraite QUE les
fichiers nouveaux (sans sortie existante) — idempotent, sûr à relancer.

Exemples :
    python -m kardia6l.batch                 # nouveaux fichiers seulement
    python -m kardia6l.batch --force         # tout retraiter
    python -m kardia6l.batch --raw-peaks     # sans correction d'artéfacts R-R
"""

from __future__ import annotations

import argparse
import glob
import os
import re

from .atc_reader import read_atc, AtcRecording
from .pipeline import run

# Marqueur de « déjà traité » : présence du tracé principal dans le dossier de sortie.
_DONE_MARKER = "ecg_6leads.png"


def find_recordings(data_dir: str) -> list[str]:
    """Renvoie les chemins .atc de `data_dir`, triés."""
    return sorted(glob.glob(os.path.join(data_dir, "*.atc")))


def already_done(out_dir: str) -> bool:
    """True si le dossier de sortie contient déjà le tracé principal."""
    return os.path.exists(os.path.join(out_dir, _DONE_MARKER))


def folder_label(recording: AtcRecording, fallback: str) -> str:
    """Nom de dossier lisible basé sur la date d'enregistrement.

    « 2026-06-15T22:42:28+0200 » → « 2026-06-15_22-42-28 ». Repli sur l'id
    d'origine (nom de fichier) si l'horodatage est absent ou illisible.
    """
    recorded_at = recording.metadata.get("recorded_at", "")
    match = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})", recorded_at)
    if match:
        date, hour, minute, second = match.groups()
        return f"{date}_{hour}-{minute}-{second}"
    return fallback


def process_all(data_dir: str = "data",
                out_root: str = "output",
                force: bool = False,
                correct_artifacts: bool = True) -> list[dict]:
    """Traite chaque .atc de `data_dir`. Renvoie un récapitulatif par fichier."""
    recordings = find_recordings(data_dir)
    if not recordings:
        print(f"Aucun fichier .atc trouvé dans {data_dir}/")
        return []

    summary: list[dict] = []
    used_labels: set[str] = set()
    for atc_path in recordings:
        stem = os.path.splitext(os.path.basename(atc_path))[0]

        # Lecture d'abord (l'en-tête contient la date → nom de dossier lisible).
        try:
            recording = read_atc(atc_path)
        except Exception as exc:  # un fichier corrompu ne doit pas tout stopper
            print(f"✗  {stem} — lecture impossible : {exc}")
            summary.append({"file": stem, "status": "error", "error": str(exc)})
            continue

        # Dossier basé sur la date ; suffixe en cas de collision (même seconde).
        label = folder_label(recording, fallback=stem)
        unique = label
        suffix = 2
        while unique in used_labels:
            unique = f"{label}_{suffix}"
            suffix += 1
        used_labels.add(unique)
        out_dir = os.path.join(out_root, unique)

        if already_done(out_dir) and not force:
            print(f"⏭  {unique} — déjà traité (utilise --force pour refaire)")
            summary.append({"file": stem, "label": unique, "status": "skipped"})
            continue

        print(f"▶  {stem} → {unique}/")
        try:
            outputs = run(recording, do_plot=True, export_csv=True,
                          correct_artifacts=correct_artifacts,
                          source_name=stem, out_dir=out_dir)
            analysis = outputs["analysis"]
            summary.append({
                "file": stem,
                "label": unique,
                "status": "ok",
                "n_leads": len(outputs["leads"]),
                "heart_rate_bpm": analysis.heart_rate_bpm,
                "n_rpeaks": int(analysis.rpeaks_idx.size),
                "out_dir": out_dir,
            })
        except Exception as exc:
            print(f"✗  {stem} — échec : {exc}")
            summary.append({"file": stem, "label": unique,
                            "status": "error", "error": str(exc)})

    _print_summary(summary)
    return summary


def _print_summary(summary: list[dict]) -> None:
    ok = [item for item in summary if item["status"] == "ok"]
    skipped = sum(1 for item in summary if item["status"] == "skipped")
    errors = [item for item in summary if item["status"] == "error"]

    print("\n" + "=" * 60)
    print(f"Récapitulatif : {len(ok)} traité(s), {skipped} ignoré(s), "
          f"{len(errors)} en erreur")
    for item in ok:
        print(f"  ✓ {item['label']}: {item['heart_rate_bpm']:.1f} bpm, "
              f"{item['n_rpeaks']} pics R, {item['n_leads']} dérivation(s) "
              f"[{item['file']}]")
    for item in errors:
        print(f"  ✗ {item['file']}: {item['error']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyse par lot des enregistrements .atc du KardiaMobile 6L."
    )
    parser.add_argument("--data-dir", default="data",
                        help="dossier des .atc (défaut: data/)")
    parser.add_argument("--out-dir", default="output",
                        help="racine des sorties (défaut: output/)")
    parser.add_argument("--force", action="store_true",
                        help="retraiter même les fichiers déjà analysés")
    parser.add_argument("--raw-peaks", action="store_true",
                        help="désactiver la correction d'artéfacts R-R")
    args = parser.parse_args(argv)

    process_all(data_dir=args.data_dir, out_root=args.out_dir,
                force=args.force, correct_artifacts=not args.raw_peaks)


if __name__ == "__main__":
    main()
