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

from .atc_reader import read_atc
from .pipeline import run

# Marqueur de « déjà traité » : présence du tracé principal dans le dossier de sortie.
_DONE_MARKER = "ecg_6leads.png"


def find_recordings(data_dir: str) -> list[str]:
    """Renvoie les chemins .atc de `data_dir`, triés."""
    return sorted(glob.glob(os.path.join(data_dir, "*.atc")))


def already_done(out_dir: str) -> bool:
    """True si le dossier de sortie contient déjà le tracé principal."""
    return os.path.exists(os.path.join(out_dir, _DONE_MARKER))


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
    for atc_path in recordings:
        stem = os.path.splitext(os.path.basename(atc_path))[0]
        out_dir = os.path.join(out_root, stem)

        if already_done(out_dir) and not force:
            print(f"⏭  {stem} — déjà traité (utilise --force pour refaire)")
            summary.append({"file": stem, "status": "skipped"})
            continue

        print(f"▶  {stem}")
        try:
            recording = read_atc(atc_path)
            outputs = run(recording, do_plot=True, export_csv=True,
                          correct_artifacts=correct_artifacts, out_dir=out_dir)
            analysis = outputs["analysis"]
            summary.append({
                "file": stem,
                "status": "ok",
                "n_leads": len(outputs["leads"]),
                "heart_rate_bpm": analysis.heart_rate_bpm,
                "n_rpeaks": int(analysis.rpeaks_idx.size),
                "out_dir": out_dir,
            })
        except Exception as exc:  # un fichier corrompu ne doit pas tout stopper
            print(f"✗  {stem} — échec : {exc}")
            summary.append({"file": stem, "status": "error", "error": str(exc)})

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
        print(f"  ✓ {item['file']}: {item['heart_rate_bpm']:.1f} bpm, "
              f"{item['n_rpeaks']} pics R, {item['n_leads']} dérivation(s)")
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
