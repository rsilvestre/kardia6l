"""
analysis.py — Détection R-peaks, fréquence cardiaque, intervalles R-R, HRV.

S'appuie sur NeuroKit2, configuré pour la fréquence d'échantillonnage native
du 6L (300 Hz). On travaille par défaut sur Lead II, où le QRS est généralement
le plus net pour la détection des pics R.

⚠️ Rappel : sortie à but exploratoire (FC, R-R, HRV). Ce n'est pas un diagnostic
de trouble du rythme. Toute anomalie apparente doit être confirmée par un médecin
sur l'ECG d'origine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import SAMPLING_RATE_HZ


@dataclass
class EcgAnalysis:
    """Résultats d'analyse d'une dérivation."""
    rpeaks_idx: np.ndarray          # indices d'échantillon des pics R
    rr_intervals_ms: np.ndarray     # intervalles R-R successifs (ms)
    heart_rate_bpm: float           # FC moyenne (battements/min)
    hrv: dict                       # métriques HRV (time + frequency domain)
    cleaned: np.ndarray             # signal nettoyé par NeuroKit2
    sampling_rate: int = SAMPLING_RATE_HZ

    def summary(self) -> str:
        rr = self.rr_intervals_ms
        if rr.size:
            rr_txt = (f"R-R: moy {rr.mean():.0f} ms, "
                      f"min {rr.min():.0f}, max {rr.max():.0f}, "
                      f"σ {rr.std():.0f} ms")
        else:
            rr_txt = "R-R: insuffisant"
        return (f"FC moyenne : {self.heart_rate_bpm:.1f} bpm | "
                f"{self.rpeaks_idx.size} pics R | {rr_txt}")


def analyze(signal_mv: np.ndarray,
            sampling_rate: int = SAMPLING_RATE_HZ,
            correct_artifacts: bool = True) -> EcgAnalysis:
    """
    Analyse complète d'une dérivation ECG.

    Args:
        signal_mv: signal d'UNE dérivation en mV (idéalement Lead II).
        sampling_rate: 300 Hz pour le 6L.
        correct_artifacts: si True (défaut), applique la correction d'artéfacts
            R-R de NeuroKit2 (méthode Lipponen-Tarvainen/Kubios). Elle retire les
            pics aberrants — y compris un faux pic de faible amplitude éloigné de
            ses voisins, qu'un simple garde de période réfractaire ne verrait pas.
            ⚠️ Cette correction peut aussi lisser de VRAIES extrasystoles (ESV).
            Passe False pour inspecter les pics bruts (utile en cas d'arythmie
            suspectée — à confirmer par un médecin sur l'ECG d'origine).

    Returns:
        EcgAnalysis.
    """
    import neurokit2 as nk  # import paresseux : démarrage CLI plus rapide

    signal_mv = np.asarray(signal_mv, dtype=np.float64)

    # 1) Nettoyage (filtrage passe-bande adapté ECG + retrait dérive de base).
    cleaned = nk.ecg_clean(signal_mv, sampling_rate=sampling_rate)

    # 2) Détection des pics R, avec correction d'artéfacts R-R optionnelle.
    _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate,
                                  correct_artifacts=correct_artifacts)
    rpeaks_idx = np.asarray(rpeaks_info["ECG_R_Peaks"], dtype=int)

    # 3) Intervalles R-R (ms) et FC moyenne.
    if rpeaks_idx.size >= 2:
        rr_samples = np.diff(rpeaks_idx)
        rr_ms = rr_samples / sampling_rate * 1000.0
        heart_rate = 60_000.0 / rr_ms.mean()
    else:
        rr_ms = np.array([])
        heart_rate = float("nan")

    # 4) HRV (domaines temporel + fréquentiel). Nécessite assez de pics.
    hrv_metrics: dict = {}
    if rpeaks_idx.size >= 4:
        try:
            hrv_df = nk.hrv(rpeaks_idx, sampling_rate=sampling_rate, show=False)
            hrv_metrics = hrv_df.iloc[0].to_dict()
        except Exception as exc:  # robustesse : la HRV échoue sur signaux courts
            hrv_metrics = {"error": str(exc)}
    # # TODO(code): ajouter métriques non linéaires (Poincaré SD1/SD2, entropie)
    #              et une analyse par fenêtre glissante pour les enregistrements longs.

    return EcgAnalysis(
        rpeaks_idx=rpeaks_idx,
        rr_intervals_ms=rr_ms,
        heart_rate_bpm=heart_rate,
        hrv=hrv_metrics,
        cleaned=cleaned,
        sampling_rate=sampling_rate,
    )
