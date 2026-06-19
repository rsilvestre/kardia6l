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

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from . import SAMPLING_RATE_HZ


@contextmanager
def _quiet_neurokit():
    """Tait les avertissements attendus de NeuroKit2 sur les signaux courts.

    Sur 30 s, l'entropie multi-échelle et la DFA long terme manquent de
    données → RuntimeWarning (division) et NeuroKitWarning bénins. On les
    masque uniquement autour des appels NeuroKit, sans toucher au reste.
    """
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield


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
            with _quiet_neurokit():
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


def compute_intervals(cleaned: np.ndarray,
                      rpeaks_idx: np.ndarray,
                      sampling_rate: int = SAMPLING_RATE_HZ) -> dict:
    """
    Délimitation P-QRS-T → intervalles médians (ms) : durée QRS, PR, QT, QTc.

    Best-effort : la délimitation peut échouer sur un signal court/bruité.
    QTc selon Bazett (QT / √RR). Toute valeur reste exploratoire, NON diagnostique.
    """
    if rpeaks_idx.size < 3:
        return {}
    import neurokit2 as nk

    def _median_ms(a_idx, b_idx) -> float:
        """médiane de (a − b) en ms, en ignorant les NaN appariés."""
        a = np.asarray(a_idx, dtype=float)
        b = np.asarray(b_idx, dtype=float)
        n = min(a.size, b.size)
        diff = a[:n] - b[:n]
        diff = diff[np.isfinite(diff) & (diff > 0)]
        if diff.size == 0:
            return float("nan")
        return float(np.median(diff)) / sampling_rate * 1000.0

    try:
        with _quiet_neurokit():
            _, waves = nk.ecg_delineate(cleaned, rpeaks_idx,
                                        sampling_rate=sampling_rate, method="dwt")
    except Exception as exc:
        return {"error": str(exc)}

    qrs_ms = _median_ms(waves.get("ECG_R_Offsets", []),
                        waves.get("ECG_R_Onsets", []))
    pr_ms = _median_ms(waves.get("ECG_R_Onsets", []),
                       waves.get("ECG_P_Onsets", []))
    qt_ms = _median_ms(waves.get("ECG_T_Offsets", []),
                       waves.get("ECG_R_Onsets", []))

    intervals = {"QRS_ms": qrs_ms, "PR_ms": pr_ms, "QT_ms": qt_ms}

    # QTc de Bazett : QT(s) / √(RR(s)) → exprimé en ms.
    if np.isfinite(qt_ms) and rpeaks_idx.size >= 2:
        rr_s = float(np.median(np.diff(rpeaks_idx))) / sampling_rate
        if rr_s > 0:
            intervals["QTc_Bazett_ms"] = qt_ms / np.sqrt(rr_s)
    return intervals


def crosscheck_device_beats(detected_idx: np.ndarray,
                            device_idx: np.ndarray,
                            sampling_rate: int = SAMPLING_RATE_HZ,
                            tolerance_ms: float = 60.0) -> dict:
    """
    Recoupe nos pics R détectés avec les battements annotés par l'appareil.

    Apparie chaque battement appareil au pic détecté le plus proche dans une
    fenêtre de ±`tolerance_ms`. Renvoie le nombre d'appariements, les battements
    vus seulement par l'appareil (qu'on a manqués), ceux vus seulement par nous
    (détections en trop), et l'écart temporel médian/max des paires.

    Args:
        detected_idx: indices d'échantillon de NOS pics R.
        device_idx: indices d'échantillon des battements de l'appareil.
        tolerance_ms: fenêtre d'appariement (défaut 60 ms).
    """
    detected = np.sort(np.asarray(detected_idx, dtype=int))
    device = np.sort(np.asarray(device_idx, dtype=int))
    tol = tolerance_ms / 1000.0 * sampling_rate  # ms → échantillons

    matched_offsets: list[float] = []
    used = np.zeros(detected.size, dtype=bool)
    device_only = 0
    for pos in device:
        if detected.size == 0:
            device_only += 1
            continue
        nearest = int(np.argmin(np.abs(detected - pos)))
        if not used[nearest] and abs(detected[nearest] - pos) <= tol:
            used[nearest] = True
            matched_offsets.append((detected[nearest] - pos) / sampling_rate * 1000.0)
        else:
            device_only += 1

    offsets = np.asarray(matched_offsets)
    return {
        "n_detected": int(detected.size),
        "n_device": int(device.size),
        "matched": int(offsets.size),
        "device_only": int(device_only),          # battements manqués par nous
        "detected_only": int((~used).sum()),       # nos détections en trop
        "tolerance_ms": tolerance_ms,
        "median_offset_ms": float(np.median(offsets)) if offsets.size else float("nan"),
        "max_abs_offset_ms": float(np.max(np.abs(offsets))) if offsets.size else float("nan"),
    }


def qtc_fridericia(qt_ms: float, rr_s: float) -> float:
    """QTc selon Fridericia : QT / RR^(1/3) (QT en ms, RR en s). NaN si invalide."""
    if not (np.isfinite(qt_ms) and qt_ms > 0 and np.isfinite(rr_s) and rr_s > 0):
        return float("nan")
    return qt_ms / (rr_s ** (1.0 / 3.0))


def estimate_qrs_axis(leads: dict[str, np.ndarray],
                      rpeaks_idx: np.ndarray,
                      sampling_rate: int = SAMPLING_RATE_HZ,
                      window_ms: float = 50.0) -> float:
    """
    Estime l'axe électrique QRS (degrés) à partir des dérivations I et aVF.

    Convention hexaxiale : I à 0°, aVF à +90°. On prend la déflexion nette du
    QRS (intégrale signée autour de chaque pic R, baseline retirée) comme proxy
    d'amplitude, puis axe = atan2(net_aVF, net_I).

    NaN si I/aVF absents (ex. enregistrement mono-canal) ou pas de pics R.
    Best-effort : sensible à la qualité du signal et à la pose pouces du 6L.
    """
    rpeaks_idx = np.asarray(rpeaks_idx, dtype=int)
    if rpeaks_idx.size == 0 or "I" not in leads or "aVF" not in leads:
        return float("nan")
    half = max(1, int(window_ms / 1000.0 * sampling_rate))

    def _net_deflection(sig: np.ndarray) -> float:
        sig = np.asarray(sig, dtype=float)
        baseline = float(np.median(sig))
        areas = [float(np.sum(sig[max(0, r - half):min(len(sig), r + half + 1)] - baseline))
                 for r in rpeaks_idx]
        return float(np.mean(areas)) if areas else float("nan")

    net_I = _net_deflection(leads["I"])
    net_aVF = _net_deflection(leads["aVF"])
    if not (np.isfinite(net_I) and np.isfinite(net_aVF)):
        return float("nan")
    return float(np.degrees(np.arctan2(net_aVF, net_I)))


def signal_quality(cleaned: np.ndarray,
                   sampling_rate: int = SAMPLING_RATE_HZ) -> float:
    """Indice de qualité moyen du signal (0–1) selon NeuroKit2. NaN si indispo."""
    import neurokit2 as nk
    try:
        with _quiet_neurokit():
            quality = nk.ecg_quality(cleaned, sampling_rate=sampling_rate)
        return float(np.nanmean(np.asarray(quality, dtype=float)))
    except Exception:
        return float("nan")
