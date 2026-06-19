"""Tests du recoupement pics R / annotations appareil (logique pure)."""

import numpy as np
import pytest

from kardia6l.analysis import (crosscheck_device_beats, qtc_fridericia,
                               estimate_qrs_axis)


def test_perfect_agreement():
    beats = np.array([300, 600, 900, 1200])
    check = crosscheck_device_beats(beats, beats, sampling_rate=300)
    assert check["matched"] == 4
    assert check["device_only"] == 0
    assert check["detected_only"] == 0
    assert check["median_offset_ms"] == 0.0


def test_constant_offset_within_tolerance():
    device = np.array([300, 600, 900])
    detected = device - 2  # 2 échantillons = 6.67 ms à 300 Hz
    check = crosscheck_device_beats(detected, device, sampling_rate=300)
    assert check["matched"] == 3
    assert check["median_offset_ms"] < 0  # détecté avant l'appareil
    assert abs(check["median_offset_ms"]) < 10


def test_extra_detection_we_have():
    device = np.array([600, 900, 1200])
    detected = np.array([100, 600, 900, 1200])  # 100 = pic en trop
    check = crosscheck_device_beats(detected, device, sampling_rate=300)
    assert check["matched"] == 3
    assert check["detected_only"] == 1
    assert check["device_only"] == 0


def test_beat_we_missed():
    device = np.array([300, 600, 900, 1200])
    detected = np.array([300, 900, 1200])  # 600 manquant
    check = crosscheck_device_beats(detected, device, sampling_rate=300)
    assert check["matched"] == 3
    assert check["device_only"] == 1
    assert check["detected_only"] == 0


def test_empty_detection():
    check = crosscheck_device_beats(np.array([]), np.array([300, 600]),
                                    sampling_rate=300)
    assert check["matched"] == 0
    assert check["device_only"] == 2
    assert np.isnan(check["median_offset_ms"])


def test_qtc_fridericia():
    # QT=400 ms, RR=1 s → 400 ; RR=0.512 s → 400 / 0.8 = 500
    assert qtc_fridericia(400, 1.0) == pytest.approx(400.0)
    assert qtc_fridericia(400, 0.512) == pytest.approx(500.0, rel=1e-3)


def test_qtc_fridericia_invalid():
    assert np.isnan(qtc_fridericia(float("nan"), 1.0))
    assert np.isnan(qtc_fridericia(400, 0))


def test_estimate_qrs_axis_45deg():
    # Déflexion nette égale et positive en I et aVF → axe ≈ 45°.
    n = 300
    base = np.zeros(n)
    spike = base.copy()
    rpeaks = np.array([100, 200])
    for r in rpeaks:
        spike[r - 2:r + 3] = 1.0
    leads = {"I": spike.copy(), "aVF": spike.copy()}
    axis = estimate_qrs_axis(leads, rpeaks, sampling_rate=300)
    assert axis == pytest.approx(45.0, abs=1.0)


def test_estimate_qrs_axis_missing_lead():
    assert np.isnan(estimate_qrs_axis({"I": np.zeros(10)}, np.array([5]), 300))
    assert np.isnan(estimate_qrs_axis({"I": np.zeros(10), "aVF": np.zeros(10)},
                                      np.array([]), 300))
