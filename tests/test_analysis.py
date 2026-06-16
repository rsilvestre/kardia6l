"""Tests du recoupement pics R / annotations appareil (logique pure)."""

import numpy as np

from kardia6l.analysis import crosscheck_device_beats


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
