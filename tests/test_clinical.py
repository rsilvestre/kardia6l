"""Tests des plages de référence et de la classification clinique."""

import numpy as np

from kardia6l import clinical


def test_classify_hr():
    assert clinical.classify("HR", 72) == clinical.WITHIN
    assert clinical.classify("HR", 45) == clinical.OUTSIDE
    assert clinical.classify("HR", 130) == clinical.OUTSIDE


def test_classify_qtc_zones():
    assert clinical.classify("QTc", 430) == clinical.WITHIN
    assert clinical.classify("QTc", 460) == clinical.BORDERLINE
    assert clinical.classify("QTc", 490) == clinical.OUTSIDE


def test_classify_qrs_zones():
    assert clinical.classify("QRS", 100) == clinical.WITHIN
    assert clinical.classify("QRS", 115) == clinical.BORDERLINE
    assert clinical.classify("QRS", 140) == clinical.OUTSIDE


def test_classify_unknown():
    assert clinical.classify("HR", float("nan")) == clinical.UNKNOWN
    assert clinical.classify("HR", None) == clinical.UNKNOWN
    assert clinical.classify("inexistant", 100) == clinical.UNKNOWN


def test_normal_range_text():
    assert clinical.normal_range_text("HR") == "60–100 bpm"
    assert clinical.normal_range_text("QTc") == "≤ 450 ms"
    assert clinical.normal_range_text("QRS") == "≤ 120 ms"


def test_interpret_axis():
    assert clinical.interpret_axis(60) == "axe normal"
    assert clinical.interpret_axis(-60) == "déviation axiale gauche"
    assert clinical.interpret_axis(120) == "déviation axiale droite"
    assert clinical.interpret_axis(float("nan")) == clinical.UNKNOWN


def test_rhythm_regularity():
    assert clinical.rhythm_regularity([800, 805, 798, 802]) == "régulier"
    assert clinical.rhythm_regularity([500, 1000, 600, 1100]) == "irrégulier"
    assert clinical.rhythm_regularity([800]) == clinical.UNKNOWN
