"""Tests des utilitaires de formatage du rapport (purs, sans NeuroKit)."""

import numpy as np

from kardia6l.report import _fmt, _table


def test_fmt_nan_is_dash():
    assert _fmt(float("nan")) == "—"


def test_fmt_float_rounded():
    assert _fmt(1.23456) == "1.23"


def test_fmt_int():
    assert _fmt(42) == "42"
    assert _fmt(np.int64(7)) == "7"


def test_fmt_string_passthrough():
    assert _fmt("abc") == "abc"


def test_table_markdown():
    md = _table([("FC", "62"), ("Durée", "30 s")])
    lines = md.splitlines()
    assert lines[0] == "| Champ | Valeur |"
    assert lines[1] == "|---|---|"
    assert "| FC | 62 |" in lines
    assert "| Durée | 30 s |" in lines
