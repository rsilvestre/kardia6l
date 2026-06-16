"""Test d'intégration de bout en bout sur un signal simulé (utilise NeuroKit2).

Plus lent que les tests unitaires (import NeuroKit + détection R + HRV), mais
valide le pipeline complet : dérivations → analyse → tracés → CSV → rapport.
"""

import os

import pytest

from kardia6l.simulate import simulate_recording
from kardia6l.pipeline import run


@pytest.fixture(scope="module")
def simulated():
    return simulate_recording(duration_s=12, heart_rate=72, seed=42)


def test_simulate_recording_shape(simulated):
    assert set(simulated.leads) >= {"I", "II"}
    assert simulated.duration_s == pytest.approx(12, abs=0.5)


def test_pipeline_end_to_end(tmp_path, simulated):
    out = run(simulated, do_plot=True, export_csv=True,
              source_name="test-sim", out_dir=str(tmp_path))

    # Analyse plausible
    analysis = out["analysis"]
    assert analysis.rpeaks_idx.size > 5
    assert 50 < analysis.heart_rate_bpm < 100

    # Six dérivations dérivées de I & II
    assert set(out["leads"]) == {"I", "II", "III", "aVR", "aVL", "aVF"}

    # Fichiers produits
    for fname in ("ecg_6leads.png", "rr_tachogram.png", "ecg_leads.csv", "report.md"):
        assert os.path.exists(tmp_path / fname), f"manquant : {fname}"

    # Le rapport contient les sections clés et l'id source
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "test-sim" in report
    assert "## 1. Source" in report
    assert "## 6. Variabilité" in report
