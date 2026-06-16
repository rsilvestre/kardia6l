"""Tests de la dérivation des 6 dérivations (Einthoven / Goldberger)."""

import numpy as np
import pytest

from kardia6l.leads import derive_six_leads, check_goldberger_invariant, LEAD_ORDER


def test_derive_six_leads_relations():
    rng = np.random.default_rng(0)
    lead_I = rng.standard_normal(500)
    lead_II = rng.standard_normal(500)
    leads = derive_six_leads(lead_I, lead_II)

    assert list(leads.keys()) == LEAD_ORDER
    np.testing.assert_allclose(leads["I"], lead_I)
    np.testing.assert_allclose(leads["II"], lead_II)
    # Einthoven : III = II - I
    np.testing.assert_allclose(leads["III"], lead_II - lead_I)
    # Goldberger
    np.testing.assert_allclose(leads["aVR"], -(lead_I + lead_II) / 2)
    np.testing.assert_allclose(leads["aVL"], lead_I - lead_II / 2)
    np.testing.assert_allclose(leads["aVF"], lead_II - lead_I / 2)


def test_goldberger_invariant_holds():
    lead_I = np.linspace(-1, 1, 300)
    lead_II = np.sin(np.linspace(0, 10, 300))
    leads = derive_six_leads(lead_I, lead_II)
    # aVR + aVL + aVF == 0 par construction
    assert check_goldberger_invariant(leads)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        derive_six_leads(np.zeros(100), np.zeros(101))
