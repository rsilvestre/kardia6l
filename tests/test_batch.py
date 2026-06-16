"""Tests des utilitaires de traitement par lot (nommage de dossier, découverte)."""

import numpy as np

from kardia6l.atc_reader import AtcRecording
from kardia6l.batch import folder_label, find_recordings, already_done


def _rec(recorded_at=None):
    meta = {} if recorded_at is None else {"recorded_at": recorded_at}
    return AtcRecording(leads={"I": np.zeros(3)}, metadata=meta)


def test_folder_label_from_date():
    rec = _rec("2026-06-15T22:46:57+0200")
    assert folder_label(rec, fallback="x") == "2026-06-15_22-46-57"


def test_folder_label_fallback_when_no_date():
    assert folder_label(_rec(), fallback="enhanced-abc") == "enhanced-abc"


def test_folder_label_fallback_on_bad_date():
    assert folder_label(_rec("pas-une-date"), fallback="fb") == "fb"


def test_find_recordings_sorted(tmp_path):
    (tmp_path / "b.atc").write_bytes(b"")
    (tmp_path / "a.atc").write_bytes(b"")
    (tmp_path / "note.txt").write_bytes(b"")  # ignoré
    found = find_recordings(str(tmp_path))
    assert [p.rsplit("/", 1)[-1] for p in found] == ["a.atc", "b.atc"]


def test_already_done(tmp_path):
    assert not already_done(str(tmp_path))
    (tmp_path / "ecg_6leads.png").write_bytes(b"x")
    assert already_done(str(tmp_path))
