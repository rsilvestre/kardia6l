"""Tests du parser .atc, sur un fichier synthétique reproduisant le format v4.

Le format est reconstruit ici à la main (signature 8 octets, blocs à checksum,
fmt <BHHBH>, blocs ecg /ecg2, info, ann) — aucune donnée de santé réelle requise.
"""

import struct

import numpy as np
import pytest

from kardia6l.atc_reader import read_atc, _parse_atc_standalone, _MAGIC


def _chunk(chunk_id: bytes, body: bytes) -> bytes:
    """id (4o) + longueur (uint32) + corps + checksum (uint32, ignoré au parsing)."""
    return chunk_id + struct.pack("<I", len(body)) + body + struct.pack("<I", 0)


def _fmt_body(fs=300, amplitude_nv=500, ecg_format=1, flags=80, reserved=0) -> bytes:
    return struct.pack("<BHHBH", ecg_format, fs, amplitude_nv, flags, reserved)


def _info_body(strings) -> bytes:
    return b"\x00".join(s.encode("ascii") for s in strings) + b"\x00\x00\x00\x00"


def _ann_body(positions) -> bytes:
    body = struct.pack("<I", 300)  # en-tête (ignoré)
    for pos in positions:
        body += struct.pack("<IH", pos, 1)
    return body


def _build_atc(path, leads_samples, *, fs=300, amplitude_nv=500,
               info_strings=None, ann_positions=None, version=4):
    tags = [b"ecg ", b"ecg2", b"ecg3", b"ecg4", b"ecg5", b"ecg6"]
    buf = _MAGIC + struct.pack("<I", version)
    if info_strings is not None:
        buf += _chunk(b"info", _info_body(info_strings))
    buf += _chunk(b"fmt ", _fmt_body(fs=fs, amplitude_nv=amplitude_nv))
    for tag, samples in zip(tags, leads_samples):
        buf += _chunk(tag, np.asarray(samples, dtype="<i2").tobytes())
    if ann_positions is not None:
        buf += _chunk(b"ann ", _ann_body(ann_positions))
    path.write_bytes(buf)
    return path


def test_parses_two_leads_and_scaling(tmp_path):
    # 2000 LSB * 500 nV/LSB / 1e6 = 1.0 mV
    lead_I = [2000, -2000, 0, 1000]
    lead_II = [1000, 1000, -1000, 0]
    f = _build_atc(tmp_path / "rec.atc", [lead_I, lead_II], amplitude_nv=500)

    rec = read_atc(str(f))
    assert rec.sampling_rate == 300
    assert set(rec.leads) == {"I", "II"}
    assert rec.metadata["amplitude_nv_per_lsb"] == 500
    np.testing.assert_allclose(rec.leads["I"], [1.0, -1.0, 0.0, 0.5])
    np.testing.assert_allclose(rec.leads["II"], [0.5, 0.5, -0.5, 0.0])
    assert rec.duration_s == pytest.approx(4 / 300)


def test_single_lead_file(tmp_path):
    f = _build_atc(tmp_path / "mono.atc", [[100, 200, 300]])
    rec = read_atc(str(f))
    assert set(rec.leads) == {"I"}


def test_six_leads_file(tmp_path):
    chans = [np.zeros(10) + i for i in range(6)]
    f = _build_atc(tmp_path / "six.atc", chans)
    rec = read_atc(str(f))
    assert list(rec.leads) == ["I", "II", "III", "aVR", "aVL", "aVF"]


def test_info_metadata_parsed(tmp_path):
    info = ["2026-06-15T22:46:57+0200",
            "3c044d01-9030-49e2-baa6-741054f6e0ef",
            "iPhone 16 Pro : iOS26.5",
            "Kardia 5.58.0",
            "kardia_6l hw19SC03.03 fw3.0.1",
            "SN=9999999999999,BAT=42"]
    f = _build_atc(tmp_path / "meta.atc", [[0, 0], [0, 0]], info_strings=info)
    rec = read_atc(str(f))
    assert rec.metadata["recorded_at"] == "2026-06-15T22:46:57+0200"
    assert rec.metadata["record_uuid"] == "3c044d01-9030-49e2-baa6-741054f6e0ef"
    assert rec.metadata["phone_os"] == "iPhone 16 Pro : iOS26.5"
    assert rec.metadata["serial_number"] == "9999999999999"
    assert rec.metadata["battery_pct"] == "42"


def test_device_beats_parsed(tmp_path):
    positions = [397, 695, 997, 1299]
    f = _build_atc(tmp_path / "ann.atc", [[0, 0], [0, 0]], ann_positions=positions)
    rec = read_atc(str(f))
    assert rec.device_beats is not None
    np.testing.assert_array_equal(rec.device_beats, positions)
    assert rec.metadata["n_device_annotations"] == 4


def test_no_device_beats_when_absent(tmp_path):
    f = _build_atc(tmp_path / "noann.atc", [[0, 0], [0, 0]])
    rec = read_atc(str(f))
    assert rec.device_beats is None
    assert rec.metadata["n_device_annotations"] == 0


def test_bad_magic_raises(tmp_path):
    f = tmp_path / "bad.atc"
    f.write_bytes(b"NOPE\x00\x00\x00\x00" + b"\x00" * 32)
    with pytest.raises(ValueError):
        _parse_atc_standalone(str(f))


def test_odd_length_ecg_body_tolerated(tmp_path):
    # Corps ECG impair : le dernier octet doit être ignoré sans erreur.
    f = tmp_path / "odd.atc"
    buf = _MAGIC + struct.pack("<I", 4)
    buf += _chunk(b"fmt ", _fmt_body())
    buf += _chunk(b"ecg ", b"\x10\x27\x05")  # 3 octets -> 1 échantillon + reste
    f.write_bytes(buf)
    rec = read_atc(str(f))
    assert rec.leads["I"].size == 1
