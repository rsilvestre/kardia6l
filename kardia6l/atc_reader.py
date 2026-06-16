"""
atc_reader.py — Lecture d'un fichier .atc (Alive File Format) → échantillons NumPy.

Format réel observé sur les enregistrements KardiaMobile 6L (version 4) :

  - Entête : signature "ALIVE\\x00\\x00\\x00" (8 octets) + version (uint32 LE).
  - Suite de blocs ("chunks"), chacun :
        identifiant  : 4 octets ASCII minuscules (ex. b"info", b"fmt ", b"ecg ")
        longueur     : uint32 LE (taille du corps, hors entête de bloc)
        corps        : <longueur> octets
        checksum     : uint32 LE (4 octets de fin de bloc — PAS un simple padding)

Blocs rencontrés :
  - b"info" : horodatage ISO 8601 + identifiant d'enregistrement (texte).
  - b"fmt " : métadonnées, layout "<BHHBH" =
        ecg_format (uint8), fs (uint16), amplitude_resolution (uint16, en nV/LSB),
        flags (uint8), reserved (uint16).
  - b"ecg " puis b"ecg2"…b"ecg6" : un bloc par dérivation, int16 little-endian.
        Ordre canonique : I, II, III, aVR, aVL, aVF. Un enregistrement mono-canal
        ne contient que b"ecg " (Lead I).
  - b"ann " : annotations (non exploitées ici).

⚠️ L'amplitude est exprimée en **nanovolts par LSB** (typiquement 500 nV ≈ 0,5 µV).
Conversion en mV : sample_LSB * amplitude_resolution_nV / 1e6.

# TODO(code): exploiter le bloc b"ann " (annotations), vérifier les checksums,
#             gérer d'éventuelles variantes de version < 4.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from . import SAMPLING_RATE_HZ

# Ordre canonique des dérivations tel que stocké dans les blocs ecg /ecg2…ecg6.
_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF"]
# Tags des blocs ECG, dans l'ordre des dérivations ci-dessus.
_ECG_TAGS = [b"ecg ", b"ecg2", b"ecg3", b"ecg4", b"ecg5", b"ecg6"]


# --- Structure de retour -----------------------------------------------------
@dataclass
class AtcRecording:
    """Résultat du parsing d'un .atc."""
    leads: dict[str, np.ndarray]          # {"I": array_mV, "II": array_mV, ...}
    sampling_rate: int = SAMPLING_RATE_HZ
    uv_per_lsb: float = 0.5               # gain : microvolts par LSB
    metadata: dict = field(default_factory=dict)
    # Positions (indices d'échantillon) des battements annotés par l'appareil
    # — issues du bloc ann . None si absent.
    device_beats: np.ndarray | None = None

    @property
    def duration_s(self) -> float:
        any_lead = next(iter(self.leads.values()))
        return len(any_lead) / self.sampling_rate


# --- Parser maison -----------------------------------------------------------
_MAGIC = b"ALIVE\x00\x00\x00"        # signature 8 octets
_HEADER_LEN = len(_MAGIC) + 4        # signature + version uint32
_CHUNK_TRAILER = 4                   # checksum uint32 en fin de chaque bloc


def _iter_chunks(buf: bytes, offset: int):
    """Génère (chunk_id, body_bytes) à partir de `offset` jusqu'à la fin."""
    n = len(buf)
    while offset + 8 <= n:
        chunk_id = buf[offset:offset + 4]
        (length,) = struct.unpack_from("<I", buf, offset + 4)
        body_start = offset + 8
        body_end = body_start + length
        if body_end > n:
            break  # bloc tronqué — on s'arrête proprement
        yield chunk_id, buf[body_start:body_end]
        # Chaque bloc est suivi d'un checksum uint32 (4 octets).
        offset = body_end + _CHUNK_TRAILER


def _samples_from_ecg_body(body: bytes) -> np.ndarray:
    """int16 little-endian → float64. Ignore un dernier octet impair éventuel."""
    if len(body) & 1:
        body = body[:-1]
    return np.frombuffer(body, dtype="<i2").astype(np.float64)


def _parse_info(body: bytes) -> dict:
    """Décode le bloc info : 6 chaînes terminées par \\x00, ordre constant.

    [0] horodatage ISO 8601   [1] UUID d'enregistrement   [2] téléphone/OS
    [3] version de l'app      [4] matériel/firmware        [5] "SN=…,BAT=…"
    """
    strings = [p.decode("ascii", "replace")
               for p in body.split(b"\x00") if p]
    labels = ["recorded_at", "record_uuid", "phone_os",
              "app_version", "device_fw", "serial_battery"]
    info = dict(zip(labels, strings))
    info["info_strings"] = strings  # liste brute, au cas où l'ordre changerait

    # "SN=XXXXXXXXXXXXX,BAT=96" → serial + niveau de batterie.
    sn_bat = info.get("serial_battery", "")
    for token in sn_bat.split(","):
        if token.startswith("SN="):
            info["serial_number"] = token[3:]
        elif token.startswith("BAT="):
            info["battery_pct"] = token[4:]
    return info


def _parse_atc_standalone(path: str) -> AtcRecording:
    with open(path, "rb") as fh:
        buf = fh.read()

    if not buf.startswith(_MAGIC):
        raise ValueError(
            f"{path}: signature ALIVE absente — fichier .atc invalide "
            f"ou format non reconnu (essaie prefer_pyatc=True)."
        )

    (version,) = struct.unpack_from("<I", buf, len(_MAGIC))

    sampling_rate = SAMPLING_RATE_HZ
    uv_per_lsb = 0.5                       # défaut plausible (500 nV/LSB)
    amplitude_nv = 500.0
    info: dict = {}
    device_beats: np.ndarray | None = None
    ecg_bodies: dict[bytes, np.ndarray] = {}

    for chunk_id, body in _iter_chunks(buf, _HEADER_LEN):
        if chunk_id == b"fmt " and len(body) >= 8:
            # layout "<BHHBH" : format, fs, amplitude (nV/LSB), flags, reserved.
            _fmt, fs, amplitude_nv_field, _flags, _res = struct.unpack_from(
                "<BHHBH", body, 0
            )
            if fs:
                sampling_rate = int(fs)
            if amplitude_nv_field:
                amplitude_nv = float(amplitude_nv_field)
                uv_per_lsb = amplitude_nv / 1000.0  # nV → µV
        elif chunk_id in _ECG_TAGS:
            ecg_bodies[chunk_id] = _samples_from_ecg_body(body)
        elif chunk_id == b"info":
            info = _parse_info(body)
        elif chunk_id == b"ann ":
            # En-tête 4 octets, puis enregistrements de 6 octets :
            # position (uint32, indice d'échantillon) + type (uint16, 1 = battement).
            n = max(0, (len(body) - 4) // 6)
            if n:
                recs = np.frombuffer(body[4:4 + n * 6],
                                     dtype=np.dtype([("pos", "<u4"),
                                                     ("type", "<u2")]))
                device_beats = recs["pos"].astype(int)

    if not ecg_bodies:
        raise ValueError(
            f"{path}: aucun bloc ECG trouvé. Format inattendu — "
            f"bascule sur pyATC recommandée."
        )

    # Conversion LSB → millivolts : sample * (nV/LSB) / 1e6.
    scale_mv = amplitude_nv / 1_000_000.0
    leads = {
        _LEAD_ORDER[i]: ecg_bodies[tag] * scale_mv
        for i, tag in enumerate(_ECG_TAGS)
        if tag in ecg_bodies
    }

    metadata = {
        "atc_version": version,
        "n_channels": len(leads),
        "amplitude_nv_per_lsb": amplitude_nv,
        "n_device_annotations": 0 if device_beats is None else int(device_beats.size),
        **info,
    }
    return AtcRecording(
        leads=leads,
        sampling_rate=sampling_rate,
        uv_per_lsb=uv_per_lsb,
        metadata=metadata,
        device_beats=device_beats,
    )


# --- Repli pyATC -------------------------------------------------------------
def _parse_atc_pyatc(path: str) -> AtcRecording:
    try:
        from pyATC import PyATC  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyATC n'est pas installé. `pip install pyATC` ou utilise "
            "le parser maison (prefer_pyatc=False)."
        ) from exc

    atc = PyATC.read_file(path)
    # L'API expose les canaux ; on récupère les dérivations disponibles.
    # # TODO(code): vérifier les noms d'attributs exacts selon la version de pyATC
    #              (mainData / leadData / channels). Adapter ci-dessous.
    fs = getattr(atc, "sampling_frequency", SAMPLING_RATE_HZ)
    amplitude_nv = float(getattr(atc, "amplitude_resolution", 500.0))
    scale_mv = amplitude_nv / 1_000_000.0

    channels = getattr(atc, "channels", None) or getattr(atc, "data", None)
    if channels is None:
        raise ValueError("pyATC: impossible de localiser les canaux ECG.")

    leads = {}
    for lead_name, idx in zip(_LEAD_ORDER, range(len(channels))):
        leads[lead_name] = np.asarray(channels[idx], dtype=np.float64) * scale_mv

    return AtcRecording(
        leads=leads, sampling_rate=int(fs), uv_per_lsb=amplitude_nv / 1000.0,
        metadata={"parser": "pyATC", "amplitude_nv_per_lsb": amplitude_nv},
    )


# --- API publique ------------------------------------------------------------
def read_atc(path: str, prefer_pyatc: bool = False) -> AtcRecording:
    """
    Lit un fichier .atc et renvoie un AtcRecording (dérivations en mV).

    Args:
        path: chemin du .atc.
        prefer_pyatc: si True, utilise pyATC en premier (plus robuste sur les
                      formats exotiques) ; sinon le parser maison, avec repli
                      automatique sur pyATC en cas d'échec.
    """
    if prefer_pyatc:
        return _parse_atc_pyatc(path)
    try:
        return _parse_atc_standalone(path)
    except (ValueError, struct.error) as exc:
        # Repli silencieux sur pyATC s'il est dispo, sinon on relance l'erreur.
        try:
            return _parse_atc_pyatc(path)
        except ImportError:
            raise exc
