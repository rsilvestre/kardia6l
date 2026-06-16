"""
kardia6l — Pipeline d'analyse du signal ECG du KardiaMobile 6L.

Outil personnel de traitement de signal / visualisation. PAS un dispositif
de diagnostic : l'interprétation clinique reste du ressort d'un cardiologue.
"""

__version__ = "0.1.0"

# --- Constantes de l'appareil (KardiaMobile 6L, AC-019) ---------------------
# Fréquence d'échantillonnage native du 6L. Tous les algorithmes en aval
# (détection R-peaks, FC, HRV) en dépendent — ne pas modifier sans raison.
SAMPLING_RATE_HZ = 300

# Résolution effective du convertisseur (spec officielle 6L). Le conteneur
# .atc stocke des entiers signés 16 bits ; le gain (µV/LSB) est lu dans l'entête.
ADC_RESOLUTION_BITS = 14

# Plage dynamique crête-à-crête de l'entrée différentielle.
DYNAMIC_RANGE_MV = 10.0
