# kardia6l — Pipeline d'analyse du signal ECG du KardiaMobile 6L

> Outil personnel de **traitement de signal et de visualisation** pour les ECG
> enregistrés avec un AliveCor KardiaMobile 6L (modèle AC-019).
>
> ⚠️ **Ce n'est pas un dispositif de diagnostic.** Le signal reconstruit/retraité
> perd le filtrage validé d'AliveCor. L'interprétation clinique reste du ressort
> d'un cardiologue. Cet outil sert à explorer *tes propres* données : visualiser
> le tracé, calculer la fréquence cardiaque, les intervalles R-R et la HRV.

---

## Contexte technique

### L'appareil
Le KardiaMobile 6L possède **trois électrodes** : deux sur la face supérieure
(pouce gauche / pouce droit) et une sous l'appareil (jambe / genou gauche).
Il numérise **deux canaux bipolaires simultanés** :

- **Lead I**  = LA − RA  (pouce gauche − pouce droit)
- **Lead II** = LL − RA  (jambe gauche − pouce droit)

Les **quatre autres dérivations sont calculées en logiciel** (sur le téléphone,
ou ici par nous) — elles ne contiennent aucune information indépendante :

```
Lead III = II − I
aVR      = −(I + II) / 2
aVL      =  I − II/2      (= (I − III)/2)
aVF      =  II − I/2      (= (II + III)/2)
```
Vérification : `aVR + aVL + aVF == 0` (à l'arrondi près).
Triangle d'Einthoven (bipolaires) + bornes de Goldberger (augmentées).

### Caractéristiques d'échantillonnage
- **Fréquence d'échantillonnage : 300 Hz** (clé pour tous les algos en aval)
- Résolution : 14 bits (spec officielle 6L) — le `.atc` stocke des entiers 16 bits
- Plage dynamique : 10 mV crête-à-crête
- Durée d'enregistrement : 30 s à 5 min
- Transmission : **Bluetooth Low Energy uniquement** (pas d'audio FM 19 kHz —
  ça, c'est l'ancien KardiaMobile mono-dérivation)

### D'où viennent les données brutes
Le format **`.atc`** (Alive File Format) contient les **vrais échantillons
numériques** — pas une image. C'est la source idéale.

**Format réel observé (version 4), validé sur les enregistrements de `data/` :**
- Signature `ALIVE\0\0\0` (8 octets) + version `uint32` LE.
- Blocs : `id` (4 octets ASCII minuscules) + `longueur` (`uint32` LE) + corps +
  **checksum `uint32`** de fin de bloc (et non un padding de 2 octets).
- `fmt ` (layout `<BHHBH`) : format, **fs = 300 Hz**, **amplitude = 500 nV/LSB**,
  flags, réservé. Conversion mV = `échantillon × nV_par_LSB / 1e6`.
- `ecg `, `ecg2`…`ecg6` : un bloc `int16` LE **par dérivation**, dans l'ordre
  `I, II, III, aVR, aVL, aVF`. Le téléphone stocke **les 6 dérivations déjà
  calculées** ; un enregistrement mono-canal ne contient que `ecg ` (Lead I).
- `info` (horodatage + UUID) et `ann ` (annotations, non exploité).

**Récupération depuis le cloud :**
1. Enregistre un ECG dans l'app, laisse-le se synchroniser.
2. Connecte-toi sur `app.alivecor.com`, ouvre l'enregistrement.
3. Dans l'URL du rapport, remplace l'extension `.pdf` par **`.atc`**.
4. Place le fichier dans `data/`.

Parsers : `alivecor/ATCpy`, ou `pyATC` (PyPI). Le format est documenté ;
`atc_reader.py` ci-dessous en fournit une implémentation autonome minimale,
avec repli sur `pyATC` si installé.

---

## Architecture du projet

```
kardia6l/
├── README.md                  ← ce fichier (contexte + sources)
├── requirements.txt
├── data/                      ← tes fichiers .atc (non versionnés)
├── output/                    ← exports PNG / CSV générés
└── kardia6l/
    ├── __init__.py
    ├── atc_reader.py          ← parsing .atc → samples NumPy (Lead I, II)
    ├── leads.py               ← dérivation des 6 dérivations (Einthoven/Goldberger)
    ├── analysis.py            ← R-peaks, FC, R-R, HRV (NeuroKit2)
    ├── plotting.py            ← tracés 6 dérivations + tachogramme R-R
    ├── simulate.py            ← générateur de signal de test (sans .atc)
    └── pipeline.py            ← orchestration de bout en bout + CLI
```

### Flux
```
.atc ──atc_reader──► (Lead I, Lead II) @300Hz
                         │
                    leads.derive_12 ──► dict des 6 dérivations
                         │
                    analysis.analyze ──► R-peaks, FC, RR, HRV
                         │
                    plotting.* ──► PNG dans output/
```

---

## Démarrage rapide

```bash
pip install -r requirements.txt

# Sans fichier .atc — tourne sur un signal simulé pour valider le pipeline :
python -m kardia6l.pipeline --simulate --plot

# Avec un vrai enregistrement :
python -m kardia6l.pipeline --atc data/mon_ecg.atc --plot --export-csv
```

---

## Points d'extension (à continuer dans Claude Code)

Cherche les marqueurs `# TODO(code):` dans les sources. Les principaux :

1. **`atc_reader.py`** — le parser autonome lit le format v4 réel (signature
   8 octets, blocs à checksum, `fmt ` `<BHHBH`, blocs `ecg `…`ecg6` mono/6-canaux,
   amplitude en nV/LSB). À durcir : vérification des checksums, exploitation du
   bloc `ann ` (annotations), variantes de version < 4. Le repli `pyATC` reste
   disponible (`--prefer-pyatc`).
2. **`analysis.py`** — la HRV est branchée sur `nk.hrv()` (time + frequency).
   À étendre : métriques non linéaires (Poincaré, entropie), fenêtrage glissant.
3. **`leads.py`** — la reconstruction suppose pouce/pouce/jambe. Note l'avertissement
   sur l'axe (contacts pouces ≠ poignets/chevilles standard) si tu interprètes l'axe.
4. **BLE temps réel** (hors scope ici) — si un jour tu veux le streaming live,
   c'est un module séparé (`bleak`), à caractériser d'abord via HCI snoop log /
   nRF Connect. Voir le guide de recherche associé.

---

## Sources

- AliveCor — spécifications 6L (300 Hz, 14 bits, 10 mV, BLE) ; Alive File Format
  Spec 1.6 ; `alivecor/ATCpy`, `alivecor/atc2json`.
- `pyATC` (PyPI) — lecture/écriture ATC, export JSON/EDF.
- Goldberger / Einthoven — dérivation des dérivations augmentées et bipolaires.
- NeuroKit2 — `ecg_process`, `ecg_peaks`, `ecg_delineate`, `hrv` @ 300 Hz.
- J. McAteer, « I Hacked an ECG Machine » (2025) — démodulation de l'ancien
  modèle audio (référence pour le contexte historique FM 19 kHz / 200 Hz/mV).

## Note légale
L'accès à **tes propres** données de santé depuis un appareil que tu possèdes est
légitime (exemption DMCA dispositifs médicaux ; partage des données patient sur
demande). Ne redistribue pas l'app/firmware d'AliveCor ; n'intercepte pas l'appareil
d'autrui. Données reconstruites = exploration personnelle, pas diagnostic.
