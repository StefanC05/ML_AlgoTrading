# Quick Start Guide

ML Algorithmic Trading Framework - Complete Machine Learning Pipeline for Stock Return Prediction

## Requirements

- Python 3.8+
- Git (for cloning repository)
- ~16GB RAM recommended (8GB minimum)
- ~50GB free storage
- TA-Lib (technical analysis library)

## Installation (5 minutes)

### 1. Clone Repository

```bash
git clone https://github.com/StefanC05/ML_AlgoTrading.git
cd ML_AlgoTrading
```

### 2. Dependencies installieren

```bash
pip install -r Requirements.txt
```

Wenn TA-Lib Probleme macht (häufig auf Windows):
```bash
# Windows:
# Lade .whl-Datei herunter: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.25-cp38-cp38-win_amd64.whl

# Linux/Mac:
conda install -c conda-forge ta-lib
```

### 3. Daten vorbereiten

**Option A: Vorbereitete Features verwenden (Empfohlen)**
```bash
# Die vorverarbeiteten Features sind bereits im Repo enthalten
# Dateien: data/DATA_03_features_targets.h5
# Direkt zu Schritt 5 gehen
```

**Option B: Von Rohdaten starten (fortgeschrittene Nutzer)**
```bash
# 1. Laden Sie historische Daten von Stooq.com herunter:
#    - Besuchen Sie: https://stooq.com/
#    - Downloaden Sie NYSE Daten als CSV
#    - Speichern Sie in: data/nyse stocks/

# 2. Führen Sie die komplette Pipeline aus:
python scripts/data_load.py          # Laden und Zusammenfassung
python scripts/preprocess.py         # Datenbereinigung (20 Minuten)
python scripts/feature_engineering.py # Feature-Erstellung (30 Minuten)
python scripts/train_models.py       # Training (3-6 Stunden)
```

## Training starten

### Phase 1: Modellentwicklung (2006-2015)

```bash
python scripts/train_models.py --phase1
```

**Erwartete Ausgaben:**
- `output_files/training_fold_results_phase1.csv` - Detaillierte Metriken
- `output_files/training_stats_summary_phase1.csv` - Aggregierte Statistiken
- `output_files/models_final_phase1.joblib` - Trainierte Modelle
- Trainingsdauer: ~2-4 Stunden

### Phase 2: Out-of-Sample Testing (2016-2025)

```bash
python scripts/train_models.py --phase2
```

**Erwartete Ausgaben:**
- `output_files/training_fold_results_phase2.csv` - Metriken für Phase 2
- `output_files/training_stats_summary_phase2.csv` - Aggregierte Statistiken
- `output_files/models_final_phase2.joblib` - Finale Modelle
- Trainingsdauer: ~3-5 Stunden

### Beide Phasen gleichzeitig

```bash
python scripts/train_models.py
```

## Ergebnisse anschauen

Nach dem Training:

```bash
# CSV-Ergebnisse öffnen
cat output_files/training_stats_summary_phase1.csv

# Oder in Excel/Editor öffnen:
# Windows:
start output_files/training_stats_summary_phase1.csv

# Linux:
xdg-open output_files/training_stats_summary_phase1.csv

# Mac:
open output_files/training_stats_summary_phase1.csv
```

## Feature-Analyse

Nach `feature_engineering.py`:

```bash
# Mutual Information Analyse betrachten
# Datei: output_files/mi_analysis.png

# Zeigt die Top-20 Features und deren Wichtigkeit
```

## Verzeichnisstruktur

```
AlgoTrading-Framework/
│
├── scripts/                    # Ausführbare Skripte
│   ├── data_load.py           # Daten laden
│   ├── preprocess.py          # Datenbereinigung
│   ├── feature_engineering.py # Features erstellen
│   └── train_models.py        # Modelle trainieren
│
├── src/                       # Python-Bibliotheken
│   ├── feature_lib.py         # Feature-Funktionen
│   ├── model_utils.py         # ML-Utilities
│   ├── random_forest_model.py # Random Forest
│   └── visualizations.py      # Plots
│
├── output_files/              # Ergebnisse (nach Training)
│   ├── models_final_phase1.joblib
│   ├── training_fold_results_phase1.csv
│   ├── training_stats_summary_phase1.csv
│   └── mi_analysis.png
│
├── data/                      # Daten (nicht im Repo bei großen Dateien)
│   └── DATA_03_features_targets.h5 # Vorverarbeitete Features
│
├── Requirements.txt           # Python-Dependencies
├── README.md                  # Projektbeschreibung
├── QUICKSTART.md             # Dieses Dokument
└── .gitignore                # Git-Einstellungen
```

## Typische Trainingsdauer

| Phase | Features | Ticks | Dauer |
|-------|----------|-------|-------|
| data_load.py | - | N/A | 5-10 Min |
| preprocess.py | - | ~2000 | 15-30 Min |
| feature_engineering.py | 50+ | ~2000 | 20-40 Min |
| train_models.py Phase 1 | 50+ | ~2000 | 2-4 Std |
| train_models.py Phase 2 | 50+ | ~2000 | 3-5 Std |

**Gesamtzeit: 6-13 Stunden** (läuft größtenteils automatisch)

## Fehlerbehebung

### ModuleNotFoundError: No module named 'talib'

```bash
# Option 1: Via conda (einfacher)
conda install -c conda-forge ta-lib

# Option 2: Via pip (Windows)
# Lade wheel herunter: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.25-cp38-cp38-win_amd64.whl
```

### MemoryError während training_models.py

```bash
# Reduziere Datengröße in scripts/preprocess.py:
# Ändere PERIOD_END = "2025-12-31" zu z.B. "2020-12-31"

# Oder: Nutze Maschine mit mehr RAM (16GB+ empfohlen)
```

### HDF5 Fehler: No module named 'tables'

```bash
pip install tables
```

### Daten nicht gefunden: FileNotFoundError

```bash
# Stelle sicher, dass die Dateien existieren:
ls data/           # Linux/Mac
dir data           # Windows

# Falls leer: Folge "Option B: Von Rohdaten starten" oben
```

## Next Steps & Workflow

### 1. Explore Pre-computed Results
```bash
# The repository already contains trained models and results
python analyze_results.py     # Generate comprehensive analysis
python timing_analysis.py     # View training performance breakdown
```

### 2. Understand the Framework
- **README.md**: Complete technical overview and results analysis
- **scripts/train_models.py**: Main training script with configuration options
- **src/model_utils.py**: Walk-forward validation implementation

### 3. Customize and Experiment
- **Modify hyperparameters** in `src/model_utils.py`
- **Add new features** in `src/feature_lib.py`
- **Re-run analysis** with updated parameters
- **Compare architectures** using the comparison tools in `scripts/`

### 4. Advanced Usage
```bash
# Quick results analysis
python analyze_results.py

# Detailed timing analysis
python timing_analysis.py

# Compare model architectures
python scripts/compare_models.py
```

## Getting the Original Data

**Important:** Due to repository size limits, this project includes pre-computed results but not raw data.

- **Data sources:** See `data/README_data.txt` for detailed instructions
- **Recommended:** Download NYSE data from Stooq.com (~5GB)
- **Time to replicate:** 30 minutes setup + 50-80 hours training

## Project Architecture Overview

```
ML_AlgoTrading/
├── data/                          # Pre-processed results (included)
│   ├── models_final_phase*.joblib       # Trained models
│   ├── training_stats_summary_phase*.csv # Performance metrics
│   ├── training_fold_results_phase*.csv   # Detailed fold data
│   └── README_data.txt                   # Data acquisition guide
├── scripts/                       # Core pipeline scripts
│   ├── data_load.py              # Data ingestion and validation
│   ├── preprocess.py             # Data cleaning and filtering
│   ├── feature_engineering.py    # Technical indicator creation
│   └── train_models.py           # Main training orchestrator
├── src/                          # Framework code
│   ├── feature_lib.py            # 50+ feature functions
│   ├── model_utils.py            # Validation and training utils
│   ├── random_forest_model.py    # RF implementation
│   ├── lstm_model.py            # LSTM network (PyTorch/Darts)
│   ├── tcn_model.py             # TCN network
│   ├── tft_model.py             # TFT transformer
│   └── visualizations.py         # Analysis plots
├── output_files/                 # Generated results
├── notebooks/                    # Demo and examples
├── docs/                        # Technical documentation
├── README.md                     # Complete project overview
├── QUICKSTART.md                 # This guide
└── Requirements.txt              # Python dependencies
```

## What to Expect

**✅ Included in Repository:**
- Complete source code (scripts/, src/)
- Pre-trained models and results
- Configuration files and documentation
- Analysis tools and visualization code

**⚠️ Requires Additional Download:**
- Raw historical NYSE data (~5GB from Stooq.com)
- Processing time: 80+ hours on standard hardware

## Performance Summary (Realistic Results)

Based on rigorous 12-fold walk-forward validation (2011-2026):

| Component | Status | Performance | Training Time |
|-----------|--------|-------------|---------------|
| **RandomForest** | 🏆 Production Ready | Sharpe +1.0 to +2.0 | 2.5 hours |
| **LSTM/TCN** | ⚠️ Requires Verification | Data quality issues detected | 4-6 hours |
| **TFT** | ❌ Convergence Issues | Underperformance | 37+ hours |
| **Feature Engineering** | ✅ Validated | 50+ technical indicators | 40 minutes |

## Support & Contact

For questions about this ML Algorithmic Trading framework:

- **GitHub Issues:** Please create issues for bugs or feature requests
- **Documentation:** Check README.md for detailed technical explanation
- **Data Setup:** Refer to `data/README_data.txt` for data acquisition help

## License

MIT License - See LICENSE file (if included in repository)

---

**Ready to explore the framework? Start with `python analyze_results.py` to see the complete results analysis!**
