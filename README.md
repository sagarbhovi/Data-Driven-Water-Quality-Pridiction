# Water Quality Monitoring System

NLP-based water quality monitoring system built with **Python** and **Streamlit**. It supports manual input of water parameters, ML-based quality prediction, actionable suggestions, and a city-wise dashboard.

---

## Project Structure

```
water_quality_monitoring/
├── app.py                 # Streamlit dashboard (main entry)
├── config.py              # Paths and quality thresholds
├── data_generator.py       # Generates dataset, creates train/test split
├── train_model.py         # Trains Random Forest classifier
├── predict_utils.py       # Load model, predict quality, generate suggestions
├── run_all.py             # Run data generation + training in one go
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # Raw and processed data
│   └── water_quality_raw.csv
├── train/                 # Training dataset (80% split)
│   └── train.csv
├── test/                  # Test dataset (20% split)
│   └── test.csv
└── models/                # Saved ML model and label encoder (created by train_model.py)
    ├── water_quality_model.joblib
    └── label_encoder.joblib
```

After running `python run_all.py` or `python data_generator.py` + `python train_model.py`, the folders `data/`, `train/`, `test/`, and `models/` will be populated.

---

## Features

- **Input & Predict**: Enter water parameters (pH, turbidity, dissolved oxygen, TDS, nitrate, etc.) and get **quality prediction** (Good / Moderate / Poor / Unsafe) and **suggestions**.
- **Dashboard**: Overview metrics and quality distribution.
- **City-wise Quality**: View quality breakdown and dominant quality per city from your data.
- **Upload Data**: Upload CSV with same columns; optionally append to training set and re-train.

---

## Setup

### 1. Install dependencies

```bash
cd water_quality_monitoring
pip install -r requirements.txt
```

### 2. Generate data and train model

```bash
python data_generator.py
python train_model.py
```

Or in one step:

```bash
python run_all.py
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Data Format

CSV should have columns:

- `city`
- `ph`, `turbidity`, `dissolved_oxygen`, `temperature`
- `conductivity`, `total_dissolved_solids`, `nitrate`
- `fecal_coliform`, `bod`
- `quality` (optional; if missing, the app can predict it when appending to train)

---

## Quality Classes

| Class     | Description |
|----------|-------------|
| Good     | Within safe ranges for key parameters |
| Moderate | Slightly off; minor treatment suggested |
| Poor     | Several parameters out of range |
| Unsafe   | Not suitable without treatment |

---

## Tech Stack

- **Python 3**
- **Streamlit** – dashboard and forms
- **pandas** – data handling
- **scikit-learn** – Random Forest classifier
- **plotly** – charts
- **joblib** – model persistence

---

## Export Checklist

When exporting the project, ensure these are included:

- [x] All Python files: `app.py`, `config.py`, `data_generator.py`, `train_model.py`, `predict_utils.py`, `run_all.py`
- [x] `requirements.txt`
- [x] `README.md`
- [x] Folders: `data/`, `train/`, `test/`, `models/` (with generated CSVs and trained model after running steps above)
