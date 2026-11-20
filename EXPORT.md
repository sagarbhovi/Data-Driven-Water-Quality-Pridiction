# Project Export Guide

Export the entire `water_quality_monitoring` folder to share or deploy the project.

## What to Include

Include these **folders and files**:

| Item | Description |
|------|-------------|
| `app.py` | Main Streamlit dashboard |
| `config.py` | Paths and quality thresholds |
| `data_generator.py` | Dataset generation and train/test split |
| `train_model.py` | Model training script |
| `predict_utils.py` | Prediction and suggestions |
| `run_all.py` | One-command setup (data + train) |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `EXPORT.md` | This file |
| `data/` | Contains `water_quality_raw.csv` (after running generator) |
| `train/` | Contains `train.csv` (after running generator) |
| `test/` | Contains `test.csv` (after running generator) |
| `models/` | Contains `water_quality_model.joblib` and `label_encoder.joblib` (after training) |

## First-Time Setup on New Machine

1. **Install Python 3.8+** and create/activate a virtual environment (optional).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Generate data and train model:**
   ```bash
   python run_all.py
   ```
   Or step by step:
   ```bash
   python data_generator.py
   python train_model.py
   ```
4. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

## Zip for Export

From the parent of `water_quality_monitoring`:

```bash
# Windows PowerShell
Compress-Archive -Path water_quality_monitoring -DestinationPath water_quality_monitoring.zip
```

Or copy the whole `water_quality_monitoring` folder as-is.
