"""Generate water quality dataset and split into train/test."""
import os
import numpy as np
import pandas as pd
from config import DATA_DIR, TRAIN_DIR, TEST_DIR, PARAM_RANGES, QUALITY_LABELS

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
]

def compute_quality_label(row):
    """Assign quality based on parameter thresholds (simplified WQI logic)."""
    score = 0
    # pH: ideal 6.5-8.5
    if 6.5 <= row["ph"] <= 8.5:
        score += 25
    elif 6 <= row["ph"] <= 9:
        score += 15
    else:
        score += 0
    # Dissolved oxygen: higher is better
    if row["dissolved_oxygen"] >= 8:
        score += 25
    elif row["dissolved_oxygen"] >= 5:
        score += 15
    else:
        score += 5
    # Turbidity: lower is better (< 5 NTU good)
    if row["turbidity"] <= 5:
        score += 20
    elif row["turbidity"] <= 10:
        score += 10
    else:
        score += 0
    # TDS: < 500 good
    if row["total_dissolved_solids"] <= 500:
        score += 15
    elif row["total_dissolved_solids"] <= 1000:
        score += 8
    else:
        score += 0
    # Nitrate: < 10 mg/L
    if row["nitrate"] <= 10:
        score += 15
    elif row["nitrate"] <= 20:
        score += 5
    else:
        score += 0
    if score >= 80:
        return "Good"
    elif score >= 60:
        return "Moderate"
    elif score >= 40:
        return "Poor"
    else:
        return "Unsafe"

def generate_dataset(n_samples=2000):
    """Generate synthetic water quality data for all cities."""
    np.random.seed(42)
    n_per_city = n_samples // len(CITIES)
    records = []
    for city in CITIES:
        for _ in range(n_per_city):
            # Vary parameters to get all quality classes
            r = np.random.random()
            if r < 0.3:
                ph = np.random.uniform(6.5, 8.5)
                do = np.random.uniform(7, 12)
                turb = np.random.uniform(0.5, 4)
                tds = np.random.uniform(100, 400)
                nitrate = np.random.uniform(0, 8)
            elif r < 0.6:
                ph = np.random.uniform(6, 9)
                do = np.random.uniform(5, 8)
                turb = np.random.uniform(4, 10)
                tds = np.random.uniform(400, 800)
                nitrate = np.random.uniform(8, 15)
            elif r < 0.85:
                ph = np.random.uniform(5.5, 9.5)
                do = np.random.uniform(3, 6)
                turb = np.random.uniform(8, 20)
                tds = np.random.uniform(700, 1200)
                nitrate = np.random.uniform(12, 25)
            else:
                ph = np.random.uniform(4, 10)
                do = np.random.uniform(1, 5)
                turb = np.random.uniform(15, 50)
                tds = np.random.uniform(1000, 2000)
                nitrate = np.random.uniform(20, 50)
            temp = np.random.uniform(15, 32)
            cond = tds * np.random.uniform(1.2, 1.8)
            fc = int(np.random.exponential(10))
            bod = np.random.uniform(1, 15)
            row = {
                "city": city,
                "ph": round(ph, 2),
                "turbidity": round(turb, 2),
                "dissolved_oxygen": round(do, 2),
                "temperature": round(temp, 2),
                "conductivity": round(cond, 2),
                "total_dissolved_solids": round(tds, 2),
                "nitrate": round(nitrate, 2),
                "fecal_coliform": min(fc, 500),
                "bod": round(bod, 2),
            }
            row["quality"] = compute_quality_label(row)
            records.append(row)
    df = pd.DataFrame(records)
    return df

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    df = generate_dataset(2000)
    df.to_csv(os.path.join(DATA_DIR, "water_quality_raw.csv"), index=False)
    # 80-20 split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(df))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    train_df.to_csv(os.path.join(TRAIN_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(TEST_DIR, "test.csv"), index=False)
    print(f"Generated {len(df)} records. Train: {len(train_df)}, Test: {len(test_df)}")
    print("Quality distribution:", df["quality"].value_counts().to_dict())

if __name__ == "__main__":
    main()
