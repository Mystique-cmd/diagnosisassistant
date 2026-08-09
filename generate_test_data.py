# ============================================================
# Generates a HELD-OUT synthetic test set for evaluation.
#
# Reuses the same disease/symptom profiles as
# modules/ml_classifier.py's _generate_synthetic_data(), but
# with a DIFFERENT random seed, so this data was never seen
# during training. Saves to data/test_data.csv, matching the
# schema expected by evaluation/metric.py:
#   - 18 symptom columns (SYMPTOM_COLUMNS)
#   - a 'disease' column (DISEASE_LABELS)
#
# Run this once from your project root:
#   python generate_test_data.py
# ============================================================

import os
import numpy as np
import pandas as pd

SYMPTOM_FEATURES = [
    'fever', 'cough', 'fatigue', 'headache',
    'body_aches', 'loss_of_smell', 'chest_pain',
    'rash', 'joint_pain', 'shortness_of_breath',
    'sweating', 'frequent_urination', 'excessive_thirst',
    'blurred_vision', 'night_sweats', 'weight_loss',
    'stiff_neck', 'light_sensitivity'
]

# Same clinically-informed profiles as ml_classifier.py's
# _generate_synthetic_data(), copied verbatim so the test
# set represents the same underlying "population."
PROFILES = {
    'flu':           {'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                       'headache': 0.70, 'body_aches': 0.80, 'loss_of_smell': 0.20},
    'covid19':       {'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90,
                       'loss_of_smell': 0.85, 'headache': 0.65, 'body_aches': 0.60},
    'dengue':        {'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85,
                       'headache': 0.90, 'fatigue': 0.80, 'body_aches': 0.88},
    'cardiac_event': {'chest_pain': 0.92, 'shortness_of_breath': 0.88,
                       'fatigue': 0.70, 'sweating': 0.75, 'headache': 0.30},
    'diabetes':      {'fatigue': 0.82, 'frequent_urination': 0.95,
                       'excessive_thirst': 0.92, 'blurred_vision': 0.70,
                       'weight_loss': 0.50},
    'common_cold':   {'cough': 0.90, 'fever': 0.50, 'headache': 0.60,
                       'fatigue': 0.55, 'body_aches': 0.50},
    'tuberculosis':  {'cough': 0.95, 'weight_loss': 0.85, 'night_sweats': 0.80,
                       'fatigue': 0.88, 'fever': 0.70},
    'meningitis':    {'headache': 0.95, 'stiff_neck': 0.90, 'fever': 0.92,
                       'light_sensitivity': 0.85, 'fatigue': 0.80},
}

# IMPORTANT: different seed from ml_classifier.py's seed=42,
# so this data is independent of what the models trained on.
TEST_SEED = 999
N_PER_CLASS = 40  # 40 patients x 8 diseases = 320 test cases


def generate_test_set(n_per_class: int = N_PER_CLASS,
                       seed: int = TEST_SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    records = []

    for disease, symptom_probs in PROFILES.items():
        for _ in range(n_per_class):
            record = {f: 0 for f in SYMPTOM_FEATURES}
            for symptom, prob in symptom_probs.items():
                if symptom in record:
                    record[symptom] = int(rng.random() < prob)
            # Same 5% background noise as training generator,
            # so the test distribution matches the training one.
            for feat in SYMPTOM_FEATURES:
                if record[feat] == 0 and rng.random() < 0.05:
                    record[feat] = 1
            record['disease'] = disease
            records.append(record)

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_test_set()

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "test_data.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} held-out test cases "
          f"({N_PER_CLASS} per disease, seed={TEST_SEED})")
    print(f"Saved to: {out_path}")
    print(f"Columns : {list(df.columns)}")
    print("\nClass distribution:")
    print(df['disease'].value_counts())
