import numpy as np
import pandas as pd
from src.config import (
    RANDOM_SEED, NUM_ROWS, NUM_COLS, NUM_CLEAN_ROWS,
    CLEAN_DATA_MEAN, CLEAN_DATA_STD,
    NOISE_UNIFORM_RANGE, NOISE_NORMAL_STD, NOISE_MULTIPLICATIVE_RANGE,
    NOISE_EXPONENTIAL_SCALE, NOISE_LAPLACE_SCALE,
    MODERATE_OUTLIER_FRACTION, MODERATE_OUTLIER_MIN_SIGMA, MODERATE_OUTLIER_MAX_SIGMA,
    OUTLIER_NEGATION_FREQUENCY,
    NUM_DUPLICATE_ROWS, NUM_SIGN_FLIP_ROWS, SIGN_FLIP_MIN_COLS, SIGN_FLIP_MAX_COLS,
    NUM_DECIMAL_SHIFT_ROWS, DECIMAL_SHIFT_MIN_COLS, DECIMAL_SHIFT_MAX_COLS, DECIMAL_SHIFT_MULTIPLIER,
    NUM_MISSING_VALUE_ROWS, MISSING_VALUE_MIN_COLS, MISSING_VALUE_MAX_COLS,
    NUM_ZERO_CORRUPTION_ROWS, ZERO_CORRUPTION_MIN_COLS, ZERO_CORRUPTION_MAX_COLS,
    IQR_MULTIPLIER, ZSCORE_THRESHOLD, MODIFIED_ZSCORE_CONSTANT, MODIFIED_ZSCORE_THRESHOLD,
    PERCENTILE_LOWER, PERCENTILE_UPPER, TOLERANCE_MULTIPLIER
)

np.random.seed(RANDOM_SEED)

clean_data = np.random.normal(loc=CLEAN_DATA_MEAN, scale=CLEAN_DATA_STD, size=(NUM_CLEAN_ROWS, NUM_COLS))

CORRECT_MEAN = clean_data.mean()

data_with_noise = clean_data.copy()

for col_idx in range(NUM_COLS):
    if col_idx == 0:
        noise = np.random.uniform(-NOISE_UNIFORM_RANGE, NOISE_UNIFORM_RANGE, size=NUM_CLEAN_ROWS)
    elif col_idx == 1:
        noise = np.random.normal(0, NOISE_NORMAL_STD, size=NUM_CLEAN_ROWS)
    elif col_idx == 2:
        noise = data_with_noise[:, col_idx] * np.random.uniform(-NOISE_MULTIPLICATIVE_RANGE, NOISE_MULTIPLICATIVE_RANGE, size=NUM_CLEAN_ROWS)
    elif col_idx == 3:
        noise = np.random.exponential(NOISE_EXPONENTIAL_SCALE, size=NUM_CLEAN_ROWS) - NOISE_EXPONENTIAL_SCALE
    elif col_idx == 4:
        noise = np.random.laplace(0, NOISE_LAPLACE_SCALE, size=NUM_CLEAN_ROWS)
    
    data_with_noise[:, col_idx] += noise

num_moderate_outliers = int(NUM_ROWS * MODERATE_OUTLIER_FRACTION)
outlier_min = CLEAN_DATA_MEAN + MODERATE_OUTLIER_MIN_SIGMA * CLEAN_DATA_STD
outlier_max = CLEAN_DATA_MEAN + MODERATE_OUTLIER_MAX_SIGMA * CLEAN_DATA_STD
moderate_outliers = np.random.uniform(outlier_min, outlier_max, size=(num_moderate_outliers, NUM_COLS))
for i in range(num_moderate_outliers):
    if i % OUTLIER_NEGATION_FREQUENCY == 0:
        moderate_outliers[i] *= -1

duplicate_indices = np.random.choice(NUM_CLEAN_ROWS, size=NUM_DUPLICATE_ROWS, replace=True)
duplicate_rows = data_with_noise[duplicate_indices]

sign_flip_rows = data_with_noise[np.random.choice(NUM_CLEAN_ROWS, size=NUM_SIGN_FLIP_ROWS, replace=False)].copy()
for i in range(NUM_SIGN_FLIP_ROWS):
    flip_cols = np.random.choice(NUM_COLS, size=np.random.randint(SIGN_FLIP_MIN_COLS, SIGN_FLIP_MAX_COLS), replace=False)
    sign_flip_rows[i, flip_cols] *= -1

decimal_shift_rows = data_with_noise[np.random.choice(NUM_CLEAN_ROWS, size=NUM_DECIMAL_SHIFT_ROWS, replace=False)].copy()
for i in range(NUM_DECIMAL_SHIFT_ROWS):
    shift_cols = np.random.choice(NUM_COLS, size=np.random.randint(DECIMAL_SHIFT_MIN_COLS, DECIMAL_SHIFT_MAX_COLS), replace=False)
    for col in shift_cols:
        if np.random.rand() > 0.5:
            decimal_shift_rows[i, col] *= DECIMAL_SHIFT_MULTIPLIER
        else:
            decimal_shift_rows[i, col] /= DECIMAL_SHIFT_MULTIPLIER

missing_value_rows = data_with_noise[np.random.choice(NUM_CLEAN_ROWS, size=NUM_MISSING_VALUE_ROWS, replace=False)].copy()
for i in range(NUM_MISSING_VALUE_ROWS):
    nan_cols = np.random.choice(NUM_COLS, size=np.random.randint(MISSING_VALUE_MIN_COLS, MISSING_VALUE_MAX_COLS), replace=False)
    missing_value_rows[i, nan_cols] = np.nan

zero_corruption_rows = data_with_noise[np.random.choice(NUM_CLEAN_ROWS, size=NUM_ZERO_CORRUPTION_ROWS, replace=False)].copy()
for i in range(NUM_ZERO_CORRUPTION_ROWS):
    zero_cols = np.random.choice(NUM_COLS, size=np.random.randint(ZERO_CORRUPTION_MIN_COLS, ZERO_CORRUPTION_MAX_COLS), replace=False)
    zero_corruption_rows[i, zero_cols] = 0

all_data = np.vstack([
    data_with_noise,
    moderate_outliers,
    duplicate_rows,
    sign_flip_rows,
    decimal_shift_rows,
    missing_value_rows,
    zero_corruption_rows,
])

np.random.shuffle(all_data)

df = pd.DataFrame(all_data, columns=[f'feature_{i}' for i in range(NUM_COLS)])

def calculate_mean_iqr_method():
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - IQR_MULTIPLIER * IQR
    upper_bound = Q3 + IQR_MULTIPLIER * IQR
    
    mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
    df_cleaned = df[mask]
    
    return df_cleaned.values.mean()

def calculate_mean_zscore_method():
    z_scores = np.abs((df - df.mean()) / df.std())
    mask = (z_scores < ZSCORE_THRESHOLD).all(axis=1)
    df_cleaned = df[mask]
    
    return df_cleaned.values.mean()

def calculate_mean_modified_zscore_method():
    median = df.median()
    mad = np.abs(df - median).median()
    modified_z_scores = MODIFIED_ZSCORE_CONSTANT * (df - median) / mad
    mask = (np.abs(modified_z_scores) < MODIFIED_ZSCORE_THRESHOLD).all(axis=1)
    df_cleaned = df[mask]
    
    return df_cleaned.values.mean()

def calculate_mean_percentile_method():
    lower = df.quantile(PERCENTILE_LOWER)
    upper = df.quantile(PERCENTILE_UPPER)
    mask = ~((df < lower) | (df > upper)).any(axis=1)
    df_cleaned = df[mask]
    
    return df_cleaned.values.mean()

iqr_result = calculate_mean_iqr_method()
zscore_result = calculate_mean_zscore_method()
modified_zscore_result = calculate_mean_modified_zscore_method()
percentile_result = calculate_mean_percentile_method()

methods_results = [iqr_result, zscore_result, modified_zscore_result, percentile_result]


valid_results = [result for result in methods_results if not np.isnan(result)]
if valid_results:
    relative_errors = [abs(result - CORRECT_MEAN) / abs(CORRECT_MEAN) * 100 for result in valid_results]
    TOLERANCE_PERCENT = sorted(relative_errors)[1] * TOLERANCE_MULTIPLIER
else:
    TOLERANCE_PERCENT = 0.5

TOLERANCE_ABSOLUTE = abs(CORRECT_MEAN) * TOLERANCE_PERCENT / 100

