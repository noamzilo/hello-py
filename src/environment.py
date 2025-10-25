import numpy as np
import pandas as pd

np.random.seed(42)

num_rows = 150
num_cols = 5

clean_data = np.random.normal(loc=50, scale=10, size=(50, num_cols))

CORRECT_MEAN = clean_data.mean()

data_with_noise = clean_data.copy()

for col_idx in range(num_cols):
    if col_idx == 0:
        noise = np.random.uniform(-8, 8, size=50)
    elif col_idx == 1:
        noise = np.random.normal(0, 5, size=50)
    elif col_idx == 2:
        noise = data_with_noise[:, col_idx] * np.random.uniform(-0.15, 0.15, size=50)
    elif col_idx == 3:
        noise = np.random.exponential(3, size=50) - 3
    elif col_idx == 4:
        noise = np.random.laplace(0, 4, size=50)
    
    data_with_noise[:, col_idx] += noise

extreme_outliers = np.array([
    [500, 480, 520, 510, 490],
    [600, 580, 610, 590, 620],
    [-300, -280, -310, -290, -320],
    [700, 720, 690, 710, 730],
    [-400, -420, -390, -410, -430],
    [800, 780, 820, 790, 810],
    [-500, -480, -520, -510, -490],
    [900, 920, 880, 910, 930],
    [-600, -580, -620, -590, -610],
    [1000, 1020, 980, 1010, 990],
])

duplicate_indices = np.random.choice(50, size=15, replace=True)
duplicate_rows = data_with_noise[duplicate_indices]

sign_flip_rows = data_with_noise[np.random.choice(50, size=10, replace=False)].copy()
for i in range(10):
    flip_cols = np.random.choice(num_cols, size=np.random.randint(1, 4), replace=False)
    sign_flip_rows[i, flip_cols] *= -1

decimal_shift_rows = data_with_noise[np.random.choice(50, size=8, replace=False)].copy()
for i in range(8):
    shift_cols = np.random.choice(num_cols, size=np.random.randint(1, 3), replace=False)
    for col in shift_cols:
        if np.random.rand() > 0.5:
            decimal_shift_rows[i, col] *= 10
        else:
            decimal_shift_rows[i, col] *= 0.1

missing_value_rows = data_with_noise[np.random.choice(50, size=12, replace=False)].copy()
for i in range(12):
    nan_cols = np.random.choice(num_cols, size=np.random.randint(1, 3), replace=False)
    missing_value_rows[i, nan_cols] = np.nan

zero_corruption_rows = data_with_noise[np.random.choice(50, size=5, replace=False)].copy()
for i in range(5):
    zero_cols = np.random.choice(num_cols, size=np.random.randint(1, 2), replace=False)
    zero_corruption_rows[i, zero_cols] = 0

all_data = np.vstack([
    data_with_noise,
    extreme_outliers,
    duplicate_rows,
    sign_flip_rows,
    decimal_shift_rows,
    missing_value_rows,
    zero_corruption_rows,
])

np.random.shuffle(all_data)

df = pd.DataFrame(all_data, columns=[f'feature_{i}' for i in range(num_cols)])

def calculate_mean_iqr_method():
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
    df_cleaned = df[mask]
    
    return df_cleaned.values.mean()

def calculate_mean_zscore_method():
    z_scores = np.abs((df - df.mean()) / df.std())
    mask = (z_scores < 3).all(axis=1)
    df_cleaned = df[mask]
    
    return df_cleaned.values.mean()

def calculate_mean_modified_zscore_method():
    median = df.median()
    mad = np.abs(df - median).median()
    modified_z_scores = 0.6745 * (df - median) / mad
    mask = (np.abs(modified_z_scores) < 3.5).all(axis=1)
    df_cleaned = df[mask]
    
    return df_cleaned.values.mean()

def calculate_mean_percentile_method():
    lower = df.quantile(0.05)
    upper = df.quantile(0.95)
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
    # relative_errors = [abs(result - CORRECT_MEAN) / abs(CORRECT_MEAN) * 100 for result in valid_results]
    # TOLERANCE_PERCENT = max(relative_errors) * 1.05
    TOLERANCE_PERCENT = CORRECT_MEAN * 0.02
else:
    TOLERANCE_PERCENT = 1.

