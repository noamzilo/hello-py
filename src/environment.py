import numpy as np
import pandas as pd

np.random.seed(42)

num_rows = 100
num_cols = 5

normal_data = np.random.normal(loc=50, scale=10, size=(num_rows - 10, num_cols))

noise = np.random.uniform(-5, 5, size=(num_rows - 10, num_cols))
data_with_noise = normal_data + noise

outliers = np.array([
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

all_data = np.vstack([data_with_noise, outliers])

np.random.shuffle(all_data)

df = pd.DataFrame(all_data, columns=[f'feature_{i}' for i in range(num_cols)])

def calculate_correct_mean_iqr():
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
    df_cleaned = df[mask]
    
    return df_cleaned.values.mean()

CORRECT_MEAN = calculate_correct_mean_iqr()

