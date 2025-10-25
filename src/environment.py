import numpy as np
import pandas as pd
from src.config import (
    RANDOM_SEED, NUM_ROWS, NUM_COLS, CLEAN_DATA_FRACTION,
    CLEAN_DATA_MEAN, CLEAN_DATA_STD,
    NOISE_UNIFORM_RANGE, NOISE_NORMAL_STD, NOISE_MULTIPLICATIVE_RANGE,
    NOISE_EXPONENTIAL_SCALE, NOISE_LAPLACE_SCALE,
    MODERATE_OUTLIER_FRACTION, MODERATE_OUTLIER_MIN_SIGMA, MODERATE_OUTLIER_MAX_SIGMA,
    OUTLIER_NEGATION_FREQUENCY,
    DUPLICATE_ROWS_FRACTION, SIGN_FLIP_ROWS_FRACTION, SIGN_FLIP_MIN_COLS, SIGN_FLIP_MAX_COLS,
    DECIMAL_SHIFT_ROWS_FRACTION, DECIMAL_SHIFT_MIN_COLS, DECIMAL_SHIFT_MAX_COLS, DECIMAL_SHIFT_MULTIPLIER,
    MISSING_VALUE_ROWS_FRACTION, MISSING_VALUE_MIN_COLS, MISSING_VALUE_MAX_COLS,
    ZERO_CORRUPTION_ROWS_FRACTION, ZERO_CORRUPTION_MIN_COLS, ZERO_CORRUPTION_MAX_COLS,
    IQR_BOUNDARY_ROWS_FRACTION, IQR_BOUNDARY_MIN_SIGMA, IQR_BOUNDARY_MAX_SIGMA,
    IQR_MULTIPLIER, ZSCORE_THRESHOLD, MODIFIED_ZSCORE_CONSTANT, MODIFIED_ZSCORE_THRESHOLD,
    PERCENTILE_LOWER, PERCENTILE_UPPER, TOLERANCE_MULTIPLIER
)


class DataGenerator:
    def __init__(self):
        np.random.seed(RANDOM_SEED)
        self.num_clean_rows = int(NUM_ROWS * CLEAN_DATA_FRACTION)
        self.num_duplicate_rows = int(NUM_ROWS * DUPLICATE_ROWS_FRACTION)
        self.num_sign_flip_rows = int(NUM_ROWS * SIGN_FLIP_ROWS_FRACTION)
        self.num_decimal_shift_rows = int(NUM_ROWS * DECIMAL_SHIFT_ROWS_FRACTION)
        self.num_missing_value_rows = int(NUM_ROWS * MISSING_VALUE_ROWS_FRACTION)
        self.num_zero_corruption_rows = int(NUM_ROWS * ZERO_CORRUPTION_ROWS_FRACTION)
        self.num_iqr_boundary_rows = int(NUM_ROWS * IQR_BOUNDARY_ROWS_FRACTION)
        self.num_moderate_outliers = int(NUM_ROWS * MODERATE_OUTLIER_FRACTION)
        
        self.clean_data = self._generate_clean_data()
        self.correct_mean = self.clean_data.mean()
    
    def _generate_clean_data(self):
        return np.random.normal(loc=CLEAN_DATA_MEAN, scale=CLEAN_DATA_STD, size=(self.num_clean_rows, NUM_COLS))
    
    def _add_noise(self, data):
        data_with_noise = data.copy()
        
        for col_idx in range(NUM_COLS):
            if col_idx == 0:
                noise = np.random.uniform(-NOISE_UNIFORM_RANGE, NOISE_UNIFORM_RANGE, size=self.num_clean_rows)
            elif col_idx == 1:
                noise = np.random.normal(0, NOISE_NORMAL_STD, size=self.num_clean_rows)
            elif col_idx == 2:
                noise = data_with_noise[:, col_idx] * np.random.uniform(-NOISE_MULTIPLICATIVE_RANGE, NOISE_MULTIPLICATIVE_RANGE, size=self.num_clean_rows)
            elif col_idx == 3:
                noise = np.random.exponential(NOISE_EXPONENTIAL_SCALE, size=self.num_clean_rows) - NOISE_EXPONENTIAL_SCALE
            elif col_idx == 4:
                noise = np.random.laplace(0, NOISE_LAPLACE_SCALE, size=self.num_clean_rows)
            
            data_with_noise[:, col_idx] += noise
        
        return data_with_noise
    
    def _generate_moderate_outliers(self):
        outlier_min = CLEAN_DATA_MEAN + MODERATE_OUTLIER_MIN_SIGMA * CLEAN_DATA_STD
        outlier_max = CLEAN_DATA_MEAN + MODERATE_OUTLIER_MAX_SIGMA * CLEAN_DATA_STD
        moderate_outliers = np.random.uniform(outlier_min, outlier_max, size=(self.num_moderate_outliers, NUM_COLS))
        
        for i in range(self.num_moderate_outliers):
            if i % OUTLIER_NEGATION_FREQUENCY == 0:
                moderate_outliers[i] *= -1
        
        return moderate_outliers
    
    def _generate_duplicate_rows(self, data):
        duplicate_indices = np.random.choice(self.num_clean_rows, size=self.num_duplicate_rows, replace=True)
        return data[duplicate_indices]
    
    def _generate_sign_flip_rows(self, data):
        sign_flip_rows = data[np.random.choice(self.num_clean_rows, size=self.num_sign_flip_rows, replace=False)].copy()
        
        for i in range(self.num_sign_flip_rows):
            flip_cols = np.random.choice(NUM_COLS, size=np.random.randint(SIGN_FLIP_MIN_COLS, SIGN_FLIP_MAX_COLS), replace=False)
            sign_flip_rows[i, flip_cols] *= -1
        
        return sign_flip_rows
    
    def _generate_decimal_shift_rows(self, data):
        decimal_shift_rows = data[np.random.choice(self.num_clean_rows, size=self.num_decimal_shift_rows, replace=False)].copy()
        
        for i in range(self.num_decimal_shift_rows):
            shift_cols = np.random.choice(NUM_COLS, size=np.random.randint(DECIMAL_SHIFT_MIN_COLS, DECIMAL_SHIFT_MAX_COLS), replace=False)
            for col in shift_cols:
                if np.random.rand() > 0.5:
                    decimal_shift_rows[i, col] *= DECIMAL_SHIFT_MULTIPLIER
                else:
                    decimal_shift_rows[i, col] /= DECIMAL_SHIFT_MULTIPLIER
        
        return decimal_shift_rows
    
    def _generate_missing_value_rows(self, data):
        missing_value_rows = data[np.random.choice(self.num_clean_rows, size=self.num_missing_value_rows, replace=False)].copy()
        
        for i in range(self.num_missing_value_rows):
            nan_cols = np.random.choice(NUM_COLS, size=np.random.randint(MISSING_VALUE_MIN_COLS, MISSING_VALUE_MAX_COLS), replace=False)
            missing_value_rows[i, nan_cols] = np.nan
        
        return missing_value_rows
    
    def _generate_zero_corruption_rows(self, data):
        zero_corruption_rows = data[np.random.choice(self.num_clean_rows, size=self.num_zero_corruption_rows, replace=False)].copy()
        
        for i in range(self.num_zero_corruption_rows):
            zero_cols = np.random.choice(NUM_COLS, size=np.random.randint(ZERO_CORRUPTION_MIN_COLS, ZERO_CORRUPTION_MAX_COLS), replace=False)
            zero_corruption_rows[i, zero_cols] = 0
        
        return zero_corruption_rows
    
    def _generate_iqr_boundary_outliers(self):
        """
        Generate balanced data around the true mean that IQR will reject but Z-score methods will keep.
        
        Strategy:
        - Create data at 2.0-2.8 sigma from the mean (70-78 and 22-30 for mean=50, std=10)
        - Split into upper and lower halves to maintain balance around true mean
        - IQR method (conservative) will reject this data as outliers
        - Z-score methods (threshold=3) will keep this data as it's within 3 sigma
        - This makes IQR lose valuable balanced data, causing its mean estimate to drift
        """
        half = self.num_iqr_boundary_rows // 2
        
        # Upper half: values slightly above mean (e.g., 70-78 for mean=50)
        upper_values = np.random.uniform(
            CLEAN_DATA_MEAN + IQR_BOUNDARY_MIN_SIGMA * CLEAN_DATA_STD,
            CLEAN_DATA_MEAN + IQR_BOUNDARY_MAX_SIGMA * CLEAN_DATA_STD,
            size=(half, NUM_COLS)
        )
        
        # Lower half: values slightly below mean (e.g., 22-30 for mean=50)
        lower_values = np.random.uniform(
            CLEAN_DATA_MEAN - IQR_BOUNDARY_MAX_SIGMA * CLEAN_DATA_STD,
            CLEAN_DATA_MEAN - IQR_BOUNDARY_MIN_SIGMA * CLEAN_DATA_STD,
            size=(self.num_iqr_boundary_rows - half, NUM_COLS)
        )
        
        # Combine upper and lower values to maintain balance around true mean
        iqr_boundary_outliers = np.vstack([upper_values, lower_values])
        
        return iqr_boundary_outliers
    
    def generate_corrupted_dataset(self):
        data_with_noise = self._add_noise(self.clean_data)
        
        all_data = np.vstack([
            data_with_noise,
            self._generate_moderate_outliers(),
            self._generate_duplicate_rows(data_with_noise),
            self._generate_sign_flip_rows(data_with_noise),
            self._generate_decimal_shift_rows(data_with_noise),
            self._generate_missing_value_rows(data_with_noise),
            self._generate_zero_corruption_rows(data_with_noise),
            self._generate_iqr_boundary_outliers(),
        ])
        
        np.random.shuffle(all_data)
        
        return pd.DataFrame(all_data, columns=[f'feature_{i}' for i in range(NUM_COLS)])


class DataProcessor:
    def __init__(self, dataframe):
        self.df_clean = dataframe.dropna()
    
    def calculate_mean_iqr_method(self):
        Q1 = self.df_clean.quantile(0.25)
        Q3 = self.df_clean.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR
        
        mask = ~((self.df_clean < lower_bound) | (self.df_clean > upper_bound)).any(axis=1)
        df_cleaned = self.df_clean[mask]
        
        return df_cleaned.values.mean()

    def calculate_mean_zscore_method(self):
        z_scores = np.abs((self.df_clean - self.df_clean.mean()) / self.df_clean.std())
        mask = (z_scores < ZSCORE_THRESHOLD).all(axis=1)
        df_cleaned = self.df_clean[mask]
        
        return df_cleaned.values.mean()

    def calculate_mean_modified_zscore_method(self):
        median = self.df_clean.median()
        mad = np.abs(self.df_clean - median).median()
        modified_z_scores = MODIFIED_ZSCORE_CONSTANT * (self.df_clean - median) / mad
        mask = (np.abs(modified_z_scores) < MODIFIED_ZSCORE_THRESHOLD).all(axis=1)
        df_cleaned = self.df_clean[mask]
        
        return df_cleaned.values.mean()

    def calculate_mean_percentile_method(self):
        lower = self.df_clean.quantile(PERCENTILE_LOWER)
        upper = self.df_clean.quantile(PERCENTILE_UPPER)
        mask = ~((self.df_clean < lower) | (self.df_clean > upper)).any(axis=1)
        df_cleaned = self.df_clean[mask]
        
        return df_cleaned.values.mean()


class Environment:
    def __init__(self):
        self.data_generator = DataGenerator()
        self.df = self.data_generator.generate_corrupted_dataset()
        self.processor = DataProcessor(self.df)
        
        self.iqr_result = self.processor.calculate_mean_iqr_method()
        self.zscore_result = self.processor.calculate_mean_zscore_method()
        self.modified_zscore_result = self.processor.calculate_mean_modified_zscore_method()
        self.percentile_result = self.processor.calculate_mean_percentile_method()
        
        self.methods_results = [
            self.iqr_result,
            self.zscore_result,
            self.modified_zscore_result,
            self.percentile_result
        ]
        
        self._calculate_tolerance()
    
    def _calculate_tolerance(self):
        relative_errors = [
            abs(result - self.data_generator.correct_mean) / abs(self.data_generator.correct_mean) * 100
            for result in self.methods_results
        ]
        self.tolerance_percent = sorted(relative_errors)[1] * TOLERANCE_MULTIPLIER
        self.tolerance_absolute = abs(self.data_generator.correct_mean) * self.tolerance_percent / 100
    
    @property
    def CORRECT_MEAN(self):
        return self.data_generator.correct_mean
    
    @property
    def TOLERANCE_PERCENT(self):
        return self.tolerance_percent
    
    @property
    def TOLERANCE_ABSOLUTE(self):
        return self.tolerance_absolute


env = Environment()

CORRECT_MEAN = env.CORRECT_MEAN
TOLERANCE_PERCENT = env.TOLERANCE_PERCENT
TOLERANCE_ABSOLUTE = env.TOLERANCE_ABSOLUTE
df = env.df
iqr_result = env.iqr_result
zscore_result = env.zscore_result
modified_zscore_result = env.modified_zscore_result
percentile_result = env.percentile_result
methods_results = env.methods_results
