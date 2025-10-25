# Dataset configuration
RANDOM_SEED = 42
NUM_ROWS = 150
NUM_COLS = 5
NUM_CLEAN_ROWS = 50

CLEAN_DATA_MEAN = 50
CLEAN_DATA_STD = 10

NOISE_UNIFORM_RANGE = 8
NOISE_NORMAL_STD = 5
NOISE_MULTIPLICATIVE_RANGE = 0.15
NOISE_EXPONENTIAL_SCALE = 3
NOISE_LAPLACE_SCALE = 4

MODERATE_OUTLIER_FRACTION = 0.2
MODERATE_OUTLIER_MIN_SIGMA = 1.5
MODERATE_OUTLIER_MAX_SIGMA = 2.5
OUTLIER_NEGATION_FREQUENCY = 3

NUM_DUPLICATE_ROWS = 15
NUM_SIGN_FLIP_ROWS = 10
SIGN_FLIP_MIN_COLS = 1
SIGN_FLIP_MAX_COLS = 4

NUM_DECIMAL_SHIFT_ROWS = 8
DECIMAL_SHIFT_MIN_COLS = 1
DECIMAL_SHIFT_MAX_COLS = 3
DECIMAL_SHIFT_MULTIPLIER = 10

NUM_MISSING_VALUE_ROWS = 12
MISSING_VALUE_MIN_COLS = 1
MISSING_VALUE_MAX_COLS = 3

NUM_ZERO_CORRUPTION_ROWS = 5
ZERO_CORRUPTION_MIN_COLS = 1
ZERO_CORRUPTION_MAX_COLS = 2

IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3
MODIFIED_ZSCORE_CONSTANT = 0.6745
MODIFIED_ZSCORE_THRESHOLD = 3.5
PERCENTILE_LOWER = 0.05
PERCENTILE_UPPER = 0.95

TOLERANCE_MULTIPLIER = 1.001

from src.environment import CORRECT_MEAN, TOLERANCE_ABSOLUTE

# Debug configuration
DEBUG = True

# Agent configuration
MAX_TOKENS = 1000
DEFAULT_MAX_STEPS = 20
DEFAULT_MODEL = "claude-haiku-4-5" # expensive for submitting
# DEFAULT_MODEL = "claude-3-haiku-20240307" # cheap for testing
DEFAULT_VERBOSE = True

# Test suite configuration
NUM_RUNS = 5

def create_prompt(max_steps: int) -> str:
    return f"""You have access to a pandas DataFrame named 'df' with numerical values that has been corrupted.

Your task:
1. Clean the data to recover the original mean as closely as possible. You are not told the corruption mode(s) upfront.
2. Compute the mean of the cleaned data (mean of all values across all rows and columns)
3. Submit the mean as your answer

IMPORTANT: You have a maximum of {max_steps} steps to complete this task. You MUST submit your answer using the submit_answer tool before reaching the maximum steps. If you don't submit an answer, the task will be considered failed.

Notes:
 - The data may contain various types of corruption. You'll need to identify and handle them appropriately.
 - You may not use any command that outputs more than 2*len(columns) values from the df to console.
 - Do not use comments in your Python code.
 - Each step will show you how many steps you have remaining.
 - Variables you define in one step will persist and be available in subsequent steps.

You also have initial access to: pandas as 'pd', numpy as 'np', scipy as 'scipy', and the raw dataframe as 'df'.  
Use the python_expression tool to write Python code, and submit_answer to submit your final result."""

TEST_MAX_STEPS = 10

# Keep the old TEST_PROMPT for backward compatibility, but it will be replaced by the function
TEST_PROMPT = create_prompt(TEST_MAX_STEPS)
TEST_VERBOSE = False

# Execution configuration
CONCURRENT = False

EXPECTED_ANSWER = CORRECT_MEAN
ANSWER_TOLERANCE = TOLERANCE_ABSOLUTE
