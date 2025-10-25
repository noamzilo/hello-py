# Debug configuration
DEBUG = True

# Agent configuration
MAX_TOKENS = 1000
DEFAULT_MAX_STEPS = 20
DEFAULT_MODEL = "claude-haiku-4-5" # expensive for submitting
# DEFAULT_MODEL = "claude-3-haiku-20240307" # cheap for testing
DEFAULT_VERBOSE = True

# Test suite configuration
NUM_RUNS = 1
TEST_PROMPT = """You have access to a pandas DataFrame named 'df' with numerical values containing extreme outliers.

Your task:
1. Inspect the dataframe to understand its structure
2. Remove the outliers using a reasonable statistical method (e.g., IQR method, Z-score, or similar)
3. Compute the mean of the cleaned data (mean of all values across all rows and columns)
4. Submit the mean as your answer

You have access to: pandas as 'pd', numpy as 'np', and the dataframe as 'df'.
Use the python_expression tool to write Python code, and submit_answer to submit your final result."""

TEST_MAX_STEPS = 10
TEST_VERBOSE = False

# Execution configuration
DEFAULT_CONCURRENT = True

from src.environment import CORRECT_MEAN, TOLERANCE
EXPECTED_ANSWER = CORRECT_MEAN
ANSWER_TOLERANCE = TOLERANCE
