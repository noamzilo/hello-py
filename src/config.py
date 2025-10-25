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
TEST_PROMPT = """You have access to a pandas DataFrame named 'df' with numerical values that has been corrupted.

Your task:
1. Clean the data to recover the original mean as closely as possible. You are not told the corruption mode(s) upfront.
2. Compute the mean of the cleaned data (mean of all values across all rows and columns)
3. Submit the mean as your answer

Notes:
 - The data may contain various types of corruption. You'll need to identify and handle them appropriately.
 - You may not use any command that outputs more than 2*len(columns) values from the df to console.
 - Do not use comments in your Python code.

You have access to: pandas as 'pd', numpy as 'np', and the dataframe as 'df'.
Use the python_expression tool to write Python code, and submit_answer to submit your final result."""

TEST_MAX_STEPS = 10
TEST_VERBOSE = False

# Execution configuration
DEFAULT_CONCURRENT = False

EXPECTED_ANSWER = CORRECT_MEAN
ANSWER_TOLERANCE = TOLERANCE_ABSOLUTE
