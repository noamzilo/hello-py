# Debug configuration
DEBUG = True

# Agent configuration
MAX_TOKENS = 1000
DEFAULT_MAX_STEPS = 20
# DEFAULT_MODEL = "claude-haiku-4-5" # expensive for submitting
DEFAULT_MODEL = "claude-3-haiku-20240307" # cheap for testing
DEFAULT_VERBOSE = True

# Test suite configuration
NUM_RUNS = 1
EXPECTED_ANSWER = 8769
TEST_PROMPT = "Calculate (2^10 + 3^5) * 7 - 100. Use the python_expression tool and then submit the answer."
TEST_MAX_STEPS = 5
TEST_VERBOSE = False

# Execution configuration
DEFAULT_CONCURRENT = True
