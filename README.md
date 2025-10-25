hello-py
===

## What This Does

This is an RL task for LLM training that tests whether a model can correctly identify and clean corrupted numerical data. The agent is given a pandas DataFrame with various types of data corruption (outliers, sign flips, decimal shifts, missing values, etc.) and must:

1. Explore and understand the corruption patterns
2. Apply appropriate outlier detection and data cleaning methods
3. Calculate and submit the mean of the cleaned dataset

The task requires the model to choose from multiple valid approaches (IQR, Z-score, Modified Z-score, Percentile methods) to successfully recover the original mean within a tolerance threshold. The dataset is designed so that multiple methods can succeed, but naive approaches will fail.

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/noamzilo/hello-py
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

5. Run the agent:
   ```
   uv run main.py
   ```

## Project Structure

```
hello-py/
├── main.py                  # Entry point - runs the test suite
├── pyproject.toml          # Project dependencies and metadata
├── uv.lock                 # Locked dependencies
├── .env.example            # Example environment variables file
├── .env                    # Your environment variables (create from .env.example)
├── README.md               # This file
├── rl_llm_exercise.md      # Original exercise description
└── src/
    ├── __init__.py         # Package initialization
    ├── config.py           # All configuration values (dataset, agent, test settings)
    ├── environment.py      # Dataset generation and corruption logic
    ├── agent.py            # Agent loop - handles tool calls and conversation
    ├── tools.py            # Tool definitions (python_expression, submit_answer)
    ├── evaluator.py        # Evaluates single test runs and checks success
    └── test_runner.py      # Runs multiple test iterations and reports results
```

### Key Files Explained

- **`main.py`**: Entry point that loads environment variables and runs the test suite
- **`src/config.py`**: Centralized configuration for dataset parameters, agent settings, and test configuration
- **`src/environment.py`**: 
  - `DataGenerator`: Creates clean data and applies various corruption patterns
  - `DataProcessor`: Implements different outlier detection methods (IQR, Z-score, etc.)
  - `Environment`: Orchestrates dataset generation and calculates success thresholds
- **`src/agent.py`**: Implements the agent loop that processes prompts, makes tool calls, and interacts with Claude
- **`src/tools.py`**: Defines tools available to the agent:
  - `python_expression_tool`: Executes Python code with output restrictions
  - `submit_answer_tool`: Submits the final answer
- **`src/evaluator.py`**: Runs individual test cases and evaluates whether the agent's answer is within tolerance
- **`src/test_runner.py`**: Runs multiple test iterations and aggregates results to calculate pass rate

## Execution Modes

**Note:** Concurrent mode has been deprecated due to rate limit issues and shared memory conflicts between parallel agent runs. Fixing this was not part of the exercise scope. The system now runs in sequential mode only.

The execution mode can be configured via the `CONCURRENT` flag in `src/config.py` (currently set to `False`).

## Debug Mode

The system includes a debug flag that shows detailed thinking process and intermediate steps when enabled.

To enable debug mode:

1. Edit `src/config.py` and set `DEBUG = True`
2. Run the test suite as normal: `uv run main.py`

In debug mode, you'll see:
- Detailed conversation flow between user and assistant
- Tool usage and execution details
- Python expression execution steps
- Answer submission process
- Test execution progress

The debug output will show detailed information about each step of the agent's thinking process.

## Configuration

All configuration values are centralized in `src/config.py`. The configuration is divided into several sections:

### Human-Configurable Settings

These are the primary settings you may want to adjust:

#### Agent Configuration
- `DEFAULT_MODEL`: Anthropic model to use (default: `"claude-haiku-4-5"`)
  - Use `"claude-haiku-4-5"` for production/submission (expensive but powerful)
  - Use `"claude-3-haiku-20240307"` for testing (cheaper)
- `MAX_TOKENS`: Maximum tokens for API calls (default: 1000)
- `DEFAULT_MAX_STEPS`: Maximum agent loop steps (default: 20)
- `DEFAULT_VERBOSE`: Whether to show verbose output (default: True)

#### Test Suite Configuration
- `NUM_RUNS`: Number of test iterations to measure pass rate (default: 10)
- `TEST_MAX_STEPS`: Max steps allowed per test run (default: 10)
- `TEST_VERBOSE`: Verbose output for individual test runs (default: False)

#### Debug Configuration
- `DEBUG`: Enable detailed debug output (default: True)

#### Execution Configuration
- `CONCURRENT`: Whether to run tests concurrently (default: False, deprecated)

### Dataset Configuration (Advanced)

These control the dataset generation and corruption patterns. **Generally you should not modify these** unless you're specifically trying to adjust task difficulty:

- `NUM_ROWS`, `NUM_COLS`: Dataset dimensions
- `CLEAN_DATA_FRACTION`: Percentage of rows that are clean data
- Noise parameters: `NOISE_UNIFORM_RANGE`, `NOISE_NORMAL_STD`, etc.
- Corruption fractions: `DUPLICATE_ROWS_FRACTION`, `SIGN_FLIP_ROWS_FRACTION`, etc.
- Outlier detection thresholds: `IQR_MULTIPLIER`, `ZSCORE_THRESHOLD`, etc.

### Auto-Calculated Values

These are computed automatically from the dataset and should not be modified:
- `EXPECTED_ANSWER`: The correct mean value (calculated from clean data)
- `ANSWER_TOLERANCE`: The tolerance threshold (calculated from method performance)
- `TOLERANCE_MULTIPLIER`: Multiplier for setting tolerance (default: 1.001)

To modify any configuration, edit the values in `src/config.py` and restart the application.
