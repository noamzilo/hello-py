hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

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

All configuration values are centralized in `src/config.py`. You can modify these values to customize the behavior:

### Agent Configuration
- `MAX_TOKENS`: Maximum tokens for API calls (default: 1000)
- `DEFAULT_MAX_STEPS`: Maximum agent loop steps (default: 20)
- `DEFAULT_MODEL`: Anthropic model to use (default: "claude-3-haiku-20240307")
- `DEFAULT_VERBOSE`: Whether to show verbose output (default: True)

### Test Suite Configuration
- `NUM_RUNS`: Number of test iterations (default: 10)
- `EXPECTED_ANSWER`: Expected result for the test (default: 8769)
- `TEST_PROMPT`: The prompt given to the agent
- `TEST_MAX_STEPS`: Max steps for test runs (default: 5)
- `TEST_VERBOSE`: Verbose output for tests (default: False)

### Execution Configuration
- `DEFAULT_CONCURRENT`: Whether to run tests concurrently (default: True)

To modify any configuration, simply edit the values in `src/config.py` and restart the application.
