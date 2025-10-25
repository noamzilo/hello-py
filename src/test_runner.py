import asyncio

from anthropic.types import ToolUnionParam

from .evaluator import run_single_test
from .tools import python_expression_tool, submit_answer_tool
from .config import DEBUG, NUM_RUNS, EXPECTED_ANSWER, ANSWER_TOLERANCE, TEST_PROMPT, TEST_MAX_STEPS, TEST_VERBOSE


async def run_test_suite(concurrent: bool = True):
	tools: list[ToolUnionParam] = [
		{
			"name": "python_expression",
			"description": "Evaluates a Python expression",
			"input_schema": {
				"type": "object",
				"properties": {
					"expression": {
						"type": "string",
						"description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
					}
				},
				"required": ["expression"],
			},
		},
		{
			"name": "submit_answer",
			"description": "Submit the final answer",
			"input_schema": {
				"type": "object",
				"properties": {"answer": {"description": "The final answer to submit"}},
				"required": ["answer"],
			},
		},
	]

	tool_handlers = {
		"python_expression": python_expression_tool,
		"submit_answer": submit_answer_tool,
	}

	# Run the test and track success rate
	num_runs = NUM_RUNS
	expected_answer = EXPECTED_ANSWER
	prompt = TEST_PROMPT

	if DEBUG:
		print(f"[DEBUG] Test suite configuration:")
		print(f"[DEBUG] - Number of runs: {num_runs}")
		print(f"[DEBUG] - Expected answer: {expected_answer}")
		print(f"[DEBUG] - Tolerance: {ANSWER_TOLERANCE}")
		print(f"[DEBUG] - Concurrent execution: {concurrent}")
		print(f"[DEBUG] - Prompt: {prompt}")

	execution_mode = "concurrently" if concurrent else "sequentially"
	print(f"Running {num_runs} test iterations {execution_mode}...")
	print("=" * 60)

	# Create all test coroutines
	tasks = [
		run_single_test(
			run_id=i + 1,
			num_runs=num_runs,
			prompt=prompt,
			tools=tools,
			tool_handlers=tool_handlers,
			expected_answer=expected_answer,
			tolerance=ANSWER_TOLERANCE,
			verbose=TEST_VERBOSE,
		)
		for i in range(num_runs)
	]

	# Run concurrently or sequentially based on the flag
	if concurrent:
		if DEBUG:
			print(f"[DEBUG] Running {len(tasks)} tasks concurrently")
		# Process results as they complete
		results = []
		for coro in asyncio.as_completed(tasks):
			result = await coro
			results.append(result)
			if DEBUG:
				print(f"[DEBUG] Completed task {result[0]}, success: {result[1]}")
	else:
		if DEBUG:
			print(f"[DEBUG] Running {len(tasks)} tasks sequentially")
		# Run sequentially by awaiting each task in order
		results = []
		for task in tasks:
			result = await task
			results.append(result)
			if DEBUG:
				print(f"[DEBUG] Completed task {result[0]}, success: {result[1]}")

	# Count successes
	successes = sum(1 for _, success, _ in results if success)
	
	if DEBUG:
		print(f"[DEBUG] Final results summary:")
		for run_id, success, result in results:
			print(f"[DEBUG] - Run {run_id}: {'SUCCESS' if success else 'FAILURE'} (result: {result})")

	# Calculate and display pass rate
	pass_rate = (successes / num_runs) * 100
	print(f"\n{'=' * 60}")
	print("Test Results:")
	print(f"  Passed: {successes}/{num_runs}")
	print(f"  Failed: {num_runs - successes}/{num_runs}")
	print(f"  Pass Rate: {pass_rate:.1f}%")
	print(f"{'=' * 60}")

