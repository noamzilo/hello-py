import asyncio

from anthropic.types import ToolUnionParam

from .evaluator import run_single_test
from .tools import python_expression_tool, submit_answer_tool


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

	# Run the test 10 times and track success rate
	num_runs = 10
	expected_answer = 8769
	prompt = "Calculate (2^10 + 3^5) * 7 - 100. Use the python_expression tool and then submit the answer."

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
			verbose=False,
		)
		for i in range(num_runs)
	]

	# Run concurrently or sequentially based on the flag
	if concurrent:
		# Process results as they complete
		results = []
		for coro in asyncio.as_completed(tasks):
			result = await coro
			results.append(result)
	else:
		# Run sequentially by awaiting each task in order
		results = []
		for task in tasks:
			result = await task
			results.append(result)

	# Count successes
	successes = sum(1 for _, success, _ in results if success)

	# Calculate and display pass rate
	pass_rate = (successes / num_runs) * 100
	print(f"\n{'=' * 60}")
	print("Test Results:")
	print(f"  Passed: {successes}/{num_runs}")
	print(f"  Failed: {num_runs - successes}/{num_runs}")
	print(f"  Pass Rate: {pass_rate:.1f}%")
	print(f"{'=' * 60}")

