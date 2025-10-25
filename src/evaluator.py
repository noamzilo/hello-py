from collections.abc import Callable
from typing import Any

from anthropic.types import ToolUnionParam

from .agent import run_agent_loop
from .config import DEBUG, TEST_MAX_STEPS


async def run_single_test(
	run_id: int,
	num_runs: int,
	prompt: str,
	tools: list[ToolUnionParam],
	tool_handlers: dict[str, Callable[..., Any]],
	expected_answer: Any,
	tolerance: float = 0.0,
	verbose: bool = False,
) -> tuple[int, bool, Any]:
	if verbose:
		print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")
	
	if DEBUG:
		print(f"[DEBUG] Starting test run {run_id}/{num_runs}")
		print(f"[DEBUG] Expected answer: {expected_answer}")
		print(f"[DEBUG] Prompt: {prompt[:100]}...")

	result = await run_agent_loop(
		prompt=prompt,
		tools=tools,
		tool_handlers=tool_handlers,
		max_steps=TEST_MAX_STEPS,
		verbose=verbose,
	)

	if result is None:
		# Handle case where agent didn't submit an answer
		success = False
	elif tolerance > 0 and isinstance(result, (int, float)) and isinstance(expected_answer, (int, float)):
		# Use absolute tolerance
		absolute_error = abs(result - expected_answer)
		success = absolute_error <= tolerance
	else:
		success = result == expected_answer
	
	if DEBUG:
		print(f"[DEBUG] Test run {run_id} completed")
		print(f"[DEBUG] Result: {result}")
		print(f"[DEBUG] Expected: {expected_answer}")
		if tolerance > 0:
			print(f"[DEBUG] Tolerance: {tolerance}")
			if result is not None and isinstance(result, (int, float)):
				print(f"[DEBUG] Absolute error: {abs(result - expected_answer)}")
			else:
				print(f"[DEBUG] Absolute error: N/A (result is None or not numeric)")
		print(f"[DEBUG] Success: {success}")

	if success:
		if tolerance > 0:
			print(f"✓ Run {run_id}: SUCCESS - Got {result} (within tolerance of {expected_answer} ± {tolerance:.6f})")
		else:
			print(f"✓ Run {run_id}: SUCCESS - Got {result}")
	else:
		if result is None:
			print(f"✗ Run {run_id}: FAILURE - No answer submitted (agent reached max steps or didn't use submit_answer tool)")
		elif tolerance > 0:
			absolute_error = abs(result - expected_answer)
			print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer} ± {tolerance:.6f} (actual error: {absolute_error:.6f})")
		else:
			print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer}")

	return run_id, success, result

