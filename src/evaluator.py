from collections.abc import Callable
from typing import Any

from anthropic.types import ToolUnionParam

from .agent import run_agent_loop


async def run_single_test(
	run_id: int,
	num_runs: int,
	prompt: str,
	tools: list[ToolUnionParam],
	tool_handlers: dict[str, Callable[..., Any]],
	expected_answer: Any,
	verbose: bool = False,
) -> tuple[int, bool, Any]:
	if verbose:
		print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

	result = await run_agent_loop(
		prompt=prompt,
		tools=tools,
		tool_handlers=tool_handlers,
		max_steps=5,
		verbose=verbose,
	)

	success = result == expected_answer

	if success:
		print(f"✓ Run {run_id}: SUCCESS - Got {result}")
	else:
		print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer}")

	return run_id, success, result

