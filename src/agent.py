import json
from collections.abc import Callable
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from .config import DEBUG, MAX_TOKENS, DEFAULT_MAX_STEPS, DEFAULT_MODEL, DEFAULT_VERBOSE


async def run_agent_loop(
	prompt: str,
	tools: list[ToolUnionParam],
	tool_handlers: dict[str, Callable[..., Any]],
	max_steps: int = DEFAULT_MAX_STEPS,
	model: str = DEFAULT_MODEL,
	verbose: bool = DEFAULT_VERBOSE,
) -> Any | None:
	"""
	Runs an agent loop with the given prompt and tools.

	Args:
		prompt: The initial prompt for the agent
		tools: List of tool definitions for Anthropic API
		tool_handlers: Dictionary mapping tool names to their handler functions
		max_steps: Maximum number of steps before stopping (default 5)
		model: The Anthropic model to use
		verbose: Whether to print detailed output (default True)

	Returns:
		The submitted answer if submit_answer was called, otherwise None
	"""
	client = AsyncAnthropic()
	messages: list[MessageParam] = [{"role": "user", "content": prompt}]

	for step in range(max_steps):
		steps_remaining = max_steps - step
		
		if verbose:
			print(f"\n{'='*70}")
			print(f"STEP {step + 1}/{max_steps} - {steps_remaining} steps remaining")
			print(f"{'='*70}")
		
		# Add step count information to the conversation
		step_info = f"\n\n[SYSTEM INFO] You are on step {step + 1} of {max_steps}. You have {steps_remaining} steps remaining. You can continue working or submit your answer using the submit_answer tool."
		messages.append({"role": "user", "content": step_info})
		
		if DEBUG:
			print(f"\n>>> Calling model: {model}")

		response = await client.messages.create(
			model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
		)
		
		if DEBUG:
			print(f"<<< Response stop_reason: {response.stop_reason}")

		assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
			f"unsupported stop_reason {response.stop_reason}"
		)
		if response.stop_reason == "max_tokens":
			print(
				f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
				"MAX_TOKENS, simplify your task, or update the code to provide "
				"a message back to the model when it exceeds MAX_TOKENS."
			)

		# Track if we need to continue
		has_tool_use = False
		tool_results = []
		submitted_answer = None

		# Process the response
		for content in response.content:
			if content.type == "text":
				if verbose or DEBUG:
					print(f"\n[Assistant]: {content.text}")
			elif content.type == "tool_use":
				has_tool_use = True
				tool_name = content.name
				
				if verbose or DEBUG:
					print(f"\n[Tool Call]: {tool_name}")

				if tool_name in tool_handlers:

					# Extract arguments based on tool
					handler = tool_handlers[tool_name]
					tool_input = content.input

					# Call the appropriate tool handler
					if tool_name == "python_expression":
						assert (
							isinstance(tool_input, dict) and "expression" in tool_input
						)
						if verbose or DEBUG:
							print("\n[Code]:")
							for line in tool_input["expression"].split("\n"):
								print(f"  {line}")
						result = handler(tool_input["expression"])
						if verbose or DEBUG:
							print(f"\n[Result]: {result}")
					elif tool_name == "submit_answer":
						assert isinstance(tool_input, dict) and "answer" in tool_input
						if verbose or DEBUG:
							print(f"\n[Submitting Answer]: {tool_input['answer']}")
						result = handler(tool_input["answer"])
						submitted_answer = result["answer"]
					else:
						# Generic handler call
						if verbose or DEBUG:
							print(f"\n[Tool Input]: {tool_input}")
						result = (
							handler(**tool_input)
							if isinstance(tool_input, dict)
							else handler(tool_input)
						)
						if verbose or DEBUG:
							print(f"\n[Tool Result]: {result}")

					tool_results.append(
						{
							"type": "tool_result",
							"tool_use_id": content.id,
							"content": json.dumps(result),
						}
					)

		# If we have tool uses, add them to the conversation
		if has_tool_use:
			messages.append({"role": "assistant", "content": response.content})
			messages.append({"role": "user", "content": tool_results})

			# If an answer was submitted, return it
			if submitted_answer is not None:
				if verbose or DEBUG:
					print(f"\n{'='*70}")
					print(f"FINAL ANSWER: {submitted_answer}")
					print(f"{'='*70}\n")
				return submitted_answer
		else:
			# No tool use, conversation might be complete
			if verbose or DEBUG:
				print(f"\n{'='*70}")
				print("No tool use in response, ending loop.")
				print(f"{'='*70}\n")
			break

	if verbose or DEBUG:
		print(f"\n{'='*70}")
		print(f"Reached maximum steps ({max_steps}) without submitting answer.")
		print(f"{'='*70}\n")
	return None

