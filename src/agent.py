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
		if verbose:
			print(f"\n=== Step {step + 1}/{max_steps} ===")
		
		if DEBUG:
			print(f"[DEBUG] Sending {len(messages)} messages to model {model}")
			for i, msg in enumerate(messages):
				if msg["role"] == "user":
					if isinstance(msg["content"], str):
						print(f"[DEBUG] User message {i}: {msg['content'][:100]}...")
					else:
						print(f"[DEBUG] User message {i}: {len(msg['content'])} tool results")
				else:
					print(f"[DEBUG] Assistant message {i}: {len(msg['content'])} content blocks")

		response = await client.messages.create(
			model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
		)
		
		if DEBUG:
			print(f"[DEBUG] Response stop_reason: {response.stop_reason}")
			print(f"[DEBUG] Response has {len(response.content)} content blocks")

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
				if verbose:
					print(f"Assistant: {content.text}")
				if DEBUG:
					print(f"[DEBUG] Assistant text content: {content.text[:200]}...")
			elif content.type == "tool_use":
				has_tool_use = True
				tool_name = content.name
				
				if DEBUG:
					print(f"[DEBUG] Tool use detected: {tool_name}")
					print(f"[DEBUG] Tool input: {content.input}")

				if tool_name in tool_handlers:
					if verbose:
						print(f"Using tool: {tool_name}")

					# Extract arguments based on tool
					handler = tool_handlers[tool_name]
					tool_input = content.input

					# Call the appropriate tool handler
					if tool_name == "python_expression":
						assert (
							isinstance(tool_input, dict) and "expression" in tool_input
						)
						if verbose:
							print("\nInput:")
							print("```")
							for line in tool_input["expression"].split("\n"):
								print(f"{line}")
							print("```")
						if DEBUG:
							print(f"[DEBUG] Executing Python expression: {tool_input['expression']}")
						result = handler(tool_input["expression"])
						if DEBUG:
							print(f"[DEBUG] Python execution result: {result}")
						if verbose:
							print("\nOutput:")
							print("```")
							print(result)
							print("```")
					elif tool_name == "submit_answer":
						assert isinstance(tool_input, dict) and "answer" in tool_input
						if DEBUG:
							print(f"[DEBUG] Submitting answer: {tool_input['answer']}")
						result = handler(tool_input["answer"])
						submitted_answer = result["answer"]
						if DEBUG:
							print(f"[DEBUG] Answer submission result: {result}")
					else:
						# Generic handler call
						if DEBUG:
							print(f"[DEBUG] Calling generic handler for {tool_name} with input: {tool_input}")
						result = (
							handler(**tool_input)
							if isinstance(tool_input, dict)
							else handler(tool_input)
						)
						if DEBUG:
							print(f"[DEBUG] Generic handler result: {result}")

					tool_results.append(
						{
							"type": "tool_result",
							"tool_use_id": content.id,
							"content": json.dumps(result),
						}
					)

		# If we have tool uses, add them to the conversation
		if has_tool_use:
			if DEBUG:
				print(f"[DEBUG] Adding assistant response to conversation ({len(response.content)} content blocks)")
			messages.append({"role": "assistant", "content": response.content})

			if DEBUG:
				print(f"[DEBUG] Adding {len(tool_results)} tool results to conversation")
			messages.append({"role": "user", "content": tool_results})

			# If an answer was submitted, return it
			if submitted_answer is not None:
				if verbose:
					print(f"\nAgent submitted answer: {submitted_answer}")
				if DEBUG:
					print(f"[DEBUG] Answer submitted, ending agent loop")
				return submitted_answer
		else:
			# No tool use, conversation might be complete
			if verbose:
				print("\nNo tool use in response, ending loop.")
			if DEBUG:
				print(f"[DEBUG] No tool use detected, ending agent loop")
			break

	if verbose:
		print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
	return None

