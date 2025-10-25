from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict
import re
import pandas as pd
import numpy as np
from .environment import df


class PythonExpressionToolResult(TypedDict):
	result: Any
	error: str | None


class SubmitAnswerToolResult(TypedDict):
	answer: Any
	submitted: bool


def is_allowed_summary_output(output: str) -> bool:
	output_lower = output.lower()
	
	# Check for describe() output (has statistical summary)
	stat_keywords = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
	if sum(1 for kw in stat_keywords if kw in output_lower) >= 3:
		return True
	
	# Check for info() output (has dtype, non-null, memory usage)
	info_keywords = ['dtype', 'non-null', 'memory usage', 'rangeindex', 'int64', 'float64']
	if any(keyword in output_lower for keyword in info_keywords):
		return True
	
	# Check for basic metadata
	metadata_keywords = ['shape', 'index', 'column']
	if any(keyword in output_lower for keyword in metadata_keywords):
		return True
	
	return False


def count_df_values_in_output(output: str, num_cols: int) -> int:	
	lines = output.strip().split('\n')
	
	if not lines:
		return 0
	
	value_count = 0
	in_dataframe = False
	
	for line in lines:
		if 'feature_' in line or (line.strip() and len(line.strip()) > 0 and line.strip()[0].isdigit() and '  ' in line):
			in_dataframe = True
		
		if in_dataframe and line.strip():
			numeric_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
			numbers = re.findall(numeric_pattern, line)
			
			if len(numbers) >= num_cols:
				value_count += num_cols
	
	return value_count


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
	"""
	Tool that evaluates Python expressions using exec.
	Use print(...) to emit output; stdout will be captured and returned.
	You have access to pandas as 'pd', numpy as 'np', and the dataframe as 'df'.
	"""
	try:
		if '#' in expression:
			return {
				"result": None,
				"error": "Comments are not allowed in Python code. Remove all # comments from your code."
			}
		
		namespace = {
			'pd': pd,
			'np': np,
			'df': df,
		}
		stdout = StringIO()
		with redirect_stdout(stdout):
			exec(expression, namespace, namespace)
		
		output = stdout.getvalue()
		num_cols = len(df.columns)
		max_allowed_values = 2 * num_cols
		
		value_count = count_df_values_in_output(output, num_cols)
		
		if not is_allowed_summary_output(output) and value_count > max_allowed_values:
			return {
				"result": None,
				"error": f"Output limit exceeded: {value_count} dataframe values output, but maximum allowed is {max_allowed_values} (2 * {num_cols} columns). Use more targeted queries like df.shape, df.dtypes, summary statistics, or small samples."
			}
		
		return {"result": output, "error": None}
	except KeyboardInterrupt:
		raise
	except Exception as e:
		return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
	"""
	Tool for submitting the final answer.
	"""
	return {"answer": answer, "submitted": True}

