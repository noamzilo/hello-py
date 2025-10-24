from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict
import pandas as pd
import numpy as np
from .environment import df


class PythonExpressionToolResult(TypedDict):
	result: Any
	error: str | None


class SubmitAnswerToolResult(TypedDict):
	answer: Any
	submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
	"""
	Tool that evaluates Python expressions using exec.
	Use print(...) to emit output; stdout will be captured and returned.
	You have access to pandas as 'pd', numpy as 'np', and the dataframe as 'df'.
	"""
	try:
		namespace = {
			'pd': pd,
			'np': np,
			'df': df,
		}
		stdout = StringIO()
		with redirect_stdout(stdout):
			exec(expression, namespace, namespace)
		return {"result": stdout.getvalue(), "error": None}
	except KeyboardInterrupt:
		raise
	except Exception as e:
		return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
	"""
	Tool for submitting the final answer.
	"""
	return {"answer": answer, "submitted": True}

