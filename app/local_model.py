from __future__ import annotations

import ast
import operator

from app.confidence import score_answer
from app.models import ModelAnswer, Tier

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("unsupported expression")


def _extract_math(prompt: str) -> str | None:
    cleaned = prompt.lower().replace("what is", "").replace("calculate", "").replace("compute", "")
    cleaned = cleaned.replace("?", "").strip()
    if not cleaned or any(ch not in "0123456789+-*/(). %\t\n" for ch in cleaned):
        return None
    return cleaned.replace("%", "/100")


def answer_locally(prompt: str) -> ModelAnswer:
    expression = _extract_math(prompt)
    if expression:
        try:
            value = _eval_expr(ast.parse(expression, mode="eval"))
            answer = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
            return ModelAnswer(answer=answer, confidence=0.99, tier=Tier.LOCAL)
        except Exception:
            pass

    if len(prompt.split()) <= 12:
        answer = "I need a stronger model for a reliable answer."
        return ModelAnswer(answer=answer, confidence=0.25, tier=Tier.LOCAL)

    answer = "Local model abstained because the task appears open-ended."
    return ModelAnswer(answer=answer, confidence=score_answer(answer), tier=Tier.LOCAL)

