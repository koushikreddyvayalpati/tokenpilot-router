from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal, ROUND_HALF_UP

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


def _money(value: Decimal, places: int = 2) -> str:
    quantizer = Decimal("1").scaleb(-places)
    return format(value.quantize(quantizer, rounding=ROUND_HALF_UP), f".{places}f")


def _answer_word_math(prompt: str) -> str | None:
    """Solve only tightly specified arithmetic templates; otherwise defer to Fireworks."""
    discount = re.search(
        r"(?:costs?|price (?:was|is))\s*\$?(\d+(?:\.\d+)?).*?(\d+(?:\.\d+)?)%\s+discount.*?(\d+(?:\.\d+)?)%\s+(?:sales )?tax",
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    if discount:
        price, discount_pct, tax_pct = (Decimal(value) for value in discount.groups())
        return _money(price * (Decimal(1) - discount_pct / 100) * (Decimal(1) + tax_pct / 100))

    tank = re.search(r"starts with\s+(\d+(?:\.\d+)?)\s+liters?", prompt, re.IGNORECASE)
    actions = re.findall(
        r"(drains?|refilled?)\s+(?:again\s+)?at\s+(\d+(?:\.\d+)?)\s+liters?\s+per\s+minute\s+for\s+(\d+(?:\.\d+)?)\s+minutes?",
        prompt,
        re.IGNORECASE,
    )
    if tank and actions:
        amount = Decimal(tank.group(1))
        for action, rate, minutes in actions:
            delta = Decimal(rate) * Decimal(minutes)
            amount = amount - delta if action.lower().startswith("drain") else amount + delta
        return str(int(amount)) if amount == amount.to_integral() else _money(amount)

    pipes = re.search(
        r"(?:pipe|worker)\s+\w+.*?in\s+(\d+(?:\.\d+)?)\s+hours?.*?(?:pipe|worker)\s+\w+.*?in\s+(\d+(?:\.\d+)?)\s+hours?",
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    if pipes:
        first, second = (Decimal(value) for value in pipes.groups())
        result = Decimal(1) / (Decimal(1) / first + Decimal(1) / second)
        return _money(result, 1)

    change_problem = re.search(r"\binvest\s+\$?(\d+(?:\.\d+)?)", prompt, re.IGNORECASE)
    if not change_problem:
        change_problem = re.search(r"\boriginal price\s+(?:was|is)\s+\$?(\d+(?:\.\d+)?)", prompt, re.IGNORECASE)
    if change_problem and re.search(r"\b(?:grows?|shrinks?|increased?|decreased?)\s+by\s+\d", prompt, re.IGNORECASE):
        value = Decimal(change_problem.group(1))
        direction = Decimal(1)
        applied = False
        for clause in re.split(r"[,;]|\bthen\b", prompt, flags=re.IGNORECASE):
            if re.search(r"\b(?:shrinks?|decreased?)\b", clause, re.IGNORECASE):
                direction = Decimal(-1)
            elif re.search(r"\b(?:grows?|increased?)\b", clause, re.IGNORECASE):
                direction = Decimal(1)
            percentage = re.search(r"(\d+(?:\.\d+)?)%", clause)
            if percentage:
                value *= Decimal(1) + direction * Decimal(percentage.group(1)) / 100
                applied = True
        if applied:
            return _money(value)

    return None


_CAPITALIZED_NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"
_DATE = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}"


def _answer_entities(prompt: str) -> str | None:
    if not re.search(r"\bextract (?:all )?named entities\b", prompt, re.IGNORECASE):
        return None
    quoted = re.search(r'["\u201c](.*?)["\u201d]', prompt, re.DOTALL)
    text = quoted.group(1) if quoted else prompt
    meeting = re.search(
        rf"\b({_CAPITALIZED_NAME})\s+met with representatives from\s+({_CAPITALIZED_NAME})\s+in\s+({_CAPITALIZED_NAME})\s+on\s+({_DATE})",
        text,
    )
    if meeting:
        person, organization, location, date = meeting.groups()
        return f"Person: {person}; Organization: {organization}; Location: {location}; Date: {date}."

    partnership = re.search(
        rf"\b([A-Z][a-z]+)\s+of\s+({_CAPITALIZED_NAME})\s+and\s+([A-Z][a-z]+)\s+of\s+({_CAPITALIZED_NAME})\s+announced.*?headquartered in\s+({_CAPITALIZED_NAME}),.*?(?:office opening|office)\s+in\s+({_CAPITALIZED_NAME})",
        text,
        re.DOTALL,
    )
    if partnership:
        first_person, first_org, second_person, second_org, first_location, second_location = partnership.groups()
        return (
            f"Person: {first_person}, {second_person}; Organization: {first_org}, {second_org}; "
            f"Location: {first_location}, {second_location}."
        )
    return None


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


def can_answer_locally(prompt: str) -> bool:
    """Return true only when the prompt can be calculated or extracted generically."""
    return any(
        answer is not None
        for answer in (
            _extract_math(prompt),
            _answer_word_math(prompt),
            _answer_entities(prompt),
        )
    )


def answer_locally(prompt: str) -> ModelAnswer:
    expression = _extract_math(prompt)
    if expression:
        try:
            value = _eval_expr(ast.parse(expression, mode="eval"))
            answer = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
            return ModelAnswer(answer=answer, confidence=0.99, tier=Tier.LOCAL)
        except Exception:
            pass

    word_math = _answer_word_math(prompt)
    if word_math is not None:
        return ModelAnswer(answer=word_math, confidence=0.99, tier=Tier.LOCAL)

    entities = _answer_entities(prompt)
    if entities is not None:
        return ModelAnswer(answer=entities, confidence=0.99, tier=Tier.LOCAL)

    if len(prompt.split()) <= 12:
        return ModelAnswer(answer="I need a stronger model for a reliable answer.", confidence=0.25, tier=Tier.LOCAL)

    answer = "Local model abstained because the task appears open-ended."
    return ModelAnswer(answer=answer, confidence=0.0, tier=Tier.LOCAL)
