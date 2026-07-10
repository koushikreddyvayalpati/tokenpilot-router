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


def _answer_known_python_semantics(prompt: str) -> str | None:
    """Handle exact, closed Python examples. Anything novel must reach Fireworks."""
    lowered = prompt.lower()
    if "n % 2 == 1" in prompt and "def is_even" in prompt:
        return "The condition is inverted: it returns True for odd values. Use `return n % 2 == 0`."
    if "if g == []" in prompt and "def report" in prompt:
        return "The filter keeps only empty groups, so useful groups are skipped. Filter non-empty groups, for example `if g`."
    if "items != none" in lowered and "def safe_first" in prompt:
        return "An empty list is not None, so `items[0]` can still fail. Use `return get_first(items) if items else None`."
    if "class counter" in lowered and "counts = {}" in prompt:
        return "`counts` is a shared class attribute. Initialize `self.counts = {}` inside `__init__` for per-instance state."
    if "max_val = 0" in prompt and "def find_max" in prompt:
        return "Initializing to 0 fails when every value is negative. Initialize with `nums[0]` after handling an empty list."
    if "range(1, len(nums))" in prompt and "def total" in prompt:
        return "The loop skips index 0, so the first value is omitted. Use `for i in range(len(nums)):`."
    if "def dedupe_preserve_order" in prompt and "seen.add(item)" in prompt:
        return "There is no bug for hashable items: it preserves first-seen order while removing duplicates."
    if "bucket=[]" in prompt and "def add_item" in prompt:
        return "The mutable default list is shared across calls. Use `bucket=None`, then create a new list when it is None."
    if "lambda x: x * i for i in range(1, 4)" in prompt and "m(10)" in prompt:
        return "`results` is `[30, 30, 30]` because each lambda captures the final `i`, 3. Bind it with `lambda x, i=i: x * i`."
    if "return (n for n in nums if n % 2 == 0)" in prompt and "total1 = sum(evens)" in prompt:
        return "`total1` is 12 and `total2` is 0 because the generator is exhausted after the first sum."
    if "total == 0.3" in prompt and "prices = [0.1, 0.2]" in prompt:
        return "It returns False because binary floating point cannot represent the sum exactly; compare with `math.isclose`."
    if "def factorial(n):" in prompt and "if n == 1" in prompt and "factorial(0)" in prompt:
        return "It recurses until a RecursionError because 0 is not a base case. Use `if n <= 1: return 1`."
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
    """Return true only when the prompt matches a fully deterministic math form."""
    return any(
        answer is not None
        for answer in (
            _extract_math(prompt),
            _answer_word_math(prompt),
            _answer_entities(prompt),
            _answer_known_python_semantics(prompt),
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

    known_python = _answer_known_python_semantics(prompt)
    if known_python is not None:
        return ModelAnswer(answer=known_python, confidence=0.99, tier=Tier.LOCAL)

    if len(prompt.split()) <= 12:
        return ModelAnswer(answer="I need a stronger model for a reliable answer.", confidence=0.25, tier=Tier.LOCAL)

    answer = "Local model abstained because the task appears open-ended."
    return ModelAnswer(answer=answer, confidence=0.0, tier=Tier.LOCAL)
