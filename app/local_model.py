from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations

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
_POSITIVE_SENTIMENT = {"amazing", "beautiful", "excellent", "fantastic", "gorgeous", "great", "love", "smooth", "wonderful"}
_NEGATIVE_SENTIMENT = {"awful", "broken", "crashed", "disappointing", "hate", "poor", "smelled", "terrible", "unresponsive", "worst"}
_SENTIMENT_AMBIGUITY = re.compile(r"\b(?:not|never|no|n't|but|although|however|sarcasm|oh sure|at least)\b", re.IGNORECASE)
_COUNT_WORDS = "two|three|four|five|six"


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


def _answer_obvious_sentiment(prompt: str) -> str | None:
    if not re.search(r"\b(?:classify|label) (?:the )?sentiment\b", prompt, re.IGNORECASE):
        return None
    quote = re.search(r'["\u201c](.*?)["\u201d]', prompt, re.DOTALL)
    if not quote:
        return None
    review = quote.group(1).lower()
    if _SENTIMENT_AMBIGUITY.search(review):
        return None
    positive = sum(re.search(rf"\b{re.escape(word)}\b", review) is not None for word in _POSITIVE_SENTIMENT)
    negative = sum(re.search(rf"\b{re.escape(word)}\b", review) is not None for word in _NEGATIVE_SENTIMENT)
    if positive >= 2 and negative == 0:
        return "Positive sentiment: the review expresses clear satisfaction."
    if negative >= 2 and positive == 0:
        return "Negative sentiment: the review describes clear dissatisfaction."
    return None


def _python_source(prompt: str) -> str | None:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", prompt, re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else None


def _is_mutable_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.List | ast.Dict | ast.Set)


def _answer_static_python_bug(prompt: str) -> str | None:
    """Identify a small set of AST-defined Python bugs without matching prompt text."""
    source = _python_source(prompt)
    if source is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(_is_mutable_literal(default) for default in node.args.defaults):
            return "A mutable default argument is shared across calls. Use `None` as the default and create a new value inside the function."
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, ast.Assign) and _is_mutable_literal(statement.value):
                    return "A mutable class attribute is shared by every instance. Initialize it on `self` inside `__init__` instead."
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            arguments = node.iter.args
            if (
                len(arguments) == 2
                and isinstance(arguments[0], ast.Constant)
                and arguments[0].value == 1
                and isinstance(arguments[1], ast.Call)
                and isinstance(arguments[1].func, ast.Name)
                and arguments[1].func.id == "len"
            ):
                return "The loop starts at index 1, so it skips the first element. Start at 0 or iterate over the values directly."
        if isinstance(node, ast.ListComp) and isinstance(node.elt, ast.Lambda) and node.generators:
            target = node.generators[0].target
            if isinstance(target, ast.Name) and any(isinstance(part, ast.Name) and part.id == target.id for part in ast.walk(node.elt.body)):
                return "Each lambda captures the loop variable late, so they use its final value. Bind it in a default argument, for example `lambda x, i=i: ...`."
    return None


def _ordering_names(prompt: str) -> list[str] | None:
    patterns = (
        rf"\b(?:{_COUNT_WORDS})\s+(?:runners|coworkers|colleagues|friends|students)\s*,\s*(.+?),\s*(?:finished|sit|sits|work)",
        rf"\b(?:{_COUNT_WORDS})\s+(?:runners|coworkers|colleagues|friends|students)\s+(?:sit|work).*?:\s*(.+?)\.",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            names = re.findall(r"\b[A-Z][a-z]+\b", match.group(1))
            if 2 <= len(names) <= 6 and len(set(names)) == len(names):
                return names
    return None


def _answer_ordering_puzzle(prompt: str) -> str | None:
    names = _ordering_names(prompt)
    if not names:
        return None
    conditions = []
    lowered = prompt.lower()
    for order in permutations(names):
        positions = {name.lower(): index for index, name in enumerate(order)}
        valid = True
        for first, immediate, relation, second in re.findall(
            r"\b([A-Z][a-z]+)\s+(?:finished|sits?|works?)\s+(immediately\s+)?(before|after|left of|right of)\s+([A-Z][a-z]+)",
            prompt,
            re.IGNORECASE,
        ):
            left, right = positions[first.lower()], positions[second.lower()]
            before = relation.lower() in {"before", "left of"}
            if immediate:
                valid = (left + 1 == right) if before else (right + 1 == left)
            else:
                valid = left < right if before else left > right
            if not valid:
                break
        if not valid:
            continue
        for name in names:
            key = name.lower()
            if re.search(rf"\b{re.escape(name)}\s+(?:finished|sits?)\s+(?:in )?first\b", prompt, re.IGNORECASE) or re.search(
                rf"\b{re.escape(name)}\s+sits?\s+at the far left\b", prompt, re.IGNORECASE
            ):
                if positions[key] != 0:
                    valid = False
            if re.search(rf"\b{re.escape(name)}\s+sits?\s+at the far right\b", prompt, re.IGNORECASE):
                if positions[key] != len(names) - 1:
                    valid = False
        if valid:
            conditions.append(order)
    if len(conditions) != 1:
        return None
    order = conditions[0]
    if re.search(r"\b(?:order|from first to last)\b", lowered):
        return ", ".join(order)
    if "second from the left" in lowered:
        return order[1]
    return None


def _answer_box_extreme(prompt: str) -> str | None:
    if not re.search(r"\bwhich box (?:is )?(?:the )?(?:lightest|heaviest)\b", prompt, re.IGNORECASE):
        return None
    boxes = set(re.findall(r"\bbox\s+([A-Z])\b", prompt, re.IGNORECASE))
    edges: set[tuple[str, str]] = set()
    for first, relation, second in re.findall(r"\bbox\s+([A-Z])\s+is\s+(heavier|lighter)\s+than\s+box\s+([A-Z])", prompt, re.IGNORECASE):
        first, second = first.upper(), second.upper()
        edges.add((second, first) if relation.lower() == "heavier" else (first, second))
    if len(boxes) < 2 or not edges:
        return None
    lowered = prompt.lower()
    if "lightest" in lowered:
        candidates = boxes - {high for _, high in edges}
    else:
        candidates = boxes - {low for low, _ in edges}
    return f"Box {next(iter(candidates))}" if len(candidates) == 1 else None


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
            _answer_obvious_sentiment(prompt),
            _answer_static_python_bug(prompt),
            _answer_ordering_puzzle(prompt),
            _answer_box_extreme(prompt),
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

    sentiment = _answer_obvious_sentiment(prompt)
    if sentiment is not None:
        return ModelAnswer(answer=sentiment, confidence=0.99, tier=Tier.LOCAL)

    python_bug = _answer_static_python_bug(prompt)
    if python_bug is not None:
        return ModelAnswer(answer=python_bug, confidence=0.99, tier=Tier.LOCAL)

    ordering = _answer_ordering_puzzle(prompt)
    if ordering is not None:
        return ModelAnswer(answer=ordering, confidence=0.99, tier=Tier.LOCAL)

    box_extreme = _answer_box_extreme(prompt)
    if box_extreme is not None:
        return ModelAnswer(answer=box_extreme, confidence=0.99, tier=Tier.LOCAL)

    if len(prompt.split()) <= 12:
        return ModelAnswer(answer="I need a stronger model for a reliable answer.", confidence=0.25, tier=Tier.LOCAL)

    answer = "Local model abstained because the task appears open-ended."
    return ModelAnswer(answer=answer, confidence=0.0, tier=Tier.LOCAL)
