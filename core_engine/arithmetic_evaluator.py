"""
GeoIPS — Arithmetic Evaluator for Geometry.

Handles numeric inference within the ForwardChainingEngine:
- Resolves Equal(Add(X,Y,Z), 180) + Equal(X,60) + Equal(Y,70) → Equal(Z,50)
- Detects when a numeric goal like Equal(Angle(ACB),50) is satisfied via
  computed facts, even if the exact string does not appear in working memory
- Evaluates arithmetic over symbolic geometry expressions
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arithmetic_evaluator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _canonical_angle(spec: str) -> str:
    """
    Reduce any angle notation to a canonical vertex-centric compact form.

    Input forms handled:
      Angle(BAC)      → 'Angle(BAC)'   vertex=A (middle of 3-char string)
      Angle(B,A,C)    → 'Angle(BAC)'   vertex=A (middle arg)
      Angle(C,A,B)    → 'Angle(BAC)'   same angle, just reversed direction

    The canonical form is always Angle(XYZ) where Y is the vertex,
    and X < Z alphabetically to remove direction ambiguity.
    """
    spec = spec.strip().replace(" ", "")

    # Already compact 3-char: Angle(BAC)
    m3 = re.fullmatch(r"Angle\(([A-Z])([A-Z])([A-Z])\)", spec)
    if m3:
        p, v, q = m3.group(1), m3.group(2), m3.group(3)
        # Canonicalize direction: smaller leg first
        if p > q:
            p, q = q, p
        return f"Angle({p}{v}{q})"

    # Comma form: Angle(B,A,C)
    mc = re.fullmatch(r"Angle\(([A-Z]),([A-Z]),([A-Z])\)", spec)
    if mc:
        p, v, q = mc.group(1), mc.group(2), mc.group(3)
        if p > q:
            p, q = q, p
        return f"Angle({p}{v}{q})"

    return spec  # Unknown form: leave unchanged


def normalize_fact_value(value: str) -> str:
    """
    Normalize all angle notations in a geometry fact value string to the
    canonical vertex-centric compact form Angle(XYZ).

    Handles both Angle(BAC) and Angle(B,A,C) forms anywhere in the string.
    """
    # Replace comma form first: Angle(X,Y,Z) → canonical
    value = re.sub(
        r"Angle\(([A-Z]),([A-Z]),([A-Z])\)",
        lambda m: _canonical_angle(f"Angle({m.group(1)},{m.group(2)},{m.group(3)})"),
        value,
    )
    # Replace compact form: Angle(XYZ) → canonical
    value = re.sub(
        r"Angle\(([A-Z]{3})\)",
        lambda m: _canonical_angle(f"Angle({m.group(1)})"),
        value,
    )
    return value


def _split_top_args(inner: str) -> List[str]:
    """Split comma-separated top-level args respecting parenthesis depth."""
    args, current, depth = [], [], 0
    for ch in inner:
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return args


def _parse_pred(expr: str) -> Optional[Tuple[str, List[str]]]:
    """Parse functor(args) → (functor, [args])."""
    expr = expr.strip().replace(" ", "")
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.+)\)", expr)
    if not m:
        return None
    return m.group(1), _split_top_args(m.group(2))


def _extract_equal_pairs(wm_values: List[str]) -> List[Tuple[str, str]]:
    """Extract all Equal(lhs, rhs) pairs from a list of fact value strings.
    Normalizes angle notation before matching."""
    pairs = []
    for v in wm_values:
        v2 = normalize_fact_value(v.strip().replace(" ", ""))
        m = re.fullmatch(r"Equal\((.+),([^,()]+)\)", v2)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


# ---------------------------------------------------------------------------
# Numeric Registry
# ---------------------------------------------------------------------------

class NumericRegistry:
    """Maps symbolic expressions to their known numeric float values."""

    def __init__(self):
        self._values: Dict[str, float] = {}

    def register(self, symbol: str, value: float) -> bool:
        """Register value. Returns True if new."""
        symbol = symbol.strip().replace(" ", "")
        if symbol not in self._values:
            self._values[symbol] = value
            return True
        return False

    def get(self, symbol: str) -> Optional[float]:
        symbol = symbol.strip().replace(" ", "")
        if _is_numeric(symbol):
            return float(symbol)
        return self._values.get(symbol)

    def known(self) -> Set[str]:
        return set(self._values.keys())


# ---------------------------------------------------------------------------
# Core Evaluator
# ---------------------------------------------------------------------------

class ArithmeticEvaluator:
    """
    Augments ForwardChainingEngine with numeric/algebraic inference.

    Call `derive_new_facts(wm_values)` after each forward-chaining iteration
    to produce additional Equal(X, n) facts derivable from arithmetic rules.
    """

    def __init__(self):
        self.registry = NumericRegistry()

    # ------------------------------------------------------------------
    # Internal resolvers
    # ------------------------------------------------------------------

    def _solve_for_missing(
        self, functor: str, args: List[str], total: float
    ) -> Optional[Tuple[str, float]]:
        """
        Given Equal(Add(A,B,...,?UNK,...), total) where exactly one arg is
        unknown, solve for the unknown symbol.

        Examples:
          Add(60, 70, Angle(ACB)) = 180  →  Angle(ACB) = 50
          Sub(180, Angle(BAC), Angle(ABC)) = Angle(ACB) not handled here
          Add(Angle(BAC),Angle(ABC),Angle(ACB)) = 180, Angle(BAC)=60, Angle(ABC)=70
          → Angle(ACB) = 50
        """
        if functor not in ("Add", "Sub"):
            return None

        known_sum = 0.0
        unknown_sym = None

        for i, a in enumerate(args):
            val = self.registry.get(a)
            if val is not None:
                if functor == "Add":
                    known_sum += val
                elif functor == "Sub":
                    known_sum = val if i == 0 else known_sum - val
            else:
                if unknown_sym is not None:
                    return None  # More than one unknown
                unknown_sym = a

        if unknown_sym is None:
            return None  # All known → nothing to solve for

        if functor == "Add":
            missing_val = total - known_sum
        elif functor == "Sub":
            # Edge: unknown is the minuend (i==0), rest are subtrahends
            first_is_unknown = (args[0] == unknown_sym)
            if first_is_unknown:
                sub_sum = sum(
                    self.registry.get(a) or 0
                    for a in args[1:]
                    if self.registry.get(a) is not None
                )
                missing_val = total + sub_sum
            else:
                minuend = self.registry.get(args[0])
                if minuend is None:
                    return None
                missing_val = minuend - total
        else:
            return None

        return unknown_sym, missing_val

    def _try_full_eval(self, functor: str, args: List[str]) -> Optional[float]:
        """Evaluate functor(args) if all args are known numerically."""
        numeric_args = []
        for a in args:
            val = self.registry.get(a)
            if val is None:
                return None
            numeric_args.append(val)

        if functor == "Add":
            return sum(numeric_args)
        elif functor == "Sub" and len(numeric_args) == 2:
            return numeric_args[0] - numeric_args[1]
        elif functor == "Mul":
            r = 1.0
            for n in numeric_args:
                r *= n
            return r
        elif functor == "Div" and len(numeric_args) == 2 and numeric_args[1] != 0:
            return numeric_args[0] / numeric_args[1]
        elif functor == "Pow" and len(numeric_args) == 2:
            return numeric_args[0] ** numeric_args[1]
        return None

    @staticmethod
    def _fmt(val: float) -> str:
        """Format a float as int string if whole, else decimal."""
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def derive_new_facts(self, wm_values: List[str]) -> List[str]:
        """
        Derive new arithmetic facts from current working memory.
        Returns a list of new fact value strings like 'Equal(Angle(ACB),50)'.
        """
        new_facts: List[str] = []
        all_values = list(wm_values)

        # Iterative fixpoint: keep deriving until nothing new is found
        changed = True
        while changed:
            changed = False

            # Step 1: Populate registry from Equal(X, numeric) in all values
            for lhs, rhs in _extract_equal_pairs(all_values):
                if _is_numeric(rhs):
                    if self.registry.register(lhs, float(rhs)):
                        changed = True

            # Step 2: For each Equal(Func(...), n), try to derive unknowns
            for lhs, rhs in _extract_equal_pairs(all_values):
                if not _is_numeric(rhs):
                    continue
                total = float(rhs)

                parsed = _parse_pred(lhs)
                if not parsed:
                    continue
                functor, args = parsed

                # 2a: Solve for one missing variable
                result = self._solve_for_missing(functor, args, total)
                if result:
                    sym, val = result
                    new_fact = f"Equal({sym},{self._fmt(val)})"
                    if new_fact not in all_values:
                        if self.registry.register(sym, val):
                            new_facts.append(new_fact)
                            all_values.append(new_fact)
                            logger.info(
                                "[ArithEval] Derived %s  (from %s=%s via %s)",
                                new_fact, lhs, rhs, functor,
                            )
                            changed = True

                # 2b: Evaluate fully-known compound expression
                full_val = self._try_full_eval(functor, args)
                if full_val is not None:
                    new_fact = f"Equal({lhs},{self._fmt(full_val)})"
                    if new_fact not in all_values:
                        new_facts.append(new_fact)
                        all_values.append(new_fact)
                        changed = True

        return new_facts


# ---------------------------------------------------------------------------
# Numeric Goal Checker
# ---------------------------------------------------------------------------

def check_numeric_goal(
    goal_value: str,
    wm_values: List[str],
    registry: NumericRegistry,
) -> bool:
    """
    Check whether a numeric goal like Equal(Angle(ACB),50) is satisfied,
    using both direct string matching and registry lookup.

    This handles cases where the WM contains 'Equal(Angle(ACB),50.0)' but
    the goal string is 'Equal(Angle(ACB),50)'.
    """
    goal_value = goal_value.strip().replace(" ", "")

    # 1. Direct string match
    if goal_value in [v.strip().replace(" ", "") for v in wm_values]:
        return True

    # 2. Parse as Equal(symbol, n) and check registry
    m = re.fullmatch(r"Equal\((.+),([^,()]+)\)", goal_value)
    if not m:
        return False

    symbol = m.group(1).strip()
    target_str = m.group(2).strip()

    if not _is_numeric(target_str):
        return False

    target = float(target_str)
    known = registry.get(symbol)
    if known is not None and abs(known - target) < 1e-9:
        return True

    return False
