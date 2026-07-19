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


def split_top_level(text: str) -> List[str]:
    args = []
    current = []
    depth = 0
    for ch in text:
        if ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            current.append(ch)
    if current:
        args.append(''.join(current).strip())
    return args


class ExprParser:
    def __init__(self, text):
        self.text = text.replace(' ', '')
        self.pos = 0

    def peek(self):
        if self.pos < len(self.text):
            return self.text[self.pos]
        return None

    def get_char(self):
        ch = self.peek()
        if ch:
            self.pos += 1
        return ch

    def match(self, expected):
        if self.peek() == expected:
            self.pos += 1
            return True
        return False

    def parse_number_or_word(self):
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] in '_?^'):
            self.pos += 1
        return self.text[start:self.pos]

    def parse_list(self, is_algebraic):
        self.match('(')
        args = []
        while True:
            args.append(self.parse_equality(is_algebraic))
            if self.match(','):
                continue
            if self.match(')'):
                break
            break
        return args

    def parse_base(self, is_algebraic):
        ch = self.peek()
        if ch == '(':
            self.get_char()
            val = self.parse_equality(is_algebraic)
            self.match(')')
            return val

        word = self.parse_number_or_word()
        
        if self.peek() == '(':
            child_algebraic = is_algebraic or (word in ('Mul', 'Add', 'Pow', 'Div', 'Sub'))
            if word == 'Length':
                child_algebraic = False
            
            args = self.parse_list(child_algebraic)
            return self.normalize_functor(word, args, is_algebraic)
        
        return self.normalize_leaf(word, is_algebraic)

    def normalize_functor(self, name, args, is_algebraic):
        name = name.strip()
        
        if name == 'Segment' and len(args) == 2:
            return ''.join(sorted(args))
        
        if name == 'Length' and len(args) == 1:
            arg = args[0]
            if len(arg) == 2 and arg.isalpha():
                arg = ''.join(sorted(arg))
            return f"Length({arg})"
        
        if name == 'Angle':
            if len(args) == 1 and len(args[0]) == 3:
                p, v, q = args[0][0], args[0][1], args[0][2]
                if p > q:
                    p, q = q, p
                return f"Angle({p}{v}{q})"
            elif len(args) == 3:
                p, v, q = args[0], args[1], args[2]
                if p > q:
                    p, q = q, p
                return f"Angle({p}{v}{q})"
            return f"Angle({','.join(args)})"

        if name == 'Equal' and len(args) == 2:
            lhs, rhs = args[0], args[1]
            if not is_algebraic:
                lhs_is_alg = 'Length(' in lhs or 'Mul(' in lhs or 'Add(' in lhs or 'Pow(' in lhs or 'Div(' in lhs or 'Sub(' in lhs or any(c.isdigit() for c in lhs)
                rhs_is_alg = 'Length(' in rhs or 'Mul(' in rhs or 'Add(' in rhs or 'Pow(' in rhs or 'Div(' in rhs or 'Sub(' in rhs or any(c.isdigit() for c in rhs)
                if lhs_is_alg or rhs_is_alg:
                    lhs = canonicalize(lhs, True)
                    rhs = canonicalize(rhs, True)
            if lhs > rhs:
                lhs, rhs = rhs, lhs
            return f"Equal({lhs},{rhs})"

        commutative_relations = {
            "Congruent", "Similar", "Parallel", "Intersect", 
            "CongruentTriangles", "SimilarTriangles",
            "Concurrent", "Collinear"
        }
        if name in commutative_relations:
            return f"{name}({','.join(sorted(args))})"

        if name == 'Mul':
            flat_args = []
            for arg in args:
                if arg.startswith('Mul(') and arg.endswith(')'):
                    flat_args.extend(split_top_level(arg[4:-1]))
                else:
                    flat_args.append(arg)
            return f"Mul({','.join(sorted(flat_args))})"

        if name == 'Add':
            flat_args = []
            for arg in args:
                if arg.startswith('Add(') and arg.endswith(')'):
                    flat_args.extend(split_top_level(arg[4:-1]))
                else:
                    flat_args.append(arg)
            return f"Add({','.join(sorted(flat_args))})"

        return f"{name}({','.join(args)})"

    def normalize_leaf(self, word, is_algebraic):
        if not word:
            return ""
        
        if len(word) == 2 and word.isupper() and word.isalpha():
            canonical_segment = ''.join(sorted(word))
            if is_algebraic:
                return f"Length({canonical_segment})"
            return canonical_segment
        
        if '^' in word:
            parts = word.split('^')
            if len(parts) == 2:
                base = self.normalize_leaf(parts[0], is_algebraic)
                exponent = self.normalize_leaf(parts[1], is_algebraic)
                return f"Pow({base},{exponent})"

        return word

    def parse_power(self, is_algebraic):
        left = self.parse_base(is_algebraic)
        if self.match('^'):
            left = self.normalize_to_algebraic(left)
            right = self.parse_power(True)
            return f"Pow({left},{right})"
        return left

    def parse_term(self, is_algebraic):
        left = self.parse_power(is_algebraic)
        while True:
            if self.match('*'):
                left = self.normalize_to_algebraic(left)
                right = self.parse_power(True)
                args = []
                for x in [left, right]:
                    if x.startswith('Mul(') and x.endswith(')'):
                        args.extend(split_top_level(x[4:-1]))
                    else:
                        args.append(x)
                left = f"Mul({','.join(sorted(args))})"
            elif self.match('/'):
                left = self.normalize_to_algebraic(left)
                right = self.parse_power(True)
                left = f"Div({left},{right})"
            else:
                break
        return left

    def parse_expr(self, is_algebraic):
        left = self.parse_term(is_algebraic)
        while True:
            if self.match('+'):
                left = self.normalize_to_algebraic(left)
                right = self.parse_term(True)
                args = []
                for x in [left, right]:
                    if x.startswith('Add(') and x.endswith(')'):
                        args.extend(split_top_level(x[4:-1]))
                    else:
                        args.append(x)
                left = f"Add({','.join(sorted(args))})"
            elif self.match('-'):
                left = self.normalize_to_algebraic(left)
                right = self.parse_term(True)
                left = f"Sub({left},{right})"
            else:
                break
        return left

    def parse_equality(self, is_algebraic):
        left = self.parse_expr(is_algebraic)
        if self.match('='):
            left = self.normalize_to_algebraic(left)
            right = self.parse_equality(True)
            return f"Equal({left},{right})"
        return left

    def normalize_to_algebraic(self, val):
        if len(val) == 2 and val.isupper() and val.isalpha():
            return f"Length({''.join(sorted(val))})"
        return val


def canonicalize(text: str, is_algebraic: bool = False) -> str:
    try:
        parser = ExprParser(text)
        return parser.parse_equality(is_algebraic)
    except Exception as e:
        logger.warning("ExprParser failed to parse '%s': %s", text, e)
        return text


def normalize_fact_value(value: str) -> str:
    """
    Canonicalize a geometry or arithmetic fact string.
    Uses ExprParser to normalize all angles, segments, commutative relations,
    and algebraic equations consistently.
    """
    return canonicalize(value)



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
    Handles both Equal(expr, val) and Equal(val, expr) robustly."""
    pairs = []
    for v in wm_values:
        v2 = normalize_fact_value(v.strip().replace(" ", ""))
        parsed = _parse_pred(v2)
        if parsed and parsed[0] == "Equal" and len(parsed[1]) == 2:
            arg1, arg2 = parsed[1][0].strip(), parsed[1][1].strip()
            if _is_numeric(arg1) and not _is_numeric(arg2):
                pairs.append((arg2, arg1))
            else:
                pairs.append((arg1, arg2))
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
