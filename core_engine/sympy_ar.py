"""
GeoIPS — SymPy-based Algebraic Reasoning Engine (AR).

Inspired by AlphaGeometry's AR module (Wu's Method).

Replaces the limited custom ArithmeticEvaluator with a full SymPy-powered
solver, enabling GeoIPS to reason over:
  - Pythagorean theorem:   a²+b²=c²  →  c = √(a²+b²)  (when a,b known)
  - Ratio equations:       a/b = c/d  →  a = bc/d
  - Systems of 2+ equations with multiple unknowns
  - Power equations:       Pow(x,2)=25  →  x = 5
  - Any algebraic identity over geometry expressions

Algorithm:
  1. Scan WM for all Equal(lhs, rhs) facts.
  2. Parse each as a SymPy equation (lhs_sympy = rhs_sympy).
  3. Substitute already-known numeric values to reduce unknowns.
  4. Solve the system with sympy.solve() / single-equation fallback.
  5. Return newly derived Equal(original_expr, numeric_value) strings.
"""

import re
import math
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sympy_ar")

# ---------------------------------------------------------------------------
# Lazy SymPy import
# ---------------------------------------------------------------------------

_sp = None


def _sympy():
    global _sp
    if _sp is None:
        import sympy
        _sp = sympy
    return _sp


# ---------------------------------------------------------------------------
# Expression string utilities
# ---------------------------------------------------------------------------

def _split_top_args(s: str) -> List[str]:
    """Split comma-separated args at parenthesis depth 0."""
    args: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in s:
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


def _normalize_segment(seg: str) -> str:
    """Canonicalize a 2-letter segment: AB == BA → AB (sorted)."""
    if len(seg) == 2 and seg.isalpha() and seg.isupper():
        return "".join(sorted(seg))
    return seg


def _normalize_angle(ang: str) -> str:
    """Canonicalize a 3-letter angle label: Angle(ABC) — vertex is middle letter.
    ABC and CBA are the same angle (same vertex B, same two rays BA and BC).
    Normalize to: vertex stays middle, the two end-points are sorted.
    """
    if len(ang) == 3 and ang.isalpha() and ang.isupper():
        p, v, q = ang[0], ang[1], ang[2]
        return v + min(p, q) + max(p, q)
    return ang


def _safe_sym_name(expr: str) -> str:
    """Make a SymPy-safe identifier from a geometry expression string."""
    return re.sub(r"[^A-Za-z0-9]", "_", expr).strip("_")


# ---------------------------------------------------------------------------
# SymPy AR Engine
# ---------------------------------------------------------------------------

class SymPyAREngine:
    """
    Full algebraic reasoning using SymPy.

    Scan working memory, build equations, solve them, and return new facts.
    All geometry symbols are treated as positive real numbers.
    """

    def __init__(self):
        # Maps canonical expression string → SymPy Symbol
        self._sym_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Symbol management
    # ------------------------------------------------------------------

    def _sym(self, name: str):
        """Return (or create) a positive SymPy symbol for a geometry quantity."""
        sp = _sympy()
        if name not in self._sym_cache:
            safe = _safe_sym_name(name)
            self._sym_cache[name] = sp.Symbol(safe, positive=True, real=True)
        return self._sym_cache[name]

    def _orig_name(self, sympy_sym) -> Optional[str]:
        """Reverse lookup: SymPy symbol → original geometry expression string."""
        target = str(sympy_sym)
        for orig, s in self._sym_cache.items():
            if str(s) == target:
                return orig
        return None

    # ------------------------------------------------------------------
    # Expression parser: geometry string → SymPy expression
    # ------------------------------------------------------------------

    def _parse(self, expr_str: str):
        """
        Recursively parse a geometry expression string into a SymPy expression.

        Supported functors: Add, Sub, Mul, Div, Pow, Length, Angle.
        Everything else is treated as a named symbol.
        """
        sp = _sympy()
        s = expr_str.strip().replace(" ", "")

        # ── Numeric literal ────────────────────────────────────────────
        try:
            return sp.Float(float(s))
        except (ValueError, TypeError):
            pass

        # ── Predicate: Functor(arg1,arg2,...) ─────────────────────────
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.+)\)", s)
        if m:
            functor = m.group(1)
            raw_args = _split_top_args(m.group(2))
            parsed_args = [self._parse(a) for a in raw_args]

            if functor == "Add":
                result = sp.Integer(0)
                for pa in parsed_args:
                    result = result + pa
                return result

            if functor == "Sub" and len(parsed_args) == 2:
                return parsed_args[0] - parsed_args[1]

            if functor == "Mul":
                result = sp.Integer(1)
                for pa in parsed_args:
                    result = result * pa
                return result

            if functor == "Div" and len(parsed_args) == 2:
                return parsed_args[0] / parsed_args[1]

            if functor == "Pow" and len(parsed_args) == 2:
                return parsed_args[0] ** parsed_args[1]

            if functor == "Length" and raw_args:
                seg = _normalize_segment(raw_args[0])
                return self._sym(f"Length({seg})")

            if functor == "Angle" and raw_args:
                ang = _normalize_angle(raw_args[0])
                return self._sym(f"Angle({ang})")

            # Unknown functor → named symbol
            return self._sym(s)

        # ── Plain atom: bare segment label, angle label, etc. ─────────
        # Two uppercase letters = segment
        if re.fullmatch(r"[A-Z]{2}", s):
            seg = _normalize_segment(s)
            return self._sym(f"Length({seg})")

        return self._sym(s)

    # ------------------------------------------------------------------
    # Equal pair extraction
    # ------------------------------------------------------------------

    def _extract_equal_pairs(self, wm_values: List[str]) -> List[Tuple[str, str]]:
        """Extract (lhs, rhs) from Equal(lhs,rhs) facts."""
        pairs: List[Tuple[str, str]] = []
        for v in wm_values:
            v = v.strip().replace(" ", "")
            m = re.fullmatch(r"Equal\((.+)\)", v)
            if not m:
                continue
            parts = _split_top_args(m.group(1))
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
        return pairs

    # ------------------------------------------------------------------
    # Numeric formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(val: float) -> str:
        """Format: integer if whole, else 6 significant figures."""
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        return f"{val:.6g}"

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def derive_new_facts(self, wm_values: List[str]) -> List[str]:
        """
        Derive new Equal(symbol, numeric_value) facts using SymPy.

        Runs iteratively until no new facts can be derived.
        """
        sp = _sympy()
        all_values = list(wm_values)
        new_facts: List[str] = []

        changed = True
        max_passes = 5
        pass_num = 0

        while changed and pass_num < max_passes:
            changed = False
            pass_num += 1

            # ── Step 1: Collect numeric registry ──────────────────────
            # Equal(symbol, number) → known values
            numeric_registry: Dict[str, float] = {}
            for lhs_str, rhs_str in self._extract_equal_pairs(all_values):
                try:
                    val = float(rhs_str)
                    numeric_registry[lhs_str] = val
                except (ValueError, TypeError):
                    pass
                try:
                    val = float(lhs_str)
                    numeric_registry[rhs_str] = val
                except (ValueError, TypeError):
                    pass

            # ── Step 2: Build system of SymPy equations ───────────────
            equations = []
            eq_metadata: List[Tuple] = []  # (lhs_str, rhs_str, eq_sympy)

            for lhs_str, rhs_str in self._extract_equal_pairs(all_values):
                try:
                    lhs_expr = self._parse(lhs_str)
                    rhs_expr = self._parse(rhs_str)

                    # Substitute known numeric values
                    for name, val in numeric_registry.items():
                        try:
                            name_sym = self._parse(name)
                            if hasattr(name_sym, "free_symbols") and not name_sym.free_symbols:
                                continue  # It's already a constant
                            lhs_expr = lhs_expr.subs(name_sym, val)
                            rhs_expr = rhs_expr.subs(name_sym, val)
                        except Exception:
                            pass

                    eq = sp.Eq(lhs_expr, rhs_expr)
                    equations.append(eq)
                    eq_metadata.append((lhs_str, rhs_str, eq))

                except Exception as e:
                    logger.debug("Parse error for '%s = %s': %s", lhs_str, rhs_str, e)

            if not equations:
                break

            # ── Step 3: One-equation solves (most common case) ────────
            for lhs_str, rhs_str, eq in eq_metadata:
                free = list(eq.free_symbols)
                if len(free) != 1:
                    continue
                sym = free[0]
                try:
                    sols = sp.solve(eq, sym)
                    for sol in sols:
                        try:
                            val = float(sol.evalf())
                            if val <= 0:
                                continue  # geometric lengths/angles must be positive
                            orig = self._orig_name(sym)
                            if not orig:
                                continue
                            new_fact = f"Equal({orig},{self._fmt(val)})"
                            if new_fact not in all_values:
                                all_values.append(new_fact)
                                new_facts.append(new_fact)
                                numeric_registry[orig] = val
                                changed = True
                                logger.info("[SymPyAR] Derived: %s", new_fact)
                        except (TypeError, ValueError, sp.core.numbers.NaN.__class__):
                            pass
                except Exception as e:
                    logger.debug("Single-eq solve failed for '%s': %s", eq, e)

            # ── Step 4: Multi-equation system solve ───────────────────
            all_free = set()
            for eq in equations:
                all_free.update(eq.free_symbols)

            if len(all_free) > 1 and len(equations) >= len(all_free):
                try:
                    solution = sp.solve(equations, list(all_free), dict=True)
                    sol_list = solution if isinstance(solution, list) else [solution]
                    for sol_dict in sol_list:
                        if not isinstance(sol_dict, dict):
                            continue
                        for sym, val_expr in sol_dict.items():
                            try:
                                val = float(val_expr.evalf())
                                if val <= 0:
                                    continue
                                orig = self._orig_name(sym)
                                if not orig:
                                    continue
                                new_fact = f"Equal({orig},{self._fmt(val)})"
                                if new_fact not in all_values:
                                    all_values.append(new_fact)
                                    new_facts.append(new_fact)
                                    numeric_registry[orig] = val
                                    changed = True
                                    logger.info("[SymPyAR] Multi-solve derived: %s", new_fact)
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug("Multi-eq solve failed: %s", e)

            # ── Step 5: Direct evaluation of fully-known expressions ──
            # If Equal(Pow(Length(AB),2), Add(Pow(Length(AC),2),Pow(Length(BC),2)))
            # and Length(AC) and Length(BC) are known → compute Pow result
            for lhs_str, rhs_str in self._extract_equal_pairs(all_values):
                for (target_str, val_str) in [(lhs_str, rhs_str), (rhs_str, lhs_str)]:
                    # If val_str resolves to a known number:
                    try:
                        val_expr = self._parse(val_str)
                        for name, nval in numeric_registry.items():
                            try:
                                val_expr = val_expr.subs(self._parse(name), nval)
                            except Exception:
                                pass
                        if not val_expr.free_symbols:
                            computed = float(val_expr.evalf())
                            # Now target_str might be solvable
                            t_expr = self._parse(target_str)
                            for name, nval in numeric_registry.items():
                                try:
                                    t_expr = t_expr.subs(self._parse(name), nval)
                                except Exception:
                                    pass
                            t_free = list(t_expr.free_symbols)
                            if len(t_free) == 1:
                                sym = t_free[0]
                                try:
                                    eq = sp.Eq(t_expr, computed)
                                    sols = sp.solve(eq, sym)
                                    for sol in sols:
                                        val = float(sol.evalf())
                                        if val <= 0:
                                            continue
                                        orig = self._orig_name(sym)
                                        if not orig:
                                            continue
                                        new_fact = f"Equal({orig},{self._fmt(val)})"
                                        if new_fact not in all_values:
                                            all_values.append(new_fact)
                                            new_facts.append(new_fact)
                                            numeric_registry[orig] = val
                                            changed = True
                                            logger.info("[SymPyAR] Eval-derived: %s", new_fact)
                                except Exception:
                                    pass
                    except Exception:
                        pass

        return new_facts

    def check_numeric_goal(self, goal_value: str, wm_values: List[str]) -> bool:
        """
        Check if a numeric goal is satisfied given WM and derived facts.
        Handles float formatting differences (e.g., 5 vs 5.0 vs 5.000000).
        """
        sp = _sympy()
        goal_value = goal_value.strip().replace(" ", "")

        # Direct string match
        if goal_value in wm_values:
            return True

        # Parse as Equal(symbol, number)
        m = re.fullmatch(r"Equal\((.+),(.+)\)", goal_value)
        if not m:
            return False

        lhs_str, rhs_str = m.group(1), m.group(2)
        try:
            target = float(rhs_str)
        except (ValueError, TypeError):
            return False

        # Check if any WM value resolves the same symbol to the target
        numeric_registry: Dict[str, float] = {}
        for lhs, rhs in self._extract_equal_pairs(wm_values):
            try:
                numeric_registry[lhs] = float(rhs)
            except (ValueError, TypeError):
                pass
            try:
                numeric_registry[rhs] = float(lhs)
            except (ValueError, TypeError):
                pass

        if lhs_str in numeric_registry:
            return abs(numeric_registry[lhs_str] - target) < 1e-6

        # Try canonicalized lookup
        try:
            lhs_parsed = self._parse(lhs_str)
            lhs_canon = str(lhs_parsed) if lhs_parsed.free_symbols else None
        except Exception:
            lhs_canon = None

        for orig, val in numeric_registry.items():
            if orig == lhs_str and abs(val - target) < 1e-6:
                return True
            try:
                orig_parsed = self._parse(orig)
                if (hasattr(lhs_parsed, "free_symbols") and
                        hasattr(orig_parsed, "free_symbols") and
                        not lhs_parsed.free_symbols and not orig_parsed.free_symbols):
                    if abs(float(lhs_parsed.evalf()) - float(orig_parsed.evalf())) < 1e-9:
                        if abs(val - target) < 1e-6:
                            return True
            except Exception:
                pass

        return False
