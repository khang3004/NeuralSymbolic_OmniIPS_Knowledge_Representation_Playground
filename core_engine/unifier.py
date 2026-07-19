"""
GeoIPS — Geometry Unifier.

Implements unification-based matching for geometry rules that contain
variables (prefixed with '?'). This enables general rules like:
    Congruent(?X, ?Y) → Congruent(?Y, ?X)
to fire for ANY pair of segments, not just hardcoded ones.

Backward compatible: facts/rules without '?' variables use the existing
exact propositional matching path — zero performance regression.

Design:
- Variables are strings starting with '?' (e.g., '?X', '?a', '?seg1').
- Matching is structural: predicates must have the same functor and arity.
- Binding is a dict mapping variable names to concrete string values.
- apply_binding substitutes bindings into a template string.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from core_engine.models import Fact, Rule


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_top_level_args(inner: str) -> List[str]:
    """
    Split comma-separated arguments at paren depth 0.
    e.g. "Angle(BAC),CD" → ["Angle(BAC)", "CD"]
    """
    args = []
    current: List[str] = []
    depth = 0
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


def _parse_predicate(expr: str) -> Optional[Tuple[str, List[str]]]:
    """
    Parse a predicate expression into (functor, args).
    e.g. "Congruent(AB,CD)" → ("Congruent", ["AB", "CD"])
    e.g. "BC^2=AB^2+AC^2" → None  (non-predicate, atom)
    Returns None if not a standard functor(args) form.
    """
    expr = expr.strip().replace(" ", "")
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.+)\)", expr)
    if not m:
        return None
    functor = m.group(1)
    inner = m.group(2)
    args = _split_top_level_args(inner)
    return functor, args


def is_variable(token: str) -> bool:
    """Return True if token is a variable (starts with '?')."""
    return token.startswith("?")


def has_variables(expr: str) -> bool:
    """Return True if expression contains any variable token."""
    return "?" in expr


# ---------------------------------------------------------------------------
# Core unification
# ---------------------------------------------------------------------------

def unify_expressions(pattern: str, fact: str) -> Optional[Dict[str, str]]:
    """
    Attempt to unify a pattern (possibly containing variables) with a concrete fact.

    Args:
        pattern: A predicate pattern, e.g. "Congruent(?X,?Y)"
        fact:    A concrete fact string, e.g. "Congruent(AB,CD)"

    Returns:
        A binding dict {"?X": "AB", "?Y": "CD"} if unification succeeds.
        None if the pattern and fact cannot be unified.

    Rules:
    - If pattern has no variables → use exact string equality.
    - Functors and arities must match.
    - Variable can bind to any concrete sub-expression.
    - A variable cannot bind to two different values in the same unification.
    """
    pattern = pattern.strip().replace(" ", "")
    fact = fact.strip().replace(" ", "")

    # Fast path: no variables → exact match
    if not has_variables(pattern):
        return {} if pattern == fact else None

    # Check for composite variables (e.g. "?A?B" matching "AB" or "A1B1")
    vars_found = re.findall(r"\?[A-Za-z0-9_]+", pattern)
    if len(vars_found) > 1 and "".join(vars_found) == pattern:
        fact_points = re.findall(r"[A-Z][0-9]*", fact)
        if len(vars_found) == len(fact_points):
            binding = {}
            for var, val in zip(vars_found, fact_points):
                binding[var] = val
            return binding
        return None

    # Both are atoms (no parentheses) — direct comparison or variable bind
    pat_parsed = _parse_predicate(pattern)
    fct_parsed = _parse_predicate(fact)

    if pat_parsed is None and fct_parsed is None:
        # Both atoms
        if is_variable(pattern):
            return {pattern: fact}
        # Handle partially bound composite atoms (e.g. "A?C" matching "AC", "A1?F" matching "A1C1")
        if "?" in pattern:
            prefix = pattern.split("?")[0]
            if prefix and fact.startswith(prefix):
                pattern = pattern[len(prefix):]
                fact = fact[len(prefix):]
                if is_variable(pattern):
                    return {pattern: fact}
        return {} if pattern == fact else None

    if pat_parsed is None or fct_parsed is None:
        # One is atom, other is predicate — can only match if pattern is a variable
        if is_variable(pattern):
            return {pattern: fact}
        return None

    pat_functor, pat_args = pat_parsed
    fct_functor, fct_args = fct_parsed

    if pat_functor != fct_functor:
        return None
    if len(pat_args) != len(fct_args):
        return None

    # Unify argument by argument, collecting binding
    binding: Dict[str, str] = {}
    for p_arg, f_arg in zip(pat_args, fct_args):
        sub = unify_expressions(p_arg, f_arg)
        if sub is None:
            return None
        # Check for conflicting bindings
        for var, val in sub.items():
            if var in binding and binding[var] != val:
                return None  # Conflict
            binding[var] = val

    return binding


def apply_binding(template: str, binding: Dict[str, str]) -> str:
    """
    Substitute variable occurrences in template with their bound values.

    Args:
        template: A predicate template, e.g. "Congruent(?Y,?X)"
        binding:  {"?X": "AB", "?Y": "CD"}

    Returns:
        "Congruent(CD,AB)"
    """
    result = template
    # Sort by length descending to avoid partial replacements (e.g. ?X before ?XY)
    for var, val in sorted(binding.items(), key=lambda kv: -len(kv[0])):
        result = result.replace(var, val)
    return result


# ---------------------------------------------------------------------------
# Rule-level matching
# ---------------------------------------------------------------------------

def find_rule_bindings(
    rule: Rule,
    working_memory: Set[Fact],
) -> List[Dict[str, str]]:
    """
    Find all valid variable bindings that let this rule's antecedents match
    facts in working_memory.

    For propositional rules (no variables), returns [{}] if all antecedents
    are present, or [] otherwise.

    For variable rules, tries every combination of WM facts against each
    antecedent pattern using unification, collecting consistent bindings.

    Returns:
        A list of binding dicts. Each dict is one valid way to instantiate
        the rule's variables. Empty list means the rule cannot fire.
    """
    wm_values = [f.value for f in working_memory]
    antecedent_patterns = [ant.value for ant in rule.antecedents]

    # Fast path: no variables in any antecedent → propositional check
    if not any(has_variables(p) for p in antecedent_patterns):
        wm_set = set(wm_values)
        if all(p in wm_set for p in antecedent_patterns):
            return [{}]
        return []

    # Variable rule: find all consistent bindings
    # Start with a single empty binding, expand through each antecedent
    valid_bindings: List[Dict[str, str]] = [{}]

    for pattern in antecedent_patterns:
        if not has_variables(pattern):
            # This antecedent is propositional — must exist as-is in WM
            if pattern not in set(wm_values):
                return []
            continue

        # Try to extend each existing binding with a match for this pattern
        new_valid: List[Dict[str, str]] = []
        for current_binding in valid_bindings:
            # Partially instantiate the pattern with what we know
            partially_instantiated = apply_binding(pattern, current_binding)

            # Try every WM fact as a candidate match
            pat_functor = partially_instantiated.split('(')[0]
            for candidate in wm_values:
                # Fast functor mismatch check
                if not pat_functor.startswith('?'):
                    cand_functor = candidate.split('(')[0]
                    if pat_functor != cand_functor:
                        continue
                sub = unify_expressions(partially_instantiated, candidate)
                if sub is not None:
                    # Merge sub into current_binding — check for conflicts
                    merged = dict(current_binding)
                    conflict = False
                    for var, val in sub.items():
                        if var in merged and merged[var] != val:
                            conflict = True
                            break
                        merged[var] = val
                    if not conflict:
                        new_valid.append(merged)

        if not new_valid:
            return []
        # Deduplicate and filter out bindings where distinct point variables map to the same point
        seen_bindings: List[Dict] = []
        for b in new_valid:
            point_vars = {k: v for k, v in b.items() if re.match(r"^\?[A-Z][0-9]*$", k)}
            if len(point_vars.values()) != len(set(point_vars.values())):
                continue  # Discard binding with non-distinct points
            if b not in seen_bindings:
                seen_bindings.append(b)
        valid_bindings = seen_bindings

    return valid_bindings


def instantiate_consequents(rule: Rule, binding: Dict[str, str], domain: str) -> List[Fact]:
    """
    Apply binding to a rule's consequents to produce concrete Fact objects.
    """
    facts = []
    for i, cons in enumerate(rule.consequents):
        instantiated_value = apply_binding(cons.value, binding)
        facts.append(Fact(
            id=f"{rule.id}_inst_{i}_{abs(hash(instantiated_value))}",
            value=instantiated_value,
            domain=domain,
            attributes={**cons.attributes, "instantiated_from": rule.id, "binding": str(binding)}
        ))
    return facts
