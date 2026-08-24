"""
GeoIPS — Core Inference Engine (AlphaGeometry-inspired DD + AR + Coord).

Upgraded to a three-layer architecture:

  Layer 1 — DD (Deductive Database):
    Forward/Backward chaining with unification-based rule matching.
    Fires when new symbolic facts can be derived from rules.

  Layer 2 — AR (Algebraic Reasoning) with SymPy:
    After each DD pass, SymPyAREngine solves the current equation system.
    Handles: Pythagorean, ratio, power, law of cosines, etc.
    This parallels AlphaGeometry's Wu's Method / AR module.

  Layer 3 — Coord (Coordinate Geometry):
    Numerical fallback. Places points in 2D and computes unknowns directly.
    Handles: right triangle, law of sines/cosines, angle sum.

  Layer 4 — Compute Goal Support:
    Goals of the form Compute(Length(AB)) or Compute(Angle(ABC)) are
    resolved by checking the numeric registry after all passes.

Both engines are domain-agnostic and operate on Fact/Rule abstractions.
"""

import re
import time
from typing import List, Set, Optional, Dict

from core_engine.models import Fact, Rule, ExecutionStep, InferenceResult
from core_engine.unifier import find_rule_bindings, instantiate_consequents, has_variables
from core_engine.arithmetic_evaluator import ArithmeticEvaluator, check_numeric_goal
from core_engine.sympy_ar import SymPyAREngine
from core_engine.coord_engine import CoordinateEngine


class ForwardChainingEngine:
    """
    AlphaGeometry-inspired forward-chaining solver.

    Architecture: DD saturation → SymPy AR → Coordinate Geometry fallback.

    After each DD pass, both AR engines run to derive numeric facts,
    which in turn may enable more DD rules to fire — creating a
    tight feedback loop between symbolic and numeric reasoning.
    """

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def solve(
        self,
        initial_facts: List[Fact],
        goal: Optional[Fact] = None,
        max_iterations: int = 15,
        max_facts: int = 800,
    ) -> InferenceResult:
        working_memory: Set[Fact] = set(initial_facts)
        applied_rule_ids: List[str] = []
        execution_trace: List[ExecutionStep] = []

        # ── Engine instances ──────────────────────────────────────────
        arith = ArithmeticEvaluator()          # Layer 1 arithmetic (fast, handles Add/Sub)
        sympy_ar = SymPyAREngine()              # Layer 2 SymPy algebra
        coord = CoordinateEngine()             # Layer 3 coordinate geometry
        from core_engine.algebraic_engine import AlgebraicReasoningEngine
        legacy_ar = AlgebraicReasoningEngine() # Legacy angle-sum solver (kept for compatibility)

        def _wm_values() -> List[str]:
            return [f.value for f in working_memory]

        # ── Goal alias & check ────────────────────────────────────────

        def _canonicalize_geometry_str(s: str) -> str:
            """Canonicalize segment names and angle endpoints inside predicates."""
            s = s.strip().replace(" ", "")
            # Canonicalize Length(BA) -> Length(AB)
            s = re.sub(r"Length\(([A-Z])([A-Z])\)", lambda m: f"Length({''.join(sorted([m.group(1), m.group(2)]))})", s)
            # Canonicalize Segment(BA) -> Segment(AB)
            s = re.sub(r"Segment\(([A-Z])([A-Z])\)", lambda m: f"Segment({''.join(sorted([m.group(1), m.group(2)]))})", s)
            # Canonicalize Angle(CBA) -> Angle(ABC) (middle letter is vertex)
            s = re.sub(r"Angle\(([A-Z])([A-Z])([A-Z])\)", lambda m: f"Angle({min(m.group(1), m.group(3))}{m.group(2)}{max(m.group(1), m.group(3))})", s)
            # Canonicalize Perpendicular(BA,DC) -> Perpendicular(AB,CD)
            s = re.sub(r"Perpendicular\(([A-Z]{2}),([A-Z]{2})\)", lambda m: f"Perpendicular({''.join(sorted(m.group(1)))},{''.join(sorted(m.group(2)))})", s)
            # Canonicalize Parallel(BA,DC) -> Parallel(AB,CD)
            s = re.sub(r"Parallel\(([A-Z]{2}),([A-Z]{2})\)", lambda m: f"Parallel({''.join(sorted(m.group(1)))},{''.join(sorted(m.group(2)))})", s)
            # Canonicalize Congruent(BA,DC) -> Congruent(AB,CD)
            s = re.sub(r"Congruent\(([A-Z]{2}),([A-Z]{2})\)", lambda m: f"Congruent({''.join(sorted(m.group(1)))},{''.join(sorted(m.group(2)))})", s)
            return s

        def _split_and_goals(s: str) -> list:
            """Split And(G1, G2, ...) handling nested parentheses."""
            s = s.strip().replace(" ", "")
            if not (s.startswith("And(") and s.endswith(")")):
                return [s]
            inner = s[4:-1].strip()
            sub_goals = []
            current = []
            depth = 0
            for char in inner:
                if char == "(":
                    depth += 1
                    current.append(char)
                elif char == ")":
                    depth -= 1
                    current.append(char)
                elif char == "," and depth == 0:
                    sub_goals.append("".join(current).strip())
                    current = []
                else:
                    current.append(char)
            if current:
                sub_goals.append("".join(current).strip())
            return sub_goals

        def _goal_aliases(goal_value: str) -> list:
            """Return semantically equivalent goal forms."""
            aliases = [goal_value]
            # CongruentTriangles ↔ Congruent for 3-char args
            m = re.match(r"CongruentTriangles\(([A-Z]{3}),([A-Z]{3})\)", goal_value)
            if m:
                aliases.append(f"Congruent({m.group(1)},{m.group(2)})")
            m2 = re.match(r"Congruent\(([A-Z]{3}),([A-Z]{3})\)", goal_value)
            if m2:
                aliases.append(f"CongruentTriangles({m2.group(1)},{m2.group(2)})")
            # SimilarTriangles ↔ Similar
            m3 = re.match(r"SimilarTriangles\(([A-Z]{3}),([A-Z]{3})\)", goal_value)
            if m3:
                aliases.append(f"Similar({m3.group(1)},{m3.group(2)})")
            # Equal(Length(AB),n) ↔ Equal(Length(BA),n) 
            m4 = re.match(r"Equal\(Length\(([A-Z]{2})\),([\d.]+)\)", goal_value)
            if m4:
                seg = m4.group(1)
                seg_rev = seg[::-1]
                if seg_rev != seg:
                    aliases.append(f"Equal(Length({seg_rev}),{m4.group(2)})")
            # CyclicQuadrilateral / Concyclic permutations (rotations & reflections)
            m_cyc = re.match(r"(?:CyclicQuadrilateral|Concyclic)\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)", goal_value)
            if m_cyc:
                pts = list(m_cyc.groups())
                for i in range(4):
                    rot = pts[i:] + pts[:i]
                    aliases.append(f"CyclicQuadrilateral({rot[0]},{rot[1]},{rot[2]},{rot[3]})")
                    aliases.append(f"Concyclic({rot[0]},{rot[1]},{rot[2]},{rot[3]})")
                    rev = rot[::-1]
                    aliases.append(f"CyclicQuadrilateral({rev[0]},{rev[1]},{rev[2]},{rev[3]})")
                    aliases.append(f"Concyclic({rev[0]},{rev[1]},{rev[2]},{rev[3]})")
            return list(set(aliases))

        def _check_single_subgoal(sub_gval: str) -> bool:
            # ── Compute(X) goal: resolved if X has a numeric value ────
            m_compute = re.fullmatch(r"Compute\((.+)\)", sub_gval)
            if m_compute:
                inner = m_compute.group(1)
                for wm_val in _wm_values():
                    wm_val = wm_val.strip().replace(" ", "")
                    if re.fullmatch(rf"Equal\({re.escape(inner)},[\d.]+\)", wm_val):
                        return True
                return sympy_ar.check_numeric_goal(f"Equal({inner},0)", _wm_values()) is not None

            canon_sub = _canonicalize_geometry_str(sub_gval)
            canon_wm = {_canonicalize_geometry_str(f.value) for f in working_memory}
            if canon_sub in canon_wm:
                return True

            for alias in _goal_aliases(sub_gval):
                if alias in {f.value for f in working_memory} or _canonicalize_geometry_str(alias) in canon_wm:
                    return True

            if check_numeric_goal(sub_gval, _wm_values(), arith.registry):
                return True

            if sympy_ar.check_numeric_goal(sub_gval, _wm_values()):
                return True

            return False

        def _check_goal() -> bool:
            if not goal:
                return False

            gval = goal.value.strip().replace(" ", "")
            sub_goals = _split_and_goals(gval)
            return all(_check_single_subgoal(sg) for sg in sub_goals)

        def _add_new_facts(new_strs: List[str], source_rule_id: str, source_repr: str) -> bool:
            """Add new fact strings to WM, record in trace. Returns True if any new."""
            added = []
            for s in new_strs:
                new_f = Fact(
                    id=f"{source_rule_id}_{abs(hash(s))}",
                    value=s,
                    domain=initial_facts[0].domain if initial_facts else "geometry",
                )
                if new_f not in working_memory:
                    working_memory.add(new_f)
                    added.append(new_f)

            if added:
                if source_rule_id not in applied_rule_ids:
                    applied_rule_ids.append(source_rule_id)
                execution_trace.append(ExecutionStep(
                    rule_id=source_rule_id,
                    fired_rule_repr=source_repr,
                    new_facts=added,
                    timestamp_ms=time.time() * 1000,
                ))
                return True
            return False

        def _run_all_ar() -> bool:
            """Run all 4 AR layers (arithmetic, legacy, SymPy, coord). Returns True if any new facts."""
            changed = False

            # Layer 1: Legacy arithmetic (Add/Sub chains)
            new_arith = arith.derive_new_facts(_wm_values())
            if _add_new_facts(new_arith, "arithmetic_evaluation",
                              "Arithmetic: numeric equation solving (Add/Sub)"):
                changed = True

            # Layer 2: Legacy algebraic (angle sum linear system)
            ar_facts_obj = legacy_ar.derive_algebraic_facts(working_memory)
            new_ar = [f.value for f in ar_facts_obj]
            if _add_new_facts(new_ar, "algebraic_reasoning",
                              "Algebraic Reasoning: angle/ratio linear elimination"):
                changed = True

            # Layer 3: SymPy full algebraic solver
            new_sympy = sympy_ar.derive_new_facts(_wm_values())
            if _add_new_facts(new_sympy, "sympy_ar",
                              "SymPy AR: full algebraic equation system (Pythagorean, powers, ratios)"):
                changed = True

            # Layer 4: Coordinate geometry engine
            new_coord = coord.derive_new_facts(_wm_values())
            if _add_new_facts(new_coord, "coord_engine",
                              "Coord Engine: numerical geometry (law of sines/cosines, Pythagorean)"):
                changed = True

            return changed

        # ── Initial AR pass ───────────────────────────────────────────
        _run_all_ar()

        if _check_goal():
            return InferenceResult(
                goal_reached=True,
                final_facts=list(working_memory),
                execution_trace=execution_trace,
                applied_rule_ids=applied_rule_ids,
            )

        # ── Main DD + AR loop ─────────────────────────────────────────
        changed = True
        iteration = 0
        while changed and iteration < max_iterations and len(working_memory) < max_facts:
            iteration += 1
            changed = False

            for rule in self.rules:
                bindings = find_rule_bindings(rule, working_memory)
                if not bindings:
                    continue

                for binding in bindings:
                    consequents = instantiate_consequents(rule, binding, rule.domain)
                    # Filter still-unbound variable facts
                    consequents = [f for f in consequents if "?" not in f.value]
                    new_inferred = [f for f in consequents if f not in working_memory]

                    if new_inferred:
                        working_memory.update(consequents)
                        changed = True

                        if rule.id not in applied_rule_ids:
                            applied_rule_ids.append(rule.id)
                            execution_trace.append(ExecutionStep(
                                rule_id=rule.id,
                                fired_rule_repr=repr(rule),
                                new_facts=new_inferred,
                                timestamp_ms=time.time() * 1000,
                            ))

                        if _check_goal():
                            return InferenceResult(
                                goal_reached=True,
                                final_facts=list(working_memory),
                                execution_trace=execution_trace,
                                applied_rule_ids=applied_rule_ids,
                            )

            # After each full rule pass: run all AR layers
            if _run_all_ar():
                changed = True
                if _check_goal():
                    return InferenceResult(
                        goal_reached=True,
                        final_facts=list(working_memory),
                        execution_trace=execution_trace,
                        applied_rule_ids=applied_rule_ids,
                    )

        # ── Final AR flush after saturation ──────────────────────────
        # Run AR once more after DD is exhausted — sometimes SymPy needs
        # all DD facts to be present before it can solve the system
        _run_all_ar()
        if _check_goal():
            return InferenceResult(
                goal_reached=True,
                final_facts=list(working_memory),
                execution_trace=execution_trace,
                applied_rule_ids=applied_rule_ids,
            )

        return InferenceResult(
            goal_reached=False if goal else None,
            final_facts=list(working_memory),
            execution_trace=execution_trace,
            applied_rule_ids=applied_rule_ids,
        )


class BackwardChainingEngine:
    """
    Domain-agnostic backward-chaining solver with unification support.

    Algorithm:
    1. To prove a goal fact, find all rules whose consequents can be unified
       with the goal (possibly with variable bindings).
    2. Recursively attempt to prove each antecedent of matching rules.
    3. If all antecedents are provable, the rule fires and the goal is proved.
    """

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def solve(self, initial_facts: List[Fact], goal: Fact) -> InferenceResult:
        working_memory: Set[Fact] = set(initial_facts)
        applied_rule_ids: List[str] = []
        execution_trace: List[ExecutionStep] = []
        visited_subgoals: Set[str] = set()

        def prove(subgoal: Fact) -> bool:
            # Base case 1: already in WM
            if subgoal in working_memory:
                return True
            # Base case 2: cycle guard
            if subgoal.value in visited_subgoals:
                return False

            visited_subgoals.add(subgoal.value)

            # Find rules that can produce the subgoal (exact or via unification)
            for rule in self.rules:
                from core_engine.unifier import unify_expressions, apply_binding
                for i, cons in enumerate(rule.consequents):
                    binding = unify_expressions(cons.value, subgoal.value)
                    if binding is None:
                        continue

                    all_proved = True
                    for ant in rule.antecedents:
                        instantiated_ant_value = apply_binding(ant.value, binding)
                        ant_fact = Fact(
                            id=f"{rule.id}_ant_inst",
                            value=instantiated_ant_value,
                            domain=rule.domain,
                        )
                        if not prove(ant_fact):
                            all_proved = False
                            break

                    if all_proved:
                        from core_engine.unifier import instantiate_consequents
                        new_cons = instantiate_consequents(rule, binding, rule.domain)
                        new_cons = [f for f in new_cons if "?" not in f.value]
                        new_inferred = [f for f in new_cons if f not in working_memory]
                        working_memory.update(new_cons)

                        if rule.id not in applied_rule_ids:
                            applied_rule_ids.append(rule.id)
                            execution_trace.append(ExecutionStep(
                                rule_id=rule.id,
                                fired_rule_repr=repr(rule),
                                new_facts=new_inferred,
                                timestamp_ms=time.time() * 1000,
                            ))

                        visited_subgoals.discard(subgoal.value)
                        return True

            visited_subgoals.discard(subgoal.value)
            return False

        goal_reached = prove(goal)
        return InferenceResult(
            goal_reached=goal_reached,
            final_facts=list(working_memory),
            execution_trace=execution_trace,
            applied_rule_ids=applied_rule_ids,
        )
