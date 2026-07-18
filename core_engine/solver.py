"""
GeoIPS — Core Inference Engine (Unification-based).

Upgraded from propositional exact matching to support:
- Propositional rules (no '?' variables): same behaviour as before.
- Variable rules (containing '?' variables): unification-based matching
  via GeometryUnifier, enabling general rules like
  Congruent(?X,?Y) → Congruent(?Y,?X) to fire for any fact pair.
- Arithmetic inference: Equal(Add(X,Y,Z),180) + Equal(X,60) + Equal(Y,70)
  → derives Equal(Z,50) automatically, enabling numeric goals like
  Equal(Angle(ACB),50) to be resolved.

Both engines are domain-agnostic and operate strictly on Fact/Rule abstractions.
"""

import time
from typing import List, Set, Optional, Dict

from core_engine.models import Fact, Rule, ExecutionStep, InferenceResult
from core_engine.unifier import find_rule_bindings, instantiate_consequents, has_variables
from core_engine.arithmetic_evaluator import ArithmeticEvaluator, check_numeric_goal


class ForwardChainingEngine:
    """
    Domain-agnostic forward-chaining solver with unification support
    and integrated arithmetic evaluation.

    Algorithm:
    1. Add all initial_facts to Working Memory.
    2. Run ArithmeticEvaluator to derive numeric facts from Equal(Add(...), n).
    3. Iterate through rules. For each unfired rule:
       a. Find all valid bindings (propositional or unification-based).
       b. For each binding, instantiate consequents and add to WM.
    4. After each rule-firing pass, run ArithmeticEvaluator again.
    5. Check goal using both exact string match and numeric equivalence.
    6. Repeat until no new facts can be derived (saturation) or goal is reached.
    """

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def solve(self, initial_facts: List[Fact], goal: Optional[Fact] = None) -> InferenceResult:
        working_memory: Set[Fact] = set(initial_facts)
        applied_rule_ids: List[str] = []
        execution_trace: List[ExecutionStep] = []

        # Arithmetic evaluator runs across all iterations
        arith = ArithmeticEvaluator()

        def _wm_values() -> List[str]:
            return [f.value for f in working_memory]

        def _check_goal() -> bool:
            if not goal:
                return False
            if goal in working_memory:
                return True
            # Numeric equivalence check (e.g. Equal(Angle(ACB),50) via registry)
            return check_numeric_goal(goal.value, _wm_values(), arith.registry)

        def _run_arithmetic():
            """Derive new numeric facts and add them to WM."""
            new_strs = arith.derive_new_facts(_wm_values())
            added = False
            for s in new_strs:
                new_f = Fact(
                    id=f"arith_{abs(hash(s))}",
                    value=s,
                    domain=initial_facts[0].domain if initial_facts else "geometry",
                )
                if new_f not in working_memory:
                    working_memory.add(new_f)
                    added = True
            return added

        # Initial arithmetic pass
        _run_arithmetic()

        # Early exit: goal already satisfied
        if _check_goal():
            return InferenceResult(
                goal_reached=True,
                final_facts=list(working_memory),
                execution_trace=execution_trace,
                applied_rule_ids=applied_rule_ids,
            )

        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                bindings = find_rule_bindings(rule, working_memory)
                if not bindings:
                    continue

                for binding in bindings:
                    consequents = instantiate_consequents(rule, binding, rule.domain)
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

            # After each rule-firing pass, run arithmetic derivation
            if _run_arithmetic():
                changed = True  # New numeric facts may enable more rules
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
                # Check if any consequent can unify with subgoal
                from core_engine.unifier import unify_expressions, apply_binding
                for i, cons in enumerate(rule.consequents):
                    binding = unify_expressions(cons.value, subgoal.value)
                    if binding is None:
                        continue

                    # This rule could produce subgoal with this binding.
                    # Try to prove all antecedents under the same binding.
                    all_proved = True
                    for ant in rule.antecedents:
                        from core_engine.unifier import has_variables
                        instantiated_ant_value = apply_binding(ant.value, binding)
                        ant_fact = Fact(
                            id=f"{rule.id}_ant_inst",
                            value=instantiated_ant_value,
                            domain=rule.domain
                        )
                        if not prove(ant_fact):
                            all_proved = False
                            break

                    if all_proved:
                        # Fire rule: add all instantiated consequents
                        from core_engine.unifier import instantiate_consequents
                        new_cons = instantiate_consequents(rule, binding, rule.domain)
                        new_inferred = [f for f in new_cons if f not in working_memory]
                        working_memory.update(new_cons)

                        if rule.id not in applied_rule_ids:
                            applied_rule_ids.append(rule.id)
                            execution_trace.append(ExecutionStep(
                                rule_id=rule.id,
                                fired_rule_repr=repr(rule),
                                new_facts=new_inferred,
                                timestamp_ms=time.time() * 1000
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
            applied_rule_ids=applied_rule_ids
        )
