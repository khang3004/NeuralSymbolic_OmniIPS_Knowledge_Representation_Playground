import sys
import os
import traceback

# Append project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine import ForwardChainingEngine, BackwardChainingEngine
from domains.chemistry import ChemistryParser
from domains.geometry import GeometryParser
from domains.algebra import AlgebraParser

def test_chemistry_domain():
    print("\n--- Running Chemistry Domain Verification (Na + H2O + HCl -> NaCl) ---")
    parser = ChemistryParser()

    # Define mock JSON rule structures
    raw_rules = [
        {
            "id": "r1",
            "name": "Sodium Hydration",
            "inputs": ["Na", "H2O"],
            "outputs": ["NaOH", "H2"],
            "description": "Sodium reacting with water to produce sodium hydroxide and hydrogen gas."
        },
        {
            "id": "r2",
            "name": "Water Synthesis",
            "inputs": ["H2", "O2"],
            "outputs": ["H2O"],
            "description": "Hydrogen reacting with oxygen to form water."
        },
        {
            "id": "r3",
            "name": "Neutralization",
            "inputs": ["NaOH", "HCl"],
            "outputs": ["NaCl", "H2O"],
            "description": "Sodium hydroxide reacting with hydrochloric acid to produce sodium chloride and water."
        }
    ]

    rules = [parser.parse_rule(r) for r in raw_rules]
    
    # 1. Forward Chaining Check
    initial_reactants = [
        parser.parse_fact("Na", "init_0"),
        parser.parse_fact("H2O", "init_1"),
        parser.parse_fact("HCl", "init_2")
    ]
    goal_fact = parser.parse_fact("NaCl", "goal_0")

    engine_fw = ForwardChainingEngine(rules)
    result_fw = engine_fw.solve(initial_reactants, goal_fact)

    print(f"Forward Chaining - Goal Reached: {result_fw.goal_reached}")
    print(f"Forward Chaining - Applied Rules: {result_fw.applied_rule_ids}")
    print("Execution Trace Steps:")
    for step in result_fw.execution_trace:
        print(f"  [{step.rule_id}]: {step.fired_rule_repr} (New Facts: {[f.value for f in step.new_facts]})")

    assert result_fw.goal_reached is True, "Chemistry Forward Chaining goal should be reached!"

    # 2. Backward Chaining Check
    engine_bw = BackwardChainingEngine(rules)
    result_bw = engine_bw.solve(initial_reactants, goal_fact)

    print(f"\nBackward Chaining - Goal Reached: {result_bw.goal_reached}")
    print(f"Backward Chaining - Applied Rules: {result_bw.applied_rule_ids}")
    print("Execution Trace Steps:")
    for step in result_bw.execution_trace:
        print(f"  [{step.rule_id}]: {step.fired_rule_repr} (New Facts: {[f.value for f in step.new_facts]})")

    assert result_bw.goal_reached is True, "Chemistry Backward Chaining goal should be reached!"
    print("✅ Chemistry Domain Verification Passed.")


def test_geometry_domain():
    print("\n--- Running Geometry Domain Verification (Transitivity of Congruence) ---")
    parser = GeometryParser()

    raw_rules = [
        {
            "id": "t_trans",
            "name": "Congruence Transitivity",
            "inputs": ["Congruent(AB, CD)", "Congruent(CD, EF)"],
            "outputs": ["Congruent(AB, EF)"],
            "description": "If segment AB is congruent to CD, and CD is congruent to EF, then AB is congruent to EF."
        }
    ]

    rules = [parser.parse_rule(r) for r in raw_rules]
    
    # Check commutative parsing
    fact_1 = parser.parse_fact("Congruent(CD, AB)", "f1")
    fact_2 = parser.parse_fact("Congruent(CD, EF)", "f2")
    
    # Note that Congruent(CD, AB) will canonicalize to Congruent(AB, CD) due to alphabet sorting
    print(f"Commutative Fact Canonicalization check: 'Congruent(CD, AB)' -> '{fact_1.value}'")
    assert fact_1.value == "Congruent(AB,CD)", "Canonicalization of Congruent relation failed."

    initial_facts = [fact_1, fact_2]
    goal_fact = parser.parse_fact("Congruent(AB, EF)", "goal_g")

    engine_fw = ForwardChainingEngine(rules)
    result_fw = engine_fw.solve(initial_facts, goal_fact)

    print(f"Forward Chaining - Goal Reached: {result_fw.goal_reached}")
    print(f"Forward Chaining - Applied Rules: {result_fw.applied_rule_ids}")
    for step in result_fw.execution_trace:
        print(f"  [{step.rule_id}]: {step.fired_rule_repr} (New Facts: {[f.value for f in step.new_facts]})")

    assert result_fw.goal_reached is True, "Geometry Transitivity proof failed!"
    print("✅ Geometry Domain Verification Passed.")


def test_algebra_domain():
    print("\n--- Running Algebra Domain Verification (Linear Equation Solving) ---")
    parser = AlgebraParser()

    raw_rules = [
        {
            "id": "a_sub_two",
            "name": "Subtraction Property",
            "inputs": ["x+2=5", "Subtract(2, both_sides)"],
            "outputs": ["x=3"],
            "description": "Subtract 2 from both sides of the equation x+2=5 to yield x=3."
        }
    ]

    rules = [parser.parse_rule(r) for r in raw_rules]
    
    initial_facts = [
        parser.parse_fact("x + 2 = 5", "eq_1"),
        parser.parse_fact("Subtract(2, both_sides)", "op_1")
    ]
    goal_fact = parser.parse_fact("x = 3", "goal_a")

    engine_fw = ForwardChainingEngine(rules)
    result_fw = engine_fw.solve(initial_facts, goal_fact)

    print(f"Forward Chaining - Goal Reached: {result_fw.goal_reached}")
    print(f"Forward Chaining - Applied Rules: {result_fw.applied_rule_ids}")
    for step in result_fw.execution_trace:
        print(f"  [{step.rule_id}]: {step.fired_rule_repr} (New Facts: {[f.value for f in step.new_facts]})")

    assert result_fw.goal_reached is True, "Algebra deduction failed!"
    print("✅ Algebra Domain Verification Passed.")


if __name__ == "__main__":
    print("==================================================")
    print("       OMNI-IPS PHASE 1 SCAFFOLD VERIFICATION     ")
    print("==================================================")
    try:
        test_chemistry_domain()
        test_geometry_domain()
        test_algebra_domain()
        print("\n🎉 ALL PHASE 1 CORE SOLVER AND DOMAIN VERIFICATIONS PASSED SUCCESSFULLY!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
