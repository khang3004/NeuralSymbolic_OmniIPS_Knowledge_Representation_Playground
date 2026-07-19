import sys
import os
import traceback

# Append project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine import ForwardChainingEngine, BackwardChainingEngine
from domains.geometry import GeometryParser


def test_geometry_congruence_transitivity():
    print("\n--- Running Geometry Verification: Transitivity of Segment Congruence ---")
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
    
    print(f"Commutative Fact Canonicalization check: 'Congruent(CD, AB)' -> '{fact_1.value}'")
    assert fact_1.value == "Congruent(AB,CD)", "Canonicalization of Congruent relation failed."

    initial_facts = [fact_1, fact_2]
    goal_fact = parser.parse_fact("Congruent(AB, EF)", "goal_g")

    # 1. Forward Chaining
    engine_fw = ForwardChainingEngine(rules)
    result_fw = engine_fw.solve(initial_facts, goal_fact)

    print(f"Forward Chaining - Goal Reached: {result_fw.goal_reached}")
    print(f"Forward Chaining - Applied Rules: {result_fw.applied_rule_ids}")
    for step in result_fw.execution_trace:
        print(f"  [{step.rule_id}]: {step.fired_rule_repr} (New Facts: {[f.value for f in step.new_facts]})")

    assert result_fw.goal_reached is True, "Geometry Transitivity forward chaining proof failed!"

    # 2. Backward Chaining
    engine_bw = BackwardChainingEngine(rules)
    result_bw = engine_bw.solve(initial_facts, goal_fact)

    print(f"Backward Chaining - Goal Reached: {result_bw.goal_reached}")
    assert result_bw.goal_reached is True, "Geometry Transitivity backward chaining proof failed!"
    print("✅ Geometry Transitivity Verification Passed.")


def test_geometry_unification_rule():
    print("\n--- Running Geometry Verification: Variable Unification Rule ---")
    parser = GeometryParser()

    raw_rules = [
        {
            "id": "geo_congruence_symmetric_var",
            "name": "Congruence Symmetric (general)",
            "inputs": ["Congruent(?A,?B)"],
            "outputs": ["Congruent(?B,?A)"],
            "description": "Symmetry: AB ≅ CD ⇒ CD ≅ AB"
        },
        {
            "id": "geo_sas_var",
            "name": "SAS Congruence (general)",
            "inputs": [
                "Triangle(?A,?B,?C)",
                "Triangle(?D,?E,?F)",
                "Congruent(?A?B,?D?E)",
                "Equal(Angle(?B?A?C),Angle(?E?D?F))",
                "Congruent(?A?C,?D?F)"
            ],
            "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
            "description": "SAS: AB=DE, ∠BAC=∠EDF, AC=DF ⇒ △ABC≅△DEF"
        }
    ]

    rules = [parser.parse_rule(r) for r in raw_rules]

    initial_facts = [
        parser.parse_fact("Triangle(X,Y,Z)", "f1"),
        parser.parse_fact("Triangle(P,Q,R)", "f2"),
        parser.parse_fact("Congruent(XY,PQ)", "f3"),
        parser.parse_fact("Equal(Angle(YXZ),Angle(QPR))", "f4"),
        parser.parse_fact("Congruent(XZ,PR)", "f5"),
    ]
    goal_fact = parser.parse_fact("CongruentTriangles(XYZ,PQR)", "goal_sas")

    engine_fw = ForwardChainingEngine(rules)
    result_fw = engine_fw.solve(initial_facts, goal_fact)

    print(f"Forward Chaining (Unification) - Goal Reached: {result_fw.goal_reached}")
    print(f"Applied Rules: {result_fw.applied_rule_ids}")
    assert result_fw.goal_reached is True, "SAS unification proof failed!"
    print("✅ Geometry Variable Unification Rule Passed.")


if __name__ == "__main__":
    print("==================================================")
    print("   ALPHAGEOMETRY IMO GEOMETRY SCAFFOLD TEST     ")
    print("==================================================")
    try:
        test_geometry_congruence_transitivity()
        test_geometry_unification_rule()
        print("\n🎉 ALL ALPHAGEOMETRY CORE SOLVER VERIFICATIONS PASSED SUCCESSFULLY!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
