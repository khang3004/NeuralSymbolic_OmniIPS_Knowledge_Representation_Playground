"""
GeoIPS — Variable Rules Seed.

General (universally quantified) geometry rules using '?' variables.
These are ingested alongside the propositional rules in ingest_geometry.py.

Variable rules enable the solver to derive conclusions for ANY matching
fact pair, not just hardcoded instances. Example:
  Congruent(?X,?Y) → Congruent(?Y,?X)
fires for Congruent(AB,CD) → Congruent(CD,AB), and also
          Congruent(PQ,RS) → Congruent(RS,PQ), etc.
"""

VARIABLE_GEOMETRY_RULES = [
    # =========================================================================
    # A. General Congruence Properties
    # =========================================================================
    {
        "id": "geo_var_congruence_symmetric",
        "name": "Congruence Symmetry (General)",
        "inputs": ["Congruent(?X,?Y)"],
        "outputs": ["Congruent(?Y,?X)"],
        "description": "For any segments X, Y: if X ≅ Y then Y ≅ X. (Variable rule)"
    },
    {
        "id": "geo_var_congruence_transitive",
        "name": "Congruence Transitivity (General)",
        "inputs": ["Congruent(?X,?Y)", "Congruent(?Y,?Z)"],
        "outputs": ["Congruent(?X,?Z)"],
        "description": "For any X, Y, Z: if X≅Y and Y≅Z then X≅Z. (Variable rule)"
    },
    {
        "id": "geo_var_congruence_reflexive",
        "name": "Congruence Reflexivity (General)",
        "inputs": ["Segment(?X)"],
        "outputs": ["Congruent(?X,?X)"],
        "description": "Any segment is congruent to itself. (Variable rule)"
    },
    # =========================================================================
    # B. General Angle Equality Properties
    # =========================================================================
    {
        "id": "geo_var_angle_equal_symmetric",
        "name": "Angle Equality Symmetry (General)",
        "inputs": ["Equal(Angle(?A),Angle(?B))"],
        "outputs": ["Equal(Angle(?B),Angle(?A))"],
        "description": "If ∠A = ∠B then ∠B = ∠A. (Variable rule)"
    },
    {
        "id": "geo_var_angle_equal_transitive",
        "name": "Angle Equality Transitivity (General)",
        "inputs": ["Equal(Angle(?A),Angle(?B))", "Equal(Angle(?B),Angle(?C))"],
        "outputs": ["Equal(Angle(?A),Angle(?C))"],
        "description": "If ∠A=∠B and ∠B=∠C then ∠A=∠C. (Variable rule)"
    },
    # =========================================================================
    # C. General Parallel Line Properties
    # =========================================================================
    {
        "id": "geo_var_parallel_symmetric",
        "name": "Parallel Symmetry (General)",
        "inputs": ["Parallel(?a,?b)"],
        "outputs": ["Parallel(?b,?a)"],
        "description": "If line a ∥ b then b ∥ a. (Variable rule)"
    },
    {
        "id": "geo_var_parallel_transitive",
        "name": "Parallel Transitivity (General)",
        "inputs": ["Parallel(?a,?b)", "Parallel(?b,?c)"],
        "outputs": ["Parallel(?a,?c)"],
        "description": "If a∥b and b∥c then a∥c. (Variable rule)"
    },
    # =========================================================================
    # D. General Perpendicularity
    # =========================================================================
    {
        "id": "geo_var_perp_symmetric",
        "name": "Perpendicular Symmetry (General)",
        "inputs": ["Perpendicular(?A,?B)"],
        "outputs": ["Perpendicular(?B,?A)"],
        "description": "If AB ⊥ CD then CD ⊥ AB. (Variable rule)"
    },
    {
        "id": "geo_var_perp_to_parallel",
        "name": "Perpendicular to Parallel (General)",
        "inputs": ["Perpendicular(?L,?a)", "Parallel(?a,?b)"],
        "outputs": ["Perpendicular(?L,?b)"],
        "description": "If L⊥a and a∥b then L⊥b. (Variable rule)"
    },
    {
        "id": "geo_var_perp_implies_parallel",
        "name": "Two Lines Perpendicular to Same (General)",
        "inputs": ["Perpendicular(?L,?a)", "Perpendicular(?L,?b)"],
        "outputs": ["Parallel(?a,?b)"],
        "description": "If L⊥a and L⊥b then a∥b. (Variable rule)"
    },
]
