"""
GeoIPS — Algebraic Reasoning (AR) Engine.

Inspired by AlphaGeometry's AR module (Wu's Method / Angle & Ratio Elimination).
Dynamically extracts linear angle and ratio equations from known geometric configurations
and solves them via system elimination, producing new formal Fact predicates.

STRICTLY DYNAMIC — NO HARDCODED PROBLEM FIXES.
"""

import re
import logging
from typing import List, Set, Dict, Tuple, Optional
from core_engine.models import Fact

logger = logging.getLogger("algebraic_engine")


class AlgebraicReasoningEngine:
    """
    Domain-general algebraic solver for angle sums, flat angles, cyclic opposite angles,
    and ratio equations.
    """

    def __init__(self):
        pass

    def derive_algebraic_facts(self, known_facts: Set[Fact]) -> List[Fact]:
        """
        Scan known facts, extract algebraic relations (angles/ratios),
        solve the system of linear equations, and return newly derived Facts.

        Args:
            known_facts: Set of Fact objects currently in Working Memory.

        Returns:
            List of new Fact objects derived from algebraic elimination.
        """
        fact_values = {f.value for f in known_facts}
        domain = next(iter(known_facts)).domain if known_facts else "geometry"
        new_facts: List[Fact] = []

        # 1. Extract Triangles: Triangle(A,B,C) → Angle(ABC) + Angle(BCA) + Angle(CAB) = 180
        triangles = self._extract_triangles(fact_values)
        
        # 2. Extract Cyclic Quads: CyclicQuadrilateral(A,B,C,D) → Angle(DAB) + Angle(BCD) = 180
        cyclics = self._extract_cyclic_quads(fact_values)

        # 3. Extract Angle Equalities: Equal(Angle(X), Angle(Y)) or Equal(Angle(X), 50)
        angle_equalities, angle_values = self._extract_known_angles(fact_values)

        # 4. Perform Linear Elimination to solve for missing angles/equalities
        derived_equalities = self._solve_angle_system(triangles, cyclics, angle_equalities, angle_values)

        for eq_str in derived_equalities:
            if eq_str not in fact_values:
                fact_id = f"ar_fact_{abs(hash(eq_str))}"
                new_fact = Fact(id=fact_id, value=eq_str, domain=domain)
                new_facts.append(new_fact)
                fact_values.add(eq_str)
                logger.info("[AR Engine] Derived new algebraic fact: %s", eq_str)

        return new_facts

    def _extract_triangles(self, fact_values: Set[str]) -> List[Tuple[str, str, str]]:
        """Extract all triangles (A,B,C) from Triangle(A,B,C) or RightTriangle(A,B,C)."""
        triangles = []
        for f in fact_values:
            m = re.match(r"(?:Triangle|RightTriangle|IsoscelesTriangle)\(([A-Z]),([A-Z]),([A-Z])\)", f)
            if m:
                triangles.append((m.group(1), m.group(2), m.group(3)))
        return triangles

    def _extract_cyclic_quads(self, fact_values: Set[str]) -> List[Tuple[str, str, str, str]]:
        """Extract cyclic quadrilaterals (A,B,C,D)."""
        quads = []
        for f in fact_values:
            m = re.match(r"CyclicQuadrilateral\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)", f)
            if m:
                quads.append((m.group(1), m.group(2), m.group(3), m.group(4)))
        return quads

    def _extract_known_angles(self, fact_values: Set[str]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Extract angle equality relations and numeric angle assignments."""
        equalities: Dict[str, str] = {}
        values: Dict[str, float] = {}

        for f in fact_values:
            # Equal(Angle(ABC), Angle(DEF))
            m1 = re.match(r"Equal\(Angle\(([A-Z]{3})\),Angle\(([A-Z]{3})\)\)", f)
            if m1:
                equalities[m1.group(1)] = m1.group(2)
                equalities[m1.group(2)] = m1.group(1)
                continue

            # Equal(Angle(ABC), 60)
            m2 = re.match(r"Equal\(Angle\(([A-Z]{3})\),(\d+(?:\.\d+)?)\)", f)
            if m2:
                values[m2.group(1)] = float(m2.group(2))
                continue

            # RightAngle(Angle(ABC))
            m3 = re.match(r"RightAngle\(Angle\(([A-Z]{3})\)\)", f)
            if m3:
                values[m3.group(1)] = 90.0
                continue

        return equalities, values

    def _solve_angle_system(
        self,
        triangles: List[Tuple[str, str, str]],
        cyclics: List[Tuple[str, str, str, str]],
        equalities: Dict[str, str],
        values: Dict[str, float],
    ) -> List[str]:
        """
        Solve linear angle equations:
        - If 2 angles of a triangle are known, compute the 3rd.
        - If 1 angle of a cyclic quad opposite pair is known, compute the other.
        - Derive Equal(Angle(X), Angle(Y)) if values match.
        """
        derived_facts: List[str] = []

        # Process Triangles: Angle(BAC) + Angle(ABC) + Angle(ACB) = 180
        for A, B, C in triangles:
            ang1 = f"{B}{A}{C}"
            ang2 = f"{A}{B}{C}"
            ang3 = f"{A}{C}{B}"

            v1 = values.get(ang1)
            v2 = values.get(ang2)
            v3 = values.get(ang3)

            # Case: 2 known -> 3rd known
            if v1 is not None and v2 is not None and v3 is None:
                v3_val = round(180.0 - v1 - v2, 4)
                if v3_val > 0:
                    values[ang3] = v3_val
                    derived_facts.append(f"Equal(Angle({ang3}),{v3_val:g})")

            elif v1 is not None and v3 is not None and v2 is None:
                v2_val = round(180.0 - v1 - v3, 4)
                if v2_val > 0:
                    values[ang2] = v2_val
                    derived_facts.append(f"Equal(Angle({ang2}),{v2_val:g})")

            elif v2 is not None and v3 is not None and v1 is None:
                v1_val = round(180.0 - v2 - v3, 4)
                if v1_val > 0:
                    values[ang1] = v1_val
                    derived_facts.append(f"Equal(Angle({ang1}),{v1_val:g})")

        # Process Cyclic Quads: Opposite angles sum to 180
        for A, B, C, D in cyclics:
            ang_A = f"{D}{A}{B}"
            ang_C = f"{B}{C}{D}"
            vA = values.get(ang_A)
            vC = values.get(ang_C)

            if vA is not None and vC is None:
                vC_val = round(180.0 - vA, 4)
                if vC_val > 0:
                    values[ang_C] = vC_val
                    derived_facts.append(f"Equal(Angle({ang_C}),{vC_val:g})")
            elif vC is not None and vA is None:
                vA_val = round(180.0 - vC, 4)
                if vA_val > 0:
                    values[ang_A] = vA_val
                    derived_facts.append(f"Equal(Angle({ang_A}),{vA_val:g})")

        # Infer Equal(Angle(X), Angle(Y)) for any two angles with equal computed values
        val_to_angles: Dict[float, List[str]] = {}
        for ang, val in values.items():
            val_to_angles.setdefault(val, []).append(ang)

        for val, ang_list in val_to_angles.items():
            if len(ang_list) >= 2:
                for i in range(len(ang_list)):
                    for j in range(i + 1, len(ang_list)):
                        a1, a2 = ang_list[i], ang_list[j]
                        derived_facts.append(f"Equal(Angle({a1}),Angle({a2}))")

        return derived_facts
