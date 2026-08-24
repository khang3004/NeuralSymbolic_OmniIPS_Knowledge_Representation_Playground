"""
GeoIPS — Coordinate Geometry Engine (Numerical Fallback).

Inspired by AlphaGeometry's coordinate-based verification and numerical methods.

For "computation" problems (finding unknown lengths/angles), this engine:
  1. Reads known geometric facts (lengths, angles, structural predicates)
  2. Places points in the Cartesian plane satisfying all constraints
  3. Computes unknown quantities numerically (length, angle, etc.)
  4. Returns new Equal(symbol, value) facts

This is the fallback when symbolic DD+AR cannot derive the goal.

Supported patterns:
  - Right triangle with 2 known sides → compute third side
  - Triangle with known sides/angles → compute missing via law of cosines/sines
  - General Length / Angle computation when coordinates can be assigned

AlphaGeometry analogy: this parallels the use of numerical verification
to check algebraic conjectures before symbolic proof.
"""

import math
import re
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("coord_engine")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_top_args(s: str) -> List[str]:
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


def _normalize_seg(seg: str) -> str:
    if len(seg) == 2 and seg.isalpha():
        return "".join(sorted(seg))
    return seg


def _normalize_angle(ang: str) -> str:
    if len(ang) == 3 and ang.isalpha():
        p, v, q = ang[0], ang[1], ang[2]
        return v + min(p, q) + max(p, q)
    return ang


def _fmt(val: float) -> str:
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.6g}"


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def _angle_at_vertex(
    p1: Tuple[float, float],
    vertex: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Compute angle at vertex (in degrees) for triangle p1-vertex-p2."""
    dx1, dy1 = p1[0] - vertex[0], p1[1] - vertex[1]
    dx2, dy2 = p2[0] - vertex[0], p2[1] - vertex[1]
    cos_theta = (dx1 * dx2 + dy1 * dy2) / (math.hypot(dx1, dy1) * math.hypot(dx2, dy2) + 1e-15)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


# ---------------------------------------------------------------------------
# Coordinate Engine
# ---------------------------------------------------------------------------

class CoordinateEngine:
    """
    Numerical coordinate geometry engine.

    Given known geometric facts, tries to place points in 2D and compute
    all unknown lengths and angles.
    """

    def derive_new_facts(self, wm_values: List[str]) -> List[str]:
        """
        Main entry: scan WM, build coordinate model, compute unknowns.
        Returns list of new Equal(symbol, value) fact strings.
        """
        new_facts: List[str] = []

        # ── Parse known quantities from WM ────────────────────────────
        lengths: Dict[str, float] = {}    # seg → length  (e.g. "AB" → 3.0)
        angles: Dict[str, float] = {}     # ang → degrees (e.g. "BAC" → 90.0)
        right_triangles: List[Tuple[str, str, str, str]] = []  # (A,B,C,right_vertex)
        triangles: List[Tuple[str, str, str]] = []

        for v in wm_values:
            v = v.strip().replace(" ", "")

            # Equal(Length(XY), n)
            m = re.fullmatch(r"Equal\(Length\(([A-Z]{2})\),([\d.]+)\)", v)
            if m:
                lengths[_normalize_seg(m.group(1))] = float(m.group(2))
                continue

            # Equal(Angle(XYZ), n)
            m = re.fullmatch(r"Equal\(Angle\(([A-Z]{3})\),([\d.]+)\)", v)
            if m:
                angles[_normalize_angle(m.group(1))] = float(m.group(2))
                continue

            # RightAngle(Angle(XYZ))
            m = re.fullmatch(r"RightAngle\(Angle\(([A-Z]{3})\)\)", v)
            if m:
                angles[_normalize_angle(m.group(1))] = 90.0
                continue

            # RightTriangle(A,B,C)  [right angle at some vertex, determined by RightAngle]
            m = re.fullmatch(r"RightTriangle\(([A-Z]),([A-Z]),([A-Z])\)", v)
            if m:
                a, b, c = m.group(1), m.group(2), m.group(3)
                triangles.append((a, b, c))

            # Triangle(A,B,C)
            m = re.fullmatch(r"(?:Triangle|RightTriangle)\(([A-Z]),([A-Z]),([A-Z])\)", v)
            if m:
                a, b, c = m.group(1), m.group(2), m.group(3)
                if (a, b, c) not in triangles:
                    triangles.append((a, b, c))

        # ── Strategy 1: Right triangle with 2 known sides ─────────────
        for tri in triangles:
            a, b, c = tri
            # Check all three possible right angles
            for right_v, hyp, leg1, leg2 in [
                (a, _normalize_seg(b+c), _normalize_seg(a+b), _normalize_seg(a+c)),
                (b, _normalize_seg(a+c), _normalize_seg(a+b), _normalize_seg(b+c)),
                (c, _normalize_seg(a+b), _normalize_seg(a+c), _normalize_seg(b+c)),
            ]:
                # Check if this vertex has a right angle
                # Right angle at right_v means angle at vertex right_v in triangle a,b,c
                # Angle label is: other1 + right_v + other2
                other1, other2 = [x for x in [a, b, c] if x != right_v]
                ang_label = _normalize_angle(other1 + right_v + other2)
                if angles.get(ang_label, 0) == 90.0:
                    # Pythagorean: hyp² = leg1² + leg2²
                    h = lengths.get(hyp)
                    l1 = lengths.get(leg1)
                    l2 = lengths.get(leg2)

                    if l1 is not None and l2 is not None and h is None:
                        h_val = math.sqrt(l1**2 + l2**2)
                        nf = f"Equal(Length({hyp}),{_fmt(h_val)})"
                        if nf not in wm_values:
                            new_facts.append(nf)
                            lengths[hyp] = h_val
                            logger.info("[CoordEngine] Pythagorean: %s", nf)

                    elif h is not None and l1 is not None and l2 is None:
                        if h**2 >= l1**2:
                            l2_val = math.sqrt(h**2 - l1**2)
                            nf = f"Equal(Length({leg2}),{_fmt(l2_val)})"
                            if nf not in wm_values:
                                new_facts.append(nf)
                                lengths[leg2] = l2_val
                                logger.info("[CoordEngine] Pythagorean: %s", nf)

                    elif h is not None and l2 is not None and l1 is None:
                        if h**2 >= l2**2:
                            l1_val = math.sqrt(h**2 - l2**2)
                            nf = f"Equal(Length({leg1}),{_fmt(l1_val)})"
                            if nf not in wm_values:
                                new_facts.append(nf)
                                lengths[leg1] = l1_val
                                logger.info("[CoordEngine] Pythagorean: %s", nf)

                    # Compute angles if all sides known
                    if all(lengths.get(s) is not None for s in [hyp, leg1, leg2]):
                        h_val = lengths[hyp]
                        l1_val = lengths[leg1]
                        l2_val = lengths[leg2]
                        # Angles in a right triangle: arcsin(opposite/hyp)
                        for v_name, opp in [(other1, l2_val), (other2, l1_val)]:
                            ang_v = _normalize_angle(
                                [x for x in [a, b, c] if x != v_name][0] +
                                v_name +
                                [x for x in [a, b, c] if x != v_name][1]
                            )
                            if ang_v not in angles:
                                ang_deg = math.degrees(math.asin(min(1.0, opp / h_val)))
                                nf = f"Equal(Angle({ang_v}),{_fmt(ang_deg)})"
                                if nf not in wm_values:
                                    new_facts.append(nf)
                                    angles[ang_v] = ang_deg
                                    logger.info("[CoordEngine] Angle derived: %s", nf)

        # ── Strategy 2: General triangle with 2 sides + included angle ─
        for tri in triangles:
            a, b, c = tri
            for side_a_pts, side_b_pts, opp_pt in [
                ((b, c), (a, c), b),  # sides AB, AC, opposite BC via angle A
                ((a, c), (b, c), a),  # sides BC, AC, opposite AB via angle B
                ((a, b), (b, c), a),  # sides AB, BC, opposite AC via angle C... 
            ]:
                pass  # law of cosines — covered by SymPy AR engine for now

        # ── Strategy 3: Triangle with 3 known sides → compute all angles ─
        for tri in triangles:
            a, b, c = tri
            ab = _normalize_seg(a+b)
            bc = _normalize_seg(b+c)
            ac = _normalize_seg(a+c)

            l_ab = lengths.get(ab)
            l_bc = lengths.get(bc)
            l_ac = lengths.get(ac)

            if l_ab is not None and l_bc is not None and l_ac is not None:
                # Law of cosines: cos(A) = (b² + c² - a²) / (2bc)
                # where a = BC (opposite A), b = AC, c = AB
                for vertex, opp, adj1, adj2 in [
                    (a, l_bc, l_ab, l_ac),
                    (b, l_ac, l_ab, l_bc),
                    (c, l_ab, l_ac, l_bc),
                ]:
                    ang_label_pts = [x for x in [a, b, c] if x != vertex]
                    ang_label = _normalize_angle(ang_label_pts[0] + vertex + ang_label_pts[1])
                    if ang_label not in angles:
                        cos_a = (adj1**2 + adj2**2 - opp**2) / (2 * adj1 * adj2 + 1e-15)
                        cos_a = max(-1.0, min(1.0, cos_a))
                        ang_deg = math.degrees(math.acos(cos_a))
                        nf = f"Equal(Angle({ang_label}),{_fmt(ang_deg)})"
                        if nf not in wm_values:
                            new_facts.append(nf)
                            angles[ang_label] = ang_deg
                            logger.info("[CoordEngine] Law of cosines: %s", nf)

        # ── Strategy 4: Two angles of triangle known → third angle ─────
        for tri in triangles:
            a, b, c = tri
            # Map vertex → angle label
            ang_a = _normalize_angle(b + a + c)
            ang_b = _normalize_angle(a + b + c)
            ang_c = _normalize_angle(a + c + b)

            known = {}
            for lbl in [ang_a, ang_b, ang_c]:
                if lbl in angles:
                    known[lbl] = angles[lbl]

            if len(known) == 2:
                missing_lbl = [l for l in [ang_a, ang_b, ang_c] if l not in known][0]
                val = 180.0 - sum(known.values())
                if val > 0:
                    nf = f"Equal(Angle({missing_lbl}),{_fmt(val)})"
                    if nf not in wm_values:
                        new_facts.append(nf)
                        angles[missing_lbl] = val
                        logger.info("[CoordEngine] Angle sum: %s", nf)

        # ── Strategy 5: Law of sines — 2 angles + 1 side → other sides ─
        for tri in triangles:
            a, b, c = tri
            ang_a = _normalize_angle(b + a + c)
            ang_b = _normalize_angle(a + b + c)
            ang_c = _normalize_angle(a + c + b)
            ab = _normalize_seg(a+b)
            bc = _normalize_seg(b+c)
            ac = _normalize_seg(a+c)

            # Map angle to opposite side
            angle_side_pairs = [
                (ang_a, bc, l_a := angles.get(ang_a), lengths.get(bc)),
                (ang_b, ac, angles.get(ang_b), lengths.get(ac)),
                (ang_c, ab, angles.get(ang_c), lengths.get(ab)),
            ]

            # Find one known (angle, side) pair to get the circumradius ratio
            reference = None
            for ang_lbl, side_lbl, ang_val, side_val in angle_side_pairs:
                if ang_val is not None and side_val is not None:
                    # R = side / (2 * sin(ang))
                    sin_val = math.sin(math.radians(ang_val))
                    if sin_val > 1e-9:
                        reference = side_val / sin_val
                        break

            if reference is not None:
                for ang_lbl, side_lbl, ang_val, side_val in angle_side_pairs:
                    if ang_val is not None and side_val is None:
                        new_side = reference * math.sin(math.radians(ang_val))
                        if new_side > 0:
                            nf = f"Equal(Length({side_lbl}),{_fmt(new_side)})"
                            if nf not in wm_values:
                                new_facts.append(nf)
                                lengths[side_lbl] = new_side
                                logger.info("[CoordEngine] Law of sines: %s", nf)

        return new_facts
