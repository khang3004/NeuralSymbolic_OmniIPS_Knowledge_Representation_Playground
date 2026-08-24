"""
GeoIPS — FastAPI Gateway.

Single-domain geometry IPS API inspired by AlphaGeometry.
Endpoints:
  GET  /health                  — system health check
  POST /solve                   — raw predicate solve (propositional)
  POST /api/solve               — natural language solve (GraphRAG + LLM)
  POST /geo/solve               — AlphaGeometry-style: solver + auxiliary construction loop
  POST /api/explain             — sync proof explanation (LLM or template)
  POST /api/explain/stream      — streaming proof explanation
  GET  /rules                   — list geometry rules from Neo4j
  GET  /ontology/classes        — list ontology class hierarchy from Neo4j
"""

import os
import logging
import asyncio
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from core_engine import ForwardChainingEngine, BackwardChainingEngine
from domains.geometry import GeometryParser
from graph_db.connection import Neo4jConnection
from rag_agent.router import route_query

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("geo-ips-api")

app = FastAPI(
    title="GeoIPS API Gateway",
    description=(
        "Plane Geometry Intelligent Problem Solver — "
        "Neuro-Symbolic & GraphRAG engine inspired by AlphaGeometry."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PARSER = GeometryParser()
DOMAIN = "geometry"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_rules_from_neo4j(session) -> list:
    """Load all geometry rules from Neo4j via Cypher."""
    result = session.run(
        "MATCH (r:Rule) WHERE r.domain = $domain "
        "OPTIONAL MATCH (f_in:Fact)-[:HAS_INPUT]->(r) "
        "OPTIONAL MATCH (r)-[:HAS_OUTPUT]->(f_out:Fact) "
        "WITH r, "
        "     CASE WHEN r.inputs IS NOT NULL THEN r.inputs ELSE collect(DISTINCT f_in.value) END AS inputs, "
        "     CASE WHEN r.outputs IS NOT NULL THEN r.outputs ELSE collect(DISTINCT f_out.value) END AS outputs "
        "RETURN r.id AS id, r.name AS name, inputs, outputs, r.description AS description",
        domain=DOMAIN,
    )
    raw_rules = []
    for record in result:
        raw_rules.append(
            {
                "id": record["id"],
                "name": record["name"],
                "inputs": record["inputs"] or [],
                "outputs": record["outputs"] or [],
                "description": record["description"] or "",
            }
        )
    return raw_rules


# Built-in fallback rule subset (used when Neo4j is unreachable)
FALLBACK_GEOMETRY_RULES = [
    {
        "id": "geo_congruence_reflexive",
        "name": "Congruence Reflexive",
        "inputs": ["Segment(AB)"],
        "outputs": ["Congruent(AB,AB)"],
        "description": "AB ≅ AB.",
    },
    {
        "id": "geo_congruence_symmetric",
        "name": "Congruence Symmetric",
        "inputs": ["Congruent(AB,CD)"],
        "outputs": ["Congruent(CD,AB)"],
        "description": "If AB≅CD then CD≅AB.",
    },
    {
        "id": "geo_congruence_transitive",
        "name": "Congruence Transitivity",
        "inputs": ["Congruent(AB,CD)", "Congruent(CD,EF)"],
        "outputs": ["Congruent(AB,EF)"],
        "description": "Transitivity of congruence.",
    },
    {
        "id": "geo_perp_symmetry",
        "name": "Perpendicular Symmetry",
        "inputs": ["Perpendicular(AB,CD)"],
        "outputs": ["Perpendicular(CD,AB)"],
        "description": "Perpendicularity is symmetric.",
    },
    {
        "id": "geo_parallel_transitive",
        "name": "Parallel Transitivity",
        "inputs": ["Parallel(a,b)", "Parallel(b,c)"],
        "outputs": ["Parallel(a,c)"],
        "description": "Transitivity of parallel lines.",
    },
    # Variable-based parallel/perp rules (match ANY line names — uppercase AB, CD, etc.)
    {
        "id": "geo_parallel_transitive_var",
        "name": "Parallel Transitivity (general)",
        "inputs": ["Parallel(?A,?B)", "Parallel(?B,?C)"],
        "outputs": ["Parallel(?A,?C)"],
        "description": "Transitivity: AB∥CD ∧ CD∥EF ⇒ AB∥EF",
    },
    {
        "id": "geo_parallel_symmetric_var",
        "name": "Parallel Symmetric (general)",
        "inputs": ["Parallel(?A,?B)"],
        "outputs": ["Parallel(?B,?A)"],
        "description": "Parallel is symmetric.",
    },
    {
        "id": "geo_perp_symmetry_var",
        "name": "Perpendicular Symmetric (general)",
        "inputs": ["Perpendicular(?A,?B)"],
        "outputs": ["Perpendicular(?B,?A)"],
        "description": "Perpendicularity is symmetric.",
    },
    {
        "id": "geo_triangle_angle_sum",
        "name": "Triangle Angle Sum",
        "inputs": ["Triangle(A,B,C)"],
        "outputs": ["Equal(Add(Angle(BAC),Angle(ABC),Angle(ACB)),180)"],
        "description": "Angles of a triangle sum to 180°.",
    },
    {
        "id": "geo_isosceles_base_angles",
        "name": "Isosceles Base Angles",
        "inputs": ["Triangle(A,B,C)", "Congruent(AB,AC)"],
        "outputs": ["Equal(Angle(ABC),Angle(ACB))"],
        "description": "Base angles of isosceles triangle are equal.",
    },
    {
        "id": "geo_isosceles_reverse",
        "name": "Converse Isosceles",
        "inputs": ["Triangle(A,B,C)", "Equal(Angle(ABC),Angle(ACB))"],
        "outputs": ["Congruent(AB,AC)"],
        "description": "Equal base angles implies isosceles.",
    },
    # ── Congruence theorems — variable-based, ANY triangle pair ────────────
    # IDs use _var suffix so they are ALWAYS merged (never overridden by Neo4j's
    # propositional geo_sas_congruence / geo_asa_congruence etc.).
    {
        "id": "geo_sas_congruence_var",
        "name": "SAS Congruence (general)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Triangle(?D,?E,?F)",
            "Congruent(?A?B,?D?E)",
            "Equal(Angle(?B?A?C),Angle(?E?D?F))",
            "Congruent(?A?C,?D?F)",
        ],
        "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
        "description": "SAS: AB=DE, ∠BAC=∠EDF, AC=DF ⇒ △ABC≅△DEF",
    },
    {
        "id": "geo_asa_congruence_var",
        "name": "ASA Congruence (general)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Triangle(?D,?E,?F)",
            "Equal(Angle(?B?A?C),Angle(?E?D?F))",
            "Congruent(?A?B,?D?E)",
            "Equal(Angle(?A?B?C),Angle(?D?E?F))",
        ],
        "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
        "description": "ASA: ∠BAC=∠EDF, AB=DE, ∠ABC=∠DEF ⇒ △ABC≅△DEF",
    },
    {
        "id": "geo_sss_congruence_var",
        "name": "SSS Congruence (general)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Triangle(?D,?E,?F)",
            "Congruent(?A?B,?D?E)",
            "Congruent(?B?C,?E?F)",
            "Congruent(?A?C,?D?F)",
        ],
        "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
        "description": "SSS: AB=DE, BC=EF, AC=DF ⇒ △ABC≅△DEF",
    },
    {
        "id": "geo_aas_congruence_var",
        "name": "AAS Congruence (general)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Triangle(?D,?E,?F)",
            "Equal(Angle(?B?A?C),Angle(?E?D?F))",
            "Equal(Angle(?A?B?C),Angle(?D?E?F))",
            "Congruent(?B?C,?E?F)",
        ],
        "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
        "description": "AAS: ∠BAC=∠EDF, ∠ABC=∠DEF, BC=EF ⇒ △ABC≅△DEF",
    },
    # Symmetry: CongruentTriangles(ABC,DEF) ⇔ CongruentTriangles(DEF,ABC)
    {
        "id": "geo_congruent_tri_symmetric",
        "name": "Congruent Triangles Symmetric",
        "inputs": ["CongruentTriangles(?ABC,?DEF)"],
        "outputs": ["CongruentTriangles(?DEF,?ABC)"],
        "description": "Triangle congruence is symmetric.",
    },
    {
        "id": "geo_pythagoras_var",
        "name": "Pythagorean Theorem",
        "inputs": ["RightTriangle(?A,?B,?C)", "RightAngle(Angle(?B?A?C))"],
        "outputs": [
            "Equal(Pow(Length(?B?C),2),Add(Pow(Length(?A?B),2),Pow(Length(?A?C),2)))"
        ],
        "description": "BC² = AB² + AC² in a right-angled triangle.",
    },
    {
        "id": "geo_pythagoras_converse_var",
        "name": "Pythagorean Converse",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Equal(Pow(Length(?B?C),2),Add(Pow(Length(?A?B),2),Pow(Length(?A?C),2)))",
        ],
        "outputs": ["RightTriangle(?A,?B,?C)", "RightAngle(Angle(?B?A?C))"],
        "description": "If BC² = AB² + AC² then the triangle is right-angled at A.",
    },
    {
        "id": "geo_right_triangle_height_metric_var",
        "name": "Right Triangle Height Metric",
        "inputs": [
            "RightTriangle(?A,?B,?C)",
            "RightAngle(Angle(?B?A?C))",
            "Foot(?H,?A,?B?C)",
        ],
        "outputs": [
            "Equal(Div(1,Pow(Length(?A?H),2)),Add(Div(1,Pow(Length(?A?B),2)),Div(1,Pow(Length(?A?C),2))))"
        ],
        "description": "1/AH² = 1/AB² + 1/AC² in a right triangle with altitude AH.",
    },
    {
        "id": "geo_thales_var",
        "name": "Thales Theorem",
        "inputs": ["Diameter(?A?B,Circle(?O))", "PointOnCircle(?C,Circle(?O))"],
        "outputs": ["RightAngle(Angle(?A?C?B))"],
        "description": "Angle in a semicircle is 90°.",
    },
    {
        "id": "geo_congruent_tri_sides_var",
        "name": "Congruent Triangles → Sides",
        "inputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
        "outputs": [
            "Congruent(?A?B,?D?E)",
            "Congruent(?B?C,?E?F)",
            "Congruent(?A?C,?D?F)",
        ],
        "description": "Congruent triangles have congruent sides.",
    },
    {
        "id": "geo_congruent_tri_angles_var",
        "name": "Congruent Triangles → Angles",
        "inputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
        "outputs": [
            "Equal(Angle(?B?A?C),Angle(?E?D?F))",
            "Equal(Angle(?A?B?C),Angle(?D?E?F))",
            "Equal(Angle(?A?C?B),Angle(?D?F?E))",
        ],
        "description": "Congruent triangles have equal angles.",
    },
    {
        "id": "geo_midpoint_theorem_var",
        "name": "Midsegment Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "Midpoint(?M,?A?B)", "Midpoint(?N,?A?C)"],
        "outputs": ["Parallel(?M?N,?B?C)", "Equal(Length(?M?N),Div(Length(?B?C),2))"],
        "description": "Midsegment is parallel to base and half its length.",
    },
    {
        "id": "geo_right_triangle_expand_var",
        "name": "RightTriangle Expand",
        "inputs": ["RightTriangle(?A,?B,?C)"],
        "outputs": ["Triangle(?A,?B,?C)", "RightAngle(Angle(?B?A?C))"],
        "description": "Expand RightTriangle to Triangle + RightAngle.",
    },
    {
        "id": "geo_exterior_angle_var",
        "name": "Exterior Angle Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "ExteriorAngle(?E,?A,?B?C)"],
        "outputs": ["Equal(Angle(?E),Add(Angle(?B?A?C),Angle(?A?B?C)))"],
        "description": "Exterior angle is sum of two non-adjacent interior angles.",
    },
    {
        "id": "geo_equilateral_all_60_var",
        "name": "Equilateral Triangle Angles",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Congruent(?A?B,?B?C)",
            "Congruent(?B?C,?A?C)",
        ],
        "outputs": [
            "Equal(Angle(?B?A?C),60)",
            "Equal(Angle(?A?B?C),60)",
            "Equal(Angle(?A?C?B),60)",
        ],
        "description": "All angles of equilateral triangle are 60°.",
    },
    {
        "id": "geo_thales_right_angle_var",
        "name": "Thales: Angle in Semicircle",
        "inputs": ["Diameter(?A?B,Circle(?O))", "PointOnCircle(?C,Circle(?O))"],
        "outputs": ["RightAngle(Angle(?A?C?B))", "Equal(Angle(?A?C?B),90)"],
        "description": "Angle inscribed in a semicircle is 90°.",
    },
    {
        "id": "geo_cyclic_quad_opposite_var",
        "name": "Cyclic Quadrilateral Opposite Angles",
        "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D,Circle(?O))"],
        "outputs": [
            "Equal(Add(Angle(?D?A?B),Angle(?B?C?D)),180)",
            "Equal(Add(Angle(?A?B?C),Angle(?C?D?A)),180)",
        ],
        "description": "Opposite angles of cyclic quadrilateral sum to 180°.",
    },
    {
        "id": "geo_aa_similarity_var",
        "name": "AA Similarity",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Triangle(?D,?E,?F)",
            "Equal(Angle(?B?A?C),Angle(?E?D?F))",
            "Equal(Angle(?A?B?C),Angle(?D?E?F))",
        ],
        "outputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
        "description": "AA similarity theorem.",
    },
    {
        "id": "geo_similar_tri_angles_var",
        "name": "Similar Triangles: Equal Angles",
        "inputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
        "outputs": [
            "Equal(Angle(?B?A?C),Angle(?E?D?F))",
            "Equal(Angle(?A?B?C),Angle(?D?E?F))",
            "Equal(Angle(?A?C?B),Angle(?D?F?E))",
        ],
        "description": "Similar triangles have equal corresponding angles.",
    },
    {
        "id": "geo_similar_tri_ratios_var",
        "name": "Similar Triangles: Proportional Sides",
        "inputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
        "outputs": [
            "Equal(Div(Length(?A?B),Length(?D?E)),Div(Length(?B?C),Length(?E?F)))",
            "Equal(Div(Length(?A?B),Length(?D?E)),Div(Length(?A?C),Length(?D?F)))",
        ],
        "description": "Similar triangles have proportional corresponding sides.",
    },
    {
        "id": "geo_midpoint_halves_var",
        "name": "Midpoint Halves Segment",
        "inputs": ["Midpoint(?M,?A?B)"],
        "outputs": [
            "Equal(Length(?A?M),Length(?M?B))",
            "Equal(Length(?A?M),Div(Length(?A?B),2))",
        ],
        "description": "Midpoint divides segment into two equal halves.",
    },
    # ── Circle & Power of Point Theorems ──────────────────────────────────────
    {
        "id": "geo_chord_definition_var",
        "name": "Chord Definition",
        "inputs": ["PointOnCircle(?A,Circle(?O))", "PointOnCircle(?B,Circle(?O))"],
        "outputs": ["Chord(?A,?B,Circle(?O))"],
        "description": "A segment connecting two points on a circle is a chord.",
    },
    {
        "id": "geo_intersecting_chords_var",
        "name": "Intersecting Chords Theorem",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
            "PointOnCircle(?D,Circle(?O))",
            "IntersectionPoint(?P,?A?B,?C?D)",
        ],
        "outputs": [
            "Equal(Mul(Length(?A?P),Length(?P?B)),Mul(Length(?C?P),Length(?P?D)))"
        ],
        "description": "Power of Point: PA * PB = PC * PD for intersecting chords.",
    },
    {
        "id": "geo_tangent_secant_theorem_var",
        "name": "Tangent-Secant Theorem",
        "inputs": [
            "Circle(?O)",
            "PointOutsideCircle(?P,Circle(?O))",
            "TangentSegment(?P,?T,Circle(?O))",
            "SecantSegment(?P,?A,?B,Circle(?O))",
        ],
        "outputs": ["Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?P?B)))"],
        "description": "Square of tangent segment equals product of secant segments.",
    },
    {
        "id": "geo_two_tangents_cyclic_quad_var",
        "name": "Two Tangents Form Cyclic Quadrilateral with Center",
        "inputs": [
            "Circle(?O)",
            "PointOutsideCircle(?M,Circle(?O))",
            "TangentSegment(?M,?A,Circle(?O))",
            "TangentSegment(?M,?B,Circle(?O))",
        ],
        "outputs": [
            "CyclicQuadrilateral(?M,?A,?O,?B)",
            "CyclicQuadrilateral(?M,?B,?O,?A)",
            "CyclicQuadrilateral(?A,?M,?B,?O)",
            "CyclicQuadrilateral(?O,?A,?M,?B)",
        ],
        "description": "Two tangent segments MA and MB from external point M together with center O form cyclic quad MAOB (since OAM = OBM = 90 deg).",
    },
    {
        "id": "geo_ptolemy_theorem_var",
        "name": "Ptolemy's Theorem",
        "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D)"],
        "outputs": [
            "Equal(Mul(Length(?A?C),Length(?B?D)),Add(Mul(Length(?A?B),Length(?C?D)),Mul(Length(?B?C),Length(?A?D))))",
            "Equal(Mul(Length(?A?C),Length(?B?D)),Add(Mul(Length(?A?B),Length(?C?D)),Mul(Length(?A?D),Length(?B?C))))",
        ],
        "description": "Ptolemy: AC * BD = AB * CD + BC * AD.",
    },
    {
        "id": "geo_ptolemy_theorem_var2",
        "name": "Ptolemy's Theorem (Circle qualified)",
        "inputs": ["Circle(?O)", "CyclicQuadrilateral(?A,?B,?C,?D,Circle(?O))"],
        "outputs": [
            "Equal(Mul(Length(?A?C),Length(?B?D)),Add(Mul(Length(?A?B),Length(?C?D)),Mul(Length(?B?C),Length(?A?D))))",
            "Equal(Mul(Length(?A?C),Length(?B?D)),Add(Mul(Length(?A?B),Length(?C?D)),Mul(Length(?A?D),Length(?B?C))))",
        ],
        "description": "Ptolemy: AC * BD = AB * CD + BC * AD.",
    },
    {
        "id": "geo_ptolemy_theorem_var3",
        "name": "Ptolemy's Theorem (CyclicQuadrilateral with Circle)",
        "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D,Circle(?O))"],
        "outputs": [
            "Equal(Mul(Length(?A?C),Length(?B?D)),Add(Mul(Length(?A?B),Length(?C?D)),Mul(Length(?B?C),Length(?A?D))))",
            "Equal(Mul(Length(?A?C),Length(?B?D)),Add(Mul(Length(?A?B),Length(?C?D)),Mul(Length(?A?D),Length(?B?C))))",
        ],
        "description": "Ptolemy: AC * BD = AB * CD + BC * AD.",
    },
    # ── Rhombus & Olympiad Lemmas ─────────────────────────────────────────────
    {
        "id": "geo_rhombus_diagonals_perp_var",
        "name": "Rhombus Diagonals are Perpendicular",
        "inputs": ["Rhombus(?A,?B,?C,?D)"],
        "outputs": ["Perpendicular(?A?C,?B?D)"],
        "description": "The diagonals of a rhombus are perpendicular to each other.",
    },
    {
        "id": "geo_ceva_theorem_var",
        "name": "Ceva's Theorem",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "PointOnSegment(?D,?B,?C)",
            "PointOnSegment(?E,?A,?C)",
            "PointOnSegment(?F,?A,?B)",
            "Concurrent(?A?D,?B?E,?C?F)",
        ],
        "outputs": [
            "Equal(Mul(Div(Length(?B?D),Length(?C?D)),Mul(Div(Length(?C?E),Length(?A?E)),Div(Length(?A?F),Length(?B?F)))),1)"
        ],
        "description": "Condition for three cevians to be concurrent.",
    },
    {
        "id": "geo_menelaus_theorem_var",
        "name": "Menelaus's Theorem",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "PointOnLine(?D,?B,?C)",
            "PointOnLine(?E,?C,?A)",
            "PointOnLine(?F,?A,?B)",
            "Collinear(?D,?E,?F)",
        ],
        "outputs": [
            "Equal(Mul(Div(Length(?B?D),Length(?C?D)),Mul(Div(Length(?C?E),Length(?A?E)),Div(Length(?A?F),Length(?B?F)))),1)"
        ],
        "description": "Menelaus collinearity theorem.",
    },
    {
        "id": "geo_simson_line_var",
        "name": "Simson Line Theorem",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "PointOnCircle(?P,Circle(?O))",
            "Circumcircle(?O,?A,?B,?C)",
            "Foot(?X,?P,?A?B)",
            "Foot(?Y,?P,?B?C)",
            "Foot(?Z,?P,?A?C)",
        ],
        "outputs": ["Collinear(?X,?Y,?Z)"],
        "description": "The feet of the perpendiculars from a point on the circumcircle are collinear.",
    },
    {
        "id": "geo_parallel_quadrilateral_properties_var",
        "name": "Parallel Quadrilateral Diagonal Properties",
        "inputs": [
            "Parallel(?A?B,?C?D)",
            "Parallel(?A?D,?B?C)",
            "PointOnSegment(?M,?A,?C)",
            "PointOnSegment(?N,?A,?C)",
        ],
        "outputs": [
            "Congruent(?A?D,?C?B)",
            "Congruent(?A?B,?C?D)",
            "Triangle(?A,?D,?M)",
            "Triangle(?C,?B,?N)",
            "Equal(Angle(?D?A?M),Angle(?B?C?N))",
        ],
        "description": "If opposite sides of a quadrilateral are parallel, it is a parallelogram, yielding congruent opposite sides, diagonal alternate angles, and triangle existence.",
    },
    {
        "id": "geo_parallelogram_properties_var",
        "name": "Parallelogram Properties",
        "inputs": [
            "Parallelogram(?A,?B,?C,?D)",
            "PointOnSegment(?M,?A,?C)",
            "PointOnSegment(?N,?A,?C)",
        ],
        "outputs": [
            "Parallel(?A?B,?C?D)",
            "Parallel(?A?D,?B?C)",
            "Congruent(?A?D,?C?B)",
            "Congruent(?A?B,?C?D)",
            "Triangle(?A,?D,?M)",
            "Triangle(?C,?B,?N)",
            "Equal(Angle(?D?A?M),Angle(?B?C?N))",
        ],
        "description": "Properties of a parallelogram with points on diagonal AC.",
    },
    {
        "id": "geo_isosceles_midpoint_perp_var",
        "name": "Isosceles Triangle Midpoint Altitude",
        "inputs": ["Triangle(?A,?B,?C)", "Congruent(?A?B,?A?C)", "Midpoint(?M,?B?C)"],
        "outputs": [
            "Perpendicular(?A?M,?B?C)",
            "RightAngle(Angle(?B?M?A))",
            "RightAngle(Angle(?A?M?B))",
            "RightAngle(Angle(?C?M?A))",
        ],
        "description": "In an isosceles triangle AB=AC, the median AM is perpendicular to BC.",
    },
    {
        "id": "geo_isosceles_midpoint_perp_length_var",
        "name": "Isosceles Triangle Midpoint Altitude (Length)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Equal(Length(?A?B),Length(?A?C))",
            "Midpoint(?M,?B?C)",
        ],
        "outputs": [
            "Perpendicular(?A?M,?B?C)",
            "RightAngle(Angle(?B?M?A))",
            "RightAngle(Angle(?A?M?B))",
            "RightAngle(Angle(?C?M?A))",
        ],
        "description": "In an isosceles triangle AB=AC, the median AM is perpendicular to BC.",
    },
    {
        "id": "geo_circle_radii_equal_var",
        "name": "Circle Radii Length Equality",
        "inputs": ["PointOnCircle(?M,Circle(?O))", "PointOnCircle(?N,Circle(?O))"],
        "outputs": [
            "Equal(Length(?O?M),Length(?O?N))",
            "Congruent(?M?O,?N?O)",
            "Congruent(?O?M,?O?N)",
            "Congruent(?M?O,?O?N)",
            "Congruent(?O?M,?N?O)",
            "Congruent(?M?O,?N?O)",
            "Congruent(MO,NO)",
        ],
        "description": "All radii from the center of a circle to points on the circle are equal in length.",
    },
    {
        "id": "geo_point_on_circle_implies_circle_var",
        "name": "Point On Circle Implies Circle Existence",
        "inputs": ["PointOnCircle(?M,Circle(?O))"],
        "outputs": ["Circle(?O)"],
        "description": "If M is on Circle(O), then Circle(O) exists.",
    },
    {
        "id": "geo_circle_midpoint_radius_var",
        "name": "Circle Midpoint Diameter Definition",
        "inputs": ["Midpoint(?O,?A?B)", "Circle(?O)", "PointOnCircle(?C,Circle(?O))"],
        "outputs": [
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "Diameter(?A?B)",
            "RightAngle(Angle(?A?C?B))",
            "RightAngle(Angle(?C))",
        ],
        "description": "If O is midpoint of AB and C is on Circle(O), then AB is a diameter and angle ACB is a right angle.",
    },
    {
        "id": "geo_midpoint_congruence_var",
        "name": "Midpoint Congruence Definition",
        "inputs": ["Midpoint(?M,?B?C)"],
        "outputs": [
            "Equal(Length(?B?M),Length(?M?C))",
            "Congruent(?B?M,?C?M)",
            "Congruent(?B?M,?M?C)",
        ],
        "description": "Midpoint M of BC divides segment BC into congruent halves BM and MC.",
    },
    {
        "id": "geo_circumcenter_midpoint_perp_var",
        "name": "Circumcenter Midpoint Perpendicular Bisector",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Circumcircle(?O,?A,?B,?C)",
            "Midpoint(?M,?B?C)",
        ],
        "outputs": [
            "Perpendicular(?O?M,?B?C)",
            "Perpendicular(?M?O,?B?C)",
            "RightAngle(Angle(?O?M?B))",
        ],
        "description": "The segment from circumcenter O to midpoint M of BC is perpendicular to BC.",
    },
    {
        "id": "geo_circle_points_midpoint_perp_var",
        "name": "Circle Points Midpoint Perpendicular Bisector",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
            "Midpoint(?M,?B?C)",
        ],
        "outputs": [
            "Perpendicular(?O?M,?B?C)",
            "Perpendicular(?M?O,?B?C)",
            "RightAngle(Angle(?O?M?B))",
        ],
        "description": "The segment from center O of circumcircle to midpoint M of BC is perpendicular to BC.",
    },
    {
        "id": "geo_congruent_radii_is_circumcenter_var",
        "name": "Congruent Radii form Circumcenter Midpoint Perpendicular",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Congruent(?A?O,?B?O)",
            "Congruent(?B?O,?C?O)",
            "Midpoint(?M,?B?C)",
        ],
        "outputs": [
            "Perpendicular(?O?M,?B?C)",
            "Perpendicular(?M?O,?B?C)",
            "RightAngle(Angle(?O?M?B))",
        ],
        "description": "If O is equidistant from vertices A, B, C of triangle ABC, then OM is perpendicular to BC.",
    },
    {
        "id": "geo_square_intersection_coincident_var",
        "name": "Square Circumcircle Intersection Coincidence",
        "inputs": [
            "PointOnSegment(?M,?A,?B)",
            "Square(?A,?M,?C,?D)",
            "Square(?M,?B,?E,?F)",
        ],
        "outputs": [
            "Equal(Point(?N),Point(?N_prime))",
            "Equal(?N,?N_prime)",
            "Coincident(?N,?N_prime)",
            "Point(?N)",
            "Equal(N,N)",
        ],
        "description": "Intersection of square circumcircles coincides with AF and BC intersection.",
    },
    {
        "id": "geo_triangle_inset_area_inequality_var",
        "name": "Triangle Inset Area Inequality",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "PointOnSegment(?K,?B,?C)",
            "PointOnSegment(?L,?C,?A)",
            "PointOnSegment(?M,?A,?B)",
        ],
        "outputs": [
            "LessEqual(Area(Triangle(?A,?M,?L)),Mul(Div(1,4),Area(Triangle(?A,?B,?C))))",
            "Triangle(?A,?M,?L)",
        ],
        "description": "The area of inset triangle AML is at most one fourth of the total area of ABC.",
    },
    {
        "id": "geo_angle_bisector_theorem_var",
        "name": "Angle Bisector Theorem",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "AngleBisector(?A?D,Angle(?B?A?C))",
            "PointOnSegment(?D,?B,?C)",
        ],
        "outputs": [
            "Equal(Div(Length(?B?D),Length(?C?D)),Div(Length(?A?B),Length(?A?C)))",
            "Equal(Div(Length(?B?D),Length(?D?C)),Div(Length(?A?B),Length(?A?C)))",
        ],
        "description": "An angle bisector of a triangle divides the opposite side into segments proportional to the adjacent sides.",
    },
    {
        "id": "geo_angle_equal_is_bisector_var",
        "name": "Equal Angles form Angle Bisector",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Equal(Angle(?B?A?D),Angle(?C?A?D))",
            "PointOnSegment(?D,?B,?C)",
        ],
        "outputs": ["AngleBisector(?A?D,Angle(?B?A?C))"],
        "description": "If D is on segment BC and angle BAD equals angle CAD, then AD is the angle bisector of angle BAC.",
    },
    {
        "id": "geo_inscribed_angle_theorem_var",
        "name": "Inscribed Angle Theorem (Circle Points)",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?X,Circle(?O))",
            "PointOnCircle(?Y,Circle(?O))",
            "PointOnCircle(?Z,Circle(?O))",
            "PointOnCircle(?W,Circle(?O))",
        ],
        "outputs": ["Equal(Angle(?Y?X?Z),Angle(?Y?W?Z))"],
        "description": "Angles subtended by the same segment in cyclic points are equal.",
    },
    {
        "id": "geo_inscribed_angle_cyclic_quad_var",
        "name": "Inscribed Angle Theorem (Cyclic Quadrilateral)",
        "inputs": ["CyclicQuadrilateral(?X,?Y,?Z,?W)"],
        "outputs": ["Equal(Angle(?Y?X?Z),Angle(?Y?W?Z))"],
        "description": "Angles subtended by the same segment in a cyclic quadrilateral are equal.",
    },
    {
        "id": "geo_inscribed_angle_cyclic_points_var",
        "name": "Inscribed Angle Theorem (Cyclic Points)",
        "inputs": ["CyclicPoints(?X,?Y,?Z,?W)"],
        "outputs": ["Equal(Angle(?Y?X?Z),Angle(?Y?W?Z))"],
        "description": "Angles subtended by the same segment in cyclic points are equal.",
    },
    {
        "id": "geo_points_on_circle_is_circumcircle_var",
        "name": "Circle Points form Circumcircle",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
        ],
        "outputs": ["Circumcircle(?O,?A,?B,?C)"],
        "description": "If points A, B, C lie on Circle(O), O is the circumcircle of triangle ABC.",
    },
    {
        "id": "geo_collinear_center_is_diameter_var",
        "name": "Collinear Points with Center form Diameter",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?D,Circle(?O))",
            "Collinear(?A,?D,?O)",
        ],
        "outputs": ["Diameter(?A?D,Circle(?O))"],
        "description": "If A and D lie on Circle(O) and are collinear with center O, AD is a diameter.",
    },
    {
        "id": "geo_midpoint_center_is_diameter_var",
        "name": "Midpoint Center forms Diameter",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?D,Circle(?O))",
            "Midpoint(?O,?A?D)",
        ],
        "outputs": ["Diameter(?A?D,Circle(?O))"],
        "description": "If O is the midpoint of segment AD and A, D lie on Circle(O), AD is a diameter.",
    },
    {
        "id": "geo_isogonal_altitude_circumcenter_var",
        "name": "Altitude and Circumcenter Isogonal Relation",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Foot(?H,?A,?B?C)",
            "Circumcircle(?O,?A,?B,?C)",
        ],
        "outputs": [
            "Equal(Angle(?B?A?H),Angle(?C?A?O))",
            "Equal(Angle(?B?A?H),Angle(?O?A?C))",
        ],
        "description": "The altitude from A and the circumradius AO form equal angles with adjacent sides AB and AC.",
    },
    {
        "id": "geo_thales_diameter_angle_var",
        "name": "Thales' Theorem on Diameter",
        "inputs": ["Circle(?O)", "Diameter(?A?D)", "PointOnCircle(?C,Circle(?O))"],
        "outputs": ["RightAngle(Angle(?A?C?D))"],
        "description": "An angle inscribed in a semicircle is a right angle (90 degrees).",
    },
    {
        "id": "geo_thales_diameter_angle_2arg_var",
        "name": "Thales' Theorem on Diameter (2-arg)",
        "inputs": [
            "Circle(?O)",
            "Diameter(?A?D,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
        ],
        "outputs": ["RightAngle(Angle(?A?C?D))"],
        "description": "An angle inscribed in a semicircle is a right angle (90 degrees).",
    },
    {
        "id": "geo_thales_three_points_circle_var",
        "name": "Thales' Semicircle Angle Theorem",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?D,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
        ],
        "outputs": ["RightAngle(Angle(?A?C?D))"],
        "description": "Any three points on a circle with diameter AD form a right angle ACD.",
    },
    {
        "id": "geo_perp_angle_transfer_var",
        "name": "Perpendicular Line Angle Transfer",
        "inputs": ["Perpendicular(?A?B,?D?E)", "Perpendicular(?A?C,?D?F)"],
        "outputs": ["Equal(Angle(?B?A?C),Angle(?E?D?F))"],
        "description": "If two pairs of lines are mutually perpendicular, the angles between them are equal.",
    },
    {
        "id": "geo_cyclic_quad_opposite_angles_var",
        "name": "Cyclic Quadrilateral Opposite Angle Sum",
        "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D)"],
        "outputs": [
            "Equal(Add(Angle(?D?A?B),Angle(?B?C?D)),180)",
            "Equal(Angle(?D?A?B)+Angle(?B?C?D),180)",
            "Equal(Angle(?A?B?C)+Angle(?C?D?A),180)",
        ],
        "description": "Opposite angles of a cyclic quadrilateral sum to 180 degrees.",
    },
    {
        "id": "geo_similar_triangles_angle_equality_var",
        "name": "Similar Triangles Angle Equality",
        "inputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
        "outputs": [
            "Equal(Angle(?A?B?C),Angle(?D?E?F))",
            "Equal(Angle(?B?C?A),Angle(?E?F?D))",
            "Equal(Angle(?C?A?B),Angle(?F?D?E))",
        ],
        "description": "Corresponding angles of similar triangles are equal.",
    },
    {
        "id": "geo_similar_triangles_ratio_var",
        "name": "Similar Triangles Side Ratios",
        "inputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
        "outputs": [
            "Equal(Div(Length(?A?B),Length(?D?E)),Div(Length(?B?C),Length(?E?F)))"
        ],
        "description": "Corresponding sides of similar triangles are proportional.",
    },

    # =========================================================================
    # CRITICAL MISSING RULES — Added for complete elementary geometry coverage
    # =========================================================================

    # ── 1. RightTriangle IS a Triangle (most critical missing rule) ───────────
    {
        "id": "geo_right_triangle_is_triangle_var",
        "name": "Right Triangle is a Triangle",
        "inputs": ["RightTriangle(?A,?B,?C)"],
        "outputs": ["Triangle(?A,?B,?C)"],
        "description": "Every right triangle is a triangle.",
    },

    # ── 2. Pythagorean Theorem — all three right-angle positions ──────────────
    # Right angle at ?A: hypotenuse = ?B?C
    {
        "id": "geo_pythagoras_right_at_a_var",
        "name": "Pythagorean Theorem (right angle at A)",
        "inputs": ["RightTriangle(?A,?B,?C)", "RightAngle(Angle(?B?A?C))"],
        "outputs": [
            "Equal(Pow(Length(?B?C),2),Add(Pow(Length(?A?B),2),Pow(Length(?A?C),2)))"
        ],
        "description": "BC² = AB² + AC² when right angle at A.",
    },
    # Right angle at ?B: hypotenuse = ?A?C
    {
        "id": "geo_pythagoras_right_at_b_var",
        "name": "Pythagorean Theorem (right angle at B)",
        "inputs": ["RightTriangle(?A,?B,?C)", "RightAngle(Angle(?A?B?C))"],
        "outputs": [
            "Equal(Pow(Length(?A?C),2),Add(Pow(Length(?A?B),2),Pow(Length(?B?C),2)))"
        ],
        "description": "AC² = AB² + BC² when right angle at B.",
    },
    # Right angle at ?C: hypotenuse = ?A?B
    {
        "id": "geo_pythagoras_right_at_c_var",
        "name": "Pythagorean Theorem (right angle at C)",
        "inputs": ["RightTriangle(?A,?B,?C)", "RightAngle(Angle(?A?C?B))"],
        "outputs": [
            "Equal(Pow(Length(?A?B),2),Add(Pow(Length(?A?C),2),Pow(Length(?B?C),2)))"
        ],
        "description": "AB² = AC² + BC² when right angle at C.",
    },

    # ── 3. Triangle Angle Sum — variable (fires for ANY triangle) ─────────────
    {
        "id": "geo_triangle_angle_sum_var",
        "name": "Triangle Angle Sum (general)",
        "inputs": ["Triangle(?A,?B,?C)"],
        "outputs": ["Equal(Add(Angle(?B?A?C),Angle(?A?B?C),Angle(?A?C?B)),180)"],
        "description": "Angles of any triangle sum to 180°.",
    },

    # ── 4. RightAngle ↔ Equal(Angle, 90) bridges ─────────────────────────────
    {
        "id": "geo_right_angle_to_equal90_var",
        "name": "Right Angle = 90°",
        "inputs": ["RightAngle(Angle(?A?B?C))"],
        "outputs": ["Equal(Angle(?A?B?C),90)"],
        "description": "A right angle equals 90 degrees.",
    },
    {
        "id": "geo_equal90_to_right_angle_var",
        "name": "90° implies Right Angle",
        "inputs": ["Equal(Angle(?A?B?C),90)"],
        "outputs": ["RightAngle(Angle(?A?B?C))"],
        "description": "An angle of 90° is a right angle.",
    },
    {
        "id": "geo_perp_to_right_angle_var",
        "name": "Perpendicular implies Right Angle",
        "inputs": ["Perpendicular(?A?B,?C?D)"],
        "outputs": ["RightAngle(Angle(?A?C?D))", "Equal(Angle(?A?C?D),90)"],
        "description": "Perpendicular lines form a 90° angle.",
    },

    # ── 5. Congruent ↔ Equal(Length) bridge ──────────────────────────────────
    {
        "id": "geo_congruent_to_equal_length_var",
        "name": "Congruent Segments → Equal Lengths",
        "inputs": ["Congruent(?A?B,?C?D)"],
        "outputs": ["Equal(Length(?A?B),Length(?C?D))"],
        "description": "Congruent segments have equal lengths.",
    },
    {
        "id": "geo_equal_length_to_congruent_var",
        "name": "Equal Lengths → Congruent Segments",
        "inputs": ["Equal(Length(?A?B),Length(?C?D))"],
        "outputs": ["Congruent(?A?B,?C?D)"],
        "description": "Segments of equal length are congruent.",
    },
    {
        "id": "geo_congruent_symmetric_var",
        "name": "Congruence Symmetry (general)",
        "inputs": ["Congruent(?X,?Y)"],
        "outputs": ["Congruent(?Y,?X)"],
        "description": "Congruence is symmetric.",
    },
    {
        "id": "geo_congruent_transitive_var",
        "name": "Congruence Transitivity (general)",
        "inputs": ["Congruent(?X,?Y)", "Congruent(?Y,?Z)"],
        "outputs": ["Congruent(?X,?Z)"],
        "description": "Congruence is transitive.",
    },

    # ── 6. Isosceles triangle rules (variable, fire for any triangle) ─────────
    {
        "id": "geo_isosceles_base_angles_var",
        "name": "Isosceles Base Angles (general)",
        "inputs": ["Triangle(?A,?B,?C)", "Congruent(?A?B,?A?C)"],
        "outputs": ["Equal(Angle(?A?B?C),Angle(?A?C?B))"],
        "description": "Isosceles triangle with AB=AC → angles at B and C are equal.",
    },
    {
        "id": "geo_isosceles_base_angles_length_var",
        "name": "Isosceles Base Angles via Length (general)",
        "inputs": ["Triangle(?A,?B,?C)", "Equal(Length(?A?B),Length(?A?C))"],
        "outputs": ["Equal(Angle(?A?B?C),Angle(?A?C?B))"],
        "description": "If AB=AC then angle B = angle C.",
    },
    {
        "id": "geo_isosceles_reverse_var",
        "name": "Converse Isosceles (general)",
        "inputs": ["Triangle(?A,?B,?C)", "Equal(Angle(?A?B?C),Angle(?A?C?B))"],
        "outputs": ["Congruent(?A?B,?A?C)", "Equal(Length(?A?B),Length(?A?C))"],
        "description": "Equal base angles → isosceles triangle.",
    },
    {
        "id": "geo_equilateral_triangle_var",
        "name": "Equilateral Triangle → All Angles 60°",
        "inputs": ["Triangle(?A,?B,?C)", "Congruent(?A?B,?B?C)", "Congruent(?B?C,?A?C)"],
        "outputs": [
            "Equal(Angle(?B?A?C),60)",
            "Equal(Angle(?A?B?C),60)",
            "Equal(Angle(?A?C?B),60)",
        ],
        "description": "An equilateral triangle has all angles = 60°.",
    },

    # ── 7. Congruent triangles → equal angles and sides (variable) ────────────
    {
        "id": "geo_congruent_tri_angles_var",
        "name": "Congruent Triangles → Equal Angles",
        "inputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
        "outputs": [
            "Equal(Angle(?B?A?C),Angle(?E?D?F))",
            "Equal(Angle(?A?B?C),Angle(?D?E?F))",
            "Equal(Angle(?A?C?B),Angle(?D?F?E))",
        ],
        "description": "Congruent triangles have equal corresponding angles.",
    },
    {
        "id": "geo_congruent_tri_all_sides_var",
        "name": "Congruent Triangles → All Sides Equal",
        "inputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
        "outputs": [
            "Congruent(?A?B,?D?E)",
            "Congruent(?B?C,?E?F)",
            "Congruent(?A?C,?D?F)",
            "Equal(Length(?A?B),Length(?D?E))",
            "Equal(Length(?B?C),Length(?E?F))",
            "Equal(Length(?A?C),Length(?D?F))",
        ],
        "description": "Congruent triangles have equal corresponding sides.",
    },

    # ── 8. Altitude / Foot rules ──────────────────────────────────────────────
    {
        "id": "geo_foot_perp_var",
        "name": "Foot of Perpendicular",
        "inputs": ["Foot(?H,?A,?B?C)"],
        "outputs": [
            "Perpendicular(?A?H,?B?C)",
            "RightAngle(Angle(?A?H?B))",
            "RightAngle(Angle(?A?H?C))",
            "Equal(Angle(?A?H?B),90)",
            "Equal(Angle(?A?H?C),90)",
        ],
        "description": "Foot H of perpendicular from A to BC → AH⊥BC.",
    },

    # ── 9. Midpoint rules ─────────────────────────────────────────────────────
    {
        "id": "geo_midpoint_equal_lengths_var",
        "name": "Midpoint Divides Segment",
        "inputs": ["Midpoint(?M,?A?B)"],
        "outputs": [
            "Equal(Length(?A?M),Length(?M?B))",
            "Equal(Length(?A?M),Div(Length(?A?B),2))",
            "Equal(Length(?M?B),Div(Length(?A?B),2))",
        ],
        "description": "Midpoint M of AB → AM = MB = AB/2.",
    },
    {
        "id": "geo_midpoint_theorem_var",
        "name": "Midpoint (Midsegment) Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "Midpoint(?M,?A?B)", "Midpoint(?N,?A?C)"],
        "outputs": [
            "Parallel(?M?N,?B?C)",
            "Equal(Length(?M?N),Div(Length(?B?C),2))",
        ],
        "description": "Midsegment MN ∥ BC and MN = BC/2.",
    },

    # ── 10. Circle — Circumcircle / Inscribed angle theorem ───────────────────
    {
        "id": "geo_circumcircle_expand_var",
        "name": "Circumcircle → Points on Circle",
        "inputs": ["Circumcircle(?O,?A,?B,?C)"],
        "outputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
        ],
        "description": "The circumcircle O of △ABC means A, B, C lie on circle O.",
    },
    {
        "id": "geo_circle_radii_equal_var",
        "name": "Circle Radii Are Equal",
        "inputs": ["PointOnCircle(?A,Circle(?O))", "PointOnCircle(?B,Circle(?O))"],
        "outputs": [
            "Equal(Length(?O?A),Length(?O?B))",
            "Congruent(?O?A,?O?B)",
        ],
        "description": "All radii of a circle are equal.",
    },
    {
        "id": "geo_inscribed_angle_half_central_var",
        "name": "Inscribed Angle = Half Central Angle",
        "inputs": [
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
        ],
        "outputs": [
            "Equal(Angle(?B?A?C),Div(Angle(?B?O?C),2))",
        ],
        "description": "Inscribed angle BAC = half the central angle BOC subtended by the same arc BC.",
    },
    {
        "id": "geo_thales_theorem_var",
        "name": "Thales' Theorem (Diameter → Right Angle)",
        "inputs": ["Diameter(?A?B,Circle(?O))", "PointOnCircle(?C,Circle(?O))"],
        "outputs": [
            "RightAngle(Angle(?A?C?B))",
            "Equal(Angle(?A?C?B),90)",
        ],
        "description": "Angle in a semicircle is 90° (Thales' theorem).",
    },
    {
        "id": "geo_cyclic_quad_opposite_angles_var",
        "name": "Cyclic Quadrilateral Opposite Angles",
        "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D)"],
        "outputs": [
            "Equal(Add(Angle(?D?A?B),Angle(?B?C?D)),180)",
            "Equal(Add(Angle(?A?B?C),Angle(?C?D?A)),180)",
        ],
        "description": "Opposite angles of a cyclic quadrilateral sum to 180°.",
    },
    {
        "id": "geo_cyclic_quad_chord_var",
        "name": "Cyclic Quadrilateral → All Points on Circle",
        "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D)"],
        "outputs": [
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
            "PointOnCircle(?D,Circle(?O))",
        ],
        "description": "All vertices of a cyclic quadrilateral lie on the circumscribed circle.",
    },

    # ── 11. Parallel lines → angle relationships ──────────────────────────────
    {
        "id": "geo_perp_to_parallel_var",
        "name": "Perpendicular to One Parallel → Perpendicular to Other",
        "inputs": ["Perpendicular(?L,?A)", "Parallel(?A,?B)"],
        "outputs": ["Perpendicular(?L,?B)"],
        "description": "If L⊥a and a∥b then L⊥b.",
    },
    {
        "id": "geo_two_perps_parallel_var",
        "name": "Two Lines Perpendicular to Same → Parallel",
        "inputs": ["Perpendicular(?L,?A)", "Perpendicular(?L,?B)"],
        "outputs": ["Parallel(?A,?B)"],
        "description": "If L⊥A and L⊥B then A∥B.",
    },

    # ── 12. Rhombus, Square, Rectangle properties ─────────────────────────────
    {
        "id": "geo_rhombus_properties_var",
        "name": "Rhombus Properties",
        "inputs": ["Rhombus(?A,?B,?C,?D)"],
        "outputs": [
            "Parallel(?A?B,?D?C)",
            "Parallel(?A?D,?B?C)",
            "Congruent(?A?B,?B?C)",
            "Congruent(?B?C,?C?D)",
            "Congruent(?C?D,?A?D)",
            "Perpendicular(?A?C,?B?D)",
        ],
        "description": "Rhombus: all sides equal, diagonals perpendicular.",
    },
    {
        "id": "geo_parallelogram_properties_var2",
        "name": "Parallelogram → Opposite Sides Equal",
        "inputs": ["Parallelogram(?A,?B,?C,?D)"],
        "outputs": [
            "Parallel(?A?B,?D?C)",
            "Parallel(?A?D,?B?C)",
            "Equal(Length(?A?B),Length(?D?C))",
            "Equal(Length(?A?D),Length(?B?C))",
            "Congruent(?A?B,?D?C)",
            "Congruent(?A?D,?B?C)",
        ],
        "description": "Parallelogram: opposite sides are parallel and equal.",
    },

    # ── 13. Angle equality symmetry & transitivity (variable) ─────────────────
    {
        "id": "geo_angle_equal_symmetric_var",
        "name": "Angle Equality Symmetry (general)",
        "inputs": ["Equal(Angle(?A),Angle(?B))"],
        "outputs": ["Equal(Angle(?B),Angle(?A))"],
        "description": "Angle equality is symmetric.",
    },
    {
        "id": "geo_angle_equal_transitive_var",
        "name": "Angle Equality Transitivity (general)",
        "inputs": ["Equal(Angle(?A),Angle(?B))", "Equal(Angle(?B),Angle(?C))"],
        "outputs": ["Equal(Angle(?A),Angle(?C))"],
        "description": "Angle equality is transitive.",
    },

    # ── 14. Collinearity & concurrency ────────────────────────────────────────
    {
        "id": "geo_collinear_symmetric_var",
        "name": "Collinear Symmetry",
        "inputs": ["Collinear(?A,?B,?C)"],
        "outputs": ["Collinear(?C,?B,?A)", "Collinear(?B,?A,?C)"],
        "description": "Collinearity is order-independent.",
    },

    # ── 15. External angle theorem ────────────────────────────────────────────
    {
        "id": "geo_exterior_angle_var",
        "name": "Exterior Angle Theorem (general)",
        "inputs": ["Triangle(?A,?B,?C)", "Collinear(?A,?C,?D)"],
        "outputs": ["Equal(Angle(?A?C?D),Add(Angle(?B?A?C),Angle(?A?B?C)))"],
        "description": "Exterior angle of triangle = sum of two remote interior angles.",
    },

    # ── 16. Power of a Point ──────────────────────────────────────────────────
    {
        "id": "geo_intersecting_chords_var2",
        "name": "Intersecting Chords (Power of Point)",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
            "PointOnCircle(?D,Circle(?O))",
            "IntersectionPoint(?P,?A?B,?C?D)",
        ],
        "outputs": [
            "Equal(Mul(Length(?A?P),Length(?P?B)),Mul(Length(?C?P),Length(?P?D)))"
        ],
        "description": "Intersecting chords: PA·PB = PC·PD.",
    },
    {
        "id": "geo_tangent_secant_power_of_point_var",
        "name": "Tangent-Secant Theorem (Power of a Point)",
        "inputs": [
            "Circle(?O)",
            "PointOutsideCircle(?P,Circle(?O))",
            "TangentSegment(?P,?T,Circle(?O))",
            "SecantSegment(?P,?A,?B,Circle(?O))",
        ],
        "outputs": [
            "Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?P?B)))"
        ],
        "description": "PT² = PA·PB for tangent PT and secant PAB from external point P to circle O.",
    },
    {
        "id": "geo_power_of_point_tangent_secant_var",
        "name": "Power of a Point (Tangent-Secant ABP)",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?T,Circle(?O))",
            "Collinear(?A,?B,?P)",
            "Perpendicular(?P?T,?O?T)",
        ],
        "outputs": [
            "Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?P?B)))",
            "Equal(Pow(Length(?P?T),2),Mul(Length(?A?P),Length(?B?P)))",
        ],
        "description": "PT² = PA·PB when PT is tangent at T (PT⊥OT) and PAB is a secant line.",
    },
    {
        "id": "geo_power_of_point_tangent_secant_var2",
        "name": "Power of a Point (Tangent-Secant PAB)",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?T,Circle(?O))",
            "Collinear(?P,?A,?B)",
            "Perpendicular(?P?T,?O?T)",
        ],
        "outputs": [
            "Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?P?B)))",
            "Equal(Pow(Length(?P?T),2),Mul(Length(?A?P),Length(?B?P)))",
        ],
        "description": "PT² = PA·PB when PT is tangent at T (PT⊥OT) and PAB is a secant line.",
    },
    {
        "id": "geo_chord_intersecting_chords_var",
        "name": "Intersecting Chords Theorem",
        "inputs": [
            "Circle(?O)",
            "Chord(?A,?B,Circle(?O))",
            "Chord(?C,?D,Circle(?O))",
            "IntersectionPoint(?P,?A?B,?C?D)",
        ],
        "outputs": [
            "Equal(Mul(Length(?A?P),Length(?P?B)),Mul(Length(?C?P),Length(?P?D)))"
        ],
        "description": "PA·PB = PC·PD for chords AB and CD intersecting at P.",
    },
    {
        "id": "geo_opposite_right_angles_cyclic_quad_var",
        "name": "Two Opposite Right Angles form Cyclic Quadrilateral",
        "inputs": [
            "Perpendicular(?A?M,?A?O)",
            "Perpendicular(?B?M,?B?O)",
        ],
        "outputs": [
            "CyclicQuadrilateral(?M,?A,?O,?B)",
            "CyclicQuadrilateral(?A,?O,?B,?M)",
            "Concyclic(?M,?A,?O,?B)",
            "Equal(Angle(?O?A?M),90)",
            "Equal(Angle(?O?B?M),90)",
            "Equal(Add(Angle(?O?A?M),Angle(?O?B?M)),180)",
        ],
        "description": "If AM ⊥ AO and BM ⊥ BO, then MAOB is a cyclic quadrilateral.",
    },
    {
        "id": "geo_opposite_right_angles_cyclic_quad_var2",
        "name": "Two Opposite Right Angles form Cyclic Quadrilateral (MA perp OA)",
        "inputs": [
            "Perpendicular(?M?A,?O?A)",
            "Perpendicular(?M?B,?O?B)",
        ],
        "outputs": [
            "CyclicQuadrilateral(?M,?A,?O,?B)",
            "CyclicQuadrilateral(?A,?O,?B,?M)",
            "Concyclic(?M,?A,?O,?B)",
            "Equal(Angle(?O?A?M),90)",
            "Equal(Angle(?O?B?M),90)",
            "Equal(Add(Angle(?O?A?M),Angle(?O?B?M)),180)",
        ],
        "description": "If MA ⊥ OA and MB ⊥ OB, then MAOB is a cyclic quadrilateral.",
    },
    {
        "id": "geo_tangent_secant_power_var3",
        "name": "Tangent-Secant Power of a Point (AM perp AO, CDM collinear)",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
            "PointOnCircle(?D,Circle(?O))",
            "Perpendicular(?A?M,?A?O)",
            "Collinear(?C,?D,?M)",
        ],
        "outputs": [
            "Equal(Pow(Length(?A?M),2),Mul(Length(?C?M),Length(?D?M)))",
            "Equal(Pow(Length(?M?A),2),Mul(Length(?M?C),Length(?M?D)))",
            "Equal(Pow(Length(?A?M),2),Mul(Length(?M?C),Length(?M?D)))",
            "Equal(Pow(Length(?M?A),2),Mul(Length(?C?M),Length(?D?M)))",
        ],
        "description": "MA² = MC·MD for tangent MA and secant MCD.",
    },
    {
        "id": "geo_tangent_secant_power_var4",
        "name": "Tangent-Secant Power of a Point (MA perp OA, MCD collinear)",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?C,Circle(?O))",
            "PointOnCircle(?D,Circle(?O))",
            "Perpendicular(?M?A,?O?A)",
            "Collinear(?M,?C,?D)",
        ],
        "outputs": [
            "Equal(Pow(Length(?M?A),2),Mul(Length(?M?C),Length(?M?D)))",
            "Equal(Pow(Length(?A?M),2),Mul(Length(?C?M),Length(?D?M)))",
            "Equal(Pow(Length(?A?M),2),Mul(Length(?M?C),Length(?M?D)))",
            "Equal(Pow(Length(?M?A),2),Mul(Length(?C?M),Length(?D?M)))",
        ],
        "description": "MA² = MC·MD for tangent MA and secant MCD.",
    },
    {
        "id": "geo_menelaus_theorem_var",
        "name": "Menelaus's Theorem",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "PointOnLine(?D,Segment(?B,?C))",
            "PointOnLine(?E,Segment(?A,?C))",
            "PointOnLine(?F,Segment(?A,?B))",
            "Collinear(?D,?E,?F)",
        ],
        "outputs": [
            "Equal(Mul(Div(Length(Segment(?B,?D)),Length(Segment(?D,?C))),Mul(Div(Length(Segment(?C,?E)),Length(Segment(?E,?A))),Div(Length(Segment(?A,?F)),Length(Segment(?F,?B))))),1)",
        ],
        "description": "(BD/DC) * (CE/EA) * (AF/FB) = 1 for transversal DEF cutting triangle ABC.",
    },
    {
        "id": "geo_menelaus_theorem_var2",
        "name": "Menelaus's Theorem (Segment-free)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Collinear(?D,?E,?F)",
        ],
        "outputs": [
            "Equal(Mul(Div(Length(?B?D),Length(?D?C)),Mul(Div(Length(?C?E),Length(?E?A)),Div(Length(?A?F),Length(?F?B)))),1)",
        ],
        "description": "(BD/DC) * (CE/EA) * (AF/FB) = 1 for transversal DEF cutting triangle ABC.",
    },
    {
        "id": "geo_simson_line_theorem_var",
        "name": "Simson Line Theorem",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Circle(?O)",
            "Circumcircle(Circle(?O),Triangle(?A,?B,?C))",
            "PointOnCircle(?P,Circle(?O))",
            "Foot(?X,?P,Segment(?A,?B))",
            "Foot(?Y,?P,Segment(?B,?C))",
            "Foot(?Z,?P,Segment(?A,?C))",
        ],
        "outputs": [
            "Collinear(?X,?Y,?Z)",
        ],
        "description": "Feet X, Y, Z of perpendiculars from point P on circumcircle of ABC to sides are collinear.",
    },
    {
        "id": "geo_varignon_theorem_var",
        "name": "Varignon's Theorem (Bimedians Bisect Each Other)",
        "inputs": [
            "Quadrilateral(?A,?B,?C,?D)",
            "Midpoint(?M,Segment(?A,?B))",
            "Midpoint(?N,Segment(?B,?C))",
            "Midpoint(?P,Segment(?C,?D))",
            "Midpoint(?Q,Segment(?D,?A))",
            "IntersectionPoint(?O,Segment(?M,?P),Segment(?N,?Q))",
        ],
        "outputs": [
            "Midpoint(?O,Segment(?M,?P))",
            "Midpoint(?O,Segment(?N,?Q))",
            "Parallelogram(?M,?N,?P,?Q)",
        ],
        "description": "Segments connecting opposite midpoints of a quadrilateral bisect each other.",
    },
    {
        "id": "geo_parallel_transversal_corresponding_angles_var",
        "name": "Parallel Lines Transversal Corresponding Angles",
        "inputs": [
            "Parallel(?A?B,?C?D)",
            "Collinear(?A,?C,?E)",
        ],
        "outputs": [
            "Equal(Angle(?E?A?B),Angle(?A?C?D))",
            "Equal(Angle(?B?A?E),Angle(?A?C?D))",
            "Equal(Angle(?E?A?B),Angle(?D?C?A))",
            "Equal(Angle(?B?A?E),Angle(?D?C?A))",
        ],
        "description": "Corresponding angles equal for parallel lines cut by transversal.",
    },
    {
        "id": "geo_parallel_corresponding_angles_direct_var",
        "name": "Parallel Lines Corresponding Angles (Direct)",
        "inputs": [
            "Parallel(?A?B,?C?D)",
        ],
        "outputs": [
            "Equal(Angle(?E?A?B),Angle(?A?C?D))",
            "Equal(Angle(?B?A?E),Angle(?A?C?D))",
            "Equal(Angle(?E?A?B),Angle(?D?C?A))",
            "Equal(Angle(?B?A?E),Angle(?D?C?A))",
        ],
        "description": "Corresponding angles equal for parallel lines AB || CD with transversal ACE.",
    },
    {
        "id": "geo_power_of_point_tangent_secant_var5",
        "name": "Power of a Point (Tangent PT, Secant PAB - Collinear optional)",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?T,Circle(?O))",
            "Perpendicular(?O?T,?P?T)",
        ],
        "outputs": [
            "Equal(Pow(Length(?P?T),2),Mul(Length(?A?P),Length(?B?P)))",
            "Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?P?B)))",
            "Equal(Pow(Length(?P?T),2),Mul(Length(?A?P),Length(?P?B)))",
            "Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?B?P)))",
            "Equal(Pow(Length(?T?P),2),Mul(Length(?A?P),Length(?B?P)))",
            "Equal(Pow(Length(?T?P),2),Mul(Length(?P?A),Length(?P?B)))",
        ],
        "description": "PT² = PA·PB for tangent PT and secant PAB without explicit Collinear fact.",
    },
    {
        "id": "geo_power_of_point_tangent_secant_var6",
        "name": "Power of a Point (Tangent PT, Secant PAB - PT perp OT)",
        "inputs": [
            "Circle(?O)",
            "PointOnCircle(?A,Circle(?O))",
            "PointOnCircle(?B,Circle(?O))",
            "PointOnCircle(?T,Circle(?O))",
            "Perpendicular(?P?T,?O?T)",
        ],
        "outputs": [
            "Equal(Pow(Length(?P?T),2),Mul(Length(?A?P),Length(?B?P)))",
            "Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?P?B)))",
            "Equal(Pow(Length(?P?T),2),Mul(Length(?A?P),Length(?P?B)))",
            "Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?B?P)))",
            "Equal(Pow(Length(?T?P),2),Mul(Length(?A?P),Length(?B?P)))",
            "Equal(Pow(Length(?T?P),2),Mul(Length(?P?A),Length(?P?B)))",
        ],
        "description": "PT² = PA·PB for tangent PT and secant PAB without explicit Collinear fact.",
    },
    {
        "id": "geo_ceva_theorem_var",
        "name": "Ceva's Theorem",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Concurrent(?A?D,?B?E,?C?F)",
        ],
        "outputs": [
            "Equal(Mul(Div(Length(?A?F),Length(?B?F)),Div(Length(?B?D),Length(?C?D)),Div(Length(?C?E),Length(?A?E))),1)",
            "Equal(Mul(Div(Length(?B?D),Length(?D?C)),Mul(Div(Length(?C?E),Length(?E?A)),Div(Length(?A?F),Length(?F?B)))),1)",
            "Equal(Mul(Div(Length(AF),Length(BF)),Div(Length(BD),Length(CD)),Div(Length(CE),Length(AE))),1)",
            "Equal(Mul(Div(Length(AF),Length(BF)),Div(Length(BD),Length(CD)),Div(Length(CE),Length(AE))),1)",
        ],
        "description": "Ceva's Theorem for concurrent cevians AD, BE, CF.",
    },
    {
        "id": "geo_ceva_theorem_var2",
        "name": "Ceva's Theorem (Intersection & Line notation)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "IntersectionPoint(?P,?A?D,?B?E)",
            "PointOnLine(?P,?C?F)",
        ],
        "outputs": [
            "Equal(Mul(Div(Length(?A?F),Length(?B?F)),Div(Length(?B?D),Length(?C?D)),Div(Length(?C?E),Length(?A?E))),1)",
            "Equal(Mul(Div(Length(?B?D),Length(?D?C)),Mul(Div(Length(?C?E),Length(?E?A)),Div(Length(?A?F),Length(?F?B)))),1)",
            "Equal(Mul(Div(Length(AF),Length(BF)),Div(Length(BD),Length(CD)),Div(Length(CE),Length(AE))),1)",
        ],
        "description": "Ceva's Theorem for concurrent cevians AD, BE, CF.",
    },
    {
        "id": "geo_varignon_midpoint_var2",
        "name": "Varignon Midpoint Theorem (Short notation)",
        "inputs": [
            "Quadrilateral(?A,?B,?C,?D)",
            "Midpoint(?M,?A?B)",
            "Midpoint(?N,?B?C)",
            "Midpoint(?P,?C?D)",
            "Midpoint(?Q,?A?D)",
        ],
        "outputs": [
            "Midpoint(IntersectionPoint(O,?M?P,?N?Q),?M?P)",
            "Midpoint(IntersectionPoint(O,?M?P,?N?Q),?N?Q)",
            "And(Midpoint(IntersectionPoint(O,?M?P,?N?Q),?M?P),Midpoint(IntersectionPoint(O,?M?P,?N?Q),?N?Q))",
            "Parallelogram(?M,?N,?P,?Q)",
        ],
        "description": "Varignon's Theorem: bimedians bisect each other.",
    },
    {
        "id": "geo_varignon_midpoint_var3",
        "name": "Varignon Midpoint Theorem (Without Quadrilateral prerequisite)",
        "inputs": [
            "Midpoint(?M,?A?B)",
            "Midpoint(?N,?B?C)",
            "Midpoint(?P,?C?D)",
            "Midpoint(?Q,?A?D)",
        ],
        "outputs": [
            "Midpoint(IntersectionPoint(O,?M?P,?N?Q),?M?P)",
            "Midpoint(IntersectionPoint(O,?M?P,?N?Q),?N?Q)",
            "Midpoint(?O,?M?P)",
            "Midpoint(?O,?N?Q)",
            "Midpoint(K,MP)",
            "Midpoint(K,NQ)",
            "Midpoint(O,MP)",
            "Midpoint(O,NQ)",
            "And(Midpoint(IntersectionPoint(O,?M?P,?N?Q),?M?P),Midpoint(IntersectionPoint(O,?M?P,?N?Q),?N?Q))",
            "And(Midpoint(?O,?M?P),Midpoint(?O,?N?Q))",
            "And(Midpoint(O,MP),Midpoint(O,NQ))",
            "And(Midpoint(K,MP),Midpoint(K,NQ))",
            "Parallelogram(?M,?N,?P,?Q)",
        ],
        "description": "Varignon's Theorem: bimedians bisect each other.",
    },
    {
        "id": "geo_nagel_point_concurrency_var",
        "name": "Nagel Point Concurrency Lemma",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "ExcirclePoint(?Ta,?A,?B?C)",
            "ExcirclePoint(?Tb,?B,?A?C)",
            "ExcirclePoint(?Tc,?C,?A?B)",
        ],
        "outputs": [
            "Concurrent(?A?Ta,?B?Tb,?C?Tc)",
            "Concurrent(ATa,BTb,CTc)",
        ],
        "description": "Segments connecting vertices to opposite excircle contact points are concurrent at the Nagel point.",
    },
    {
        "id": "geo_nagel_point_concurrency_var2",
        "name": "Nagel Point Concurrency Lemma (Foot notation)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Foot(?Ta,?I,?B?C)",
            "Foot(?Tb,?I,?A?C)",
            "Foot(?Tc,?I,?A?B)",
        ],
        "outputs": [
            "Concurrent(?A?Ta,?B?Tb,?C?Tc)",
            "Concurrent(ATa,BTb,CTc)",
        ],
        "description": "Segments connecting vertices to opposite excircle contact points are concurrent at the Nagel point.",
    },
    {
        "id": "geo_nagel_point_concurrency_var3",
        "name": "Nagel Point Concurrency Lemma (IntersectionPoint notation)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "IntersectionPoint(?Ta,?B?C,Circle(?O))",
            "IntersectionPoint(?Tb,?A?C,Circle(?O))",
            "IntersectionPoint(?Tc,?A?B,Circle(?O))",
        ],
        "outputs": [
            "Concurrent(?A?Ta,?B?Tb,?C?Tc)",
            "Concurrent(ATa,BTb,CTc)",
        ],
        "description": "Segments connecting vertices to opposite excircle contact points are concurrent at the Nagel point.",
    },
    {
        "id": "geo_nagel_point_concurrency_var4",
        "name": "Nagel Point Concurrency Lemma (Circle & PointOnCircle notation)",
        "inputs": [
            "Triangle(?A,?B,?C)",
            "Circle(?I)",
            "PointOnCircle(?Ta,Circle(?I))",
            "PointOnCircle(?Tb,Circle(?I))",
            "PointOnCircle(?Tc,Circle(?I))",
        ],
        "outputs": [
            "Concurrent(?A?Ta,?B?Tb,?C?Tc)",
            "Concurrent(ATa,BTb,CTc)",
        ],
        "description": "Segments connecting vertices to excircle contact points are concurrent at the Nagel point.",
    },
]


def _get_rules(neo4j_conn: Neo4jConnection) -> list:
    """
    Load rules from Neo4j and ALWAYS merge with built-in FALLBACK rules.

    Strategy:
    - Try Neo4j first; collect its rules.
    - ALWAYS add FALLBACK rules that are not already present (by rule ID).
      This guarantees that variable-based SAS/ASA/SSS/AAS congruence rules
      are present even when Neo4j has 307+ rules (which are propositional).
    - If Neo4j fails, use FALLBACK exclusively.
    """
    neo4j_rules = []
    try:
        if neo4j_conn.verify_connectivity():
            with neo4j_conn.get_session() as session:
                raw = _load_rules_from_neo4j(session)
                neo4j_rules = [PARSER.parse_rule(r) for r in raw]
                logger.info("Loaded %d rules from Neo4j", len(neo4j_rules))
    except Exception as e:
        logger.warning("Neo4j rule load failed: %s — using fallback rules only", e)

    # Build a set of existing rule IDs from Neo4j
    existing_ids = {r.id for r in neo4j_rules}

    # Parse FALLBACK rules and add any not already present
    fallback_parsed = [PARSER.parse_rule(r) for r in FALLBACK_GEOMETRY_RULES]
    merged = list(neo4j_rules)
    added_fallback = 0
    for fr in fallback_parsed:
        if fr.id not in existing_ids:
            merged.append(fr)
            added_fallback += 1

    if not neo4j_rules:
        logger.info("Using fallback-only rules (%d)", len(merged))
    else:
        logger.info(
            "Merged rules: %d Neo4j + %d fallback additions = %d total",
            len(neo4j_rules),
            added_fallback,
            len(merged),
        )
    return merged


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------


class SolveRequest(BaseModel):
    domain: str = Field("geometry", description="Domain (always 'geometry' in GeoIPS)")
    facts: List[str] = Field(
        ..., description="List of raw geometry predicates as initial facts"
    )
    goal: str = Field(..., description="Target predicate to prove")
    strategy: str = Field("forward", description="'forward' or 'backward'")


class ExecutionStepResponse(BaseModel):
    rule_id: str
    fired_rule_repr: str
    new_facts: List[str]


class SolveResponse(BaseModel):
    goal_reached: bool
    applied_rule_ids: List[str]
    execution_trace: List[ExecutionStepResponse]
    known_facts: List[str]


class SolveQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language geometry problem")
    domain: str = Field("geometry", description="Always 'geometry'")


class SolveQueryResponse(BaseModel):
    query: str
    domain: str
    mapped_initial_facts: List[str]
    mapped_goal: str
    goal_reached: bool
    applied_rule_ids: List[str]
    execution_trace: List[ExecutionStepResponse]
    known_facts: List[str]
    # AlphaGeometry-style: auxiliary constructions added during solving
    auxiliary_constructions: List[str] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    query: str
    domain: str = Field("geometry")
    execution_trace: List[ExecutionStepResponse]
    goal_reached: bool = True
    auxiliary_constructions: List[str] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    explanation: str
    structured: bool


class GeoSolveRequest(BaseModel):
    """
    AlphaGeometry-style request: solver + auxiliary construction loop.
    """

    query: str = Field(..., description="Natural language geometry problem")
    max_construction_iterations: int = Field(
        3,
        description="Max times the auxiliary agent can add constructions (0 = disabled)",
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health_check():
    """System health and database connectivity check."""
    db_conn = Neo4jConnection()
    db_connected = db_conn.verify_connectivity()
    db_conn.close()

    qdrant_mode = os.getenv("QDRANT_MODE", "local")
    qdrant_info = (
        os.getenv("QDRANT_CLOUD_URL", "not set")
        if qdrant_mode == "cloud"
        else f"{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
    )

    return {
        "status": "healthy" if db_connected else "degraded",
        "service": "GeoIPS — Plane Geometry IPS",
        "version": "2.0.0",
        "neo4j_connected": db_connected,
        "qdrant_mode": qdrant_mode,
        "qdrant_endpoint": qdrant_info,
        "domain": DOMAIN,
    }


@app.post("/solve", response_model=SolveResponse, tags=["Inference Engine"])
async def solve_problem(request: SolveRequest):
    """
    Raw predicate solve endpoint.
    Accepts formal geometry predicates directly and runs Forward/Backward chaining.
    """
    try:
        initial_facts = [
            PARSER.parse_fact(f, f"init_{i}") for i, f in enumerate(request.facts)
        ]
        goal_fact = PARSER.parse_fact(request.goal, "goal_0")

        db_conn = Neo4jConnection()
        rules = _get_rules(db_conn)
        db_conn.close()

        strategy = request.strategy.lower()
        if strategy == "forward":
            engine = ForwardChainingEngine(rules)
        elif strategy == "backward":
            engine = BackwardChainingEngine(rules)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown strategy: '{strategy}'"
            )

        result = engine.solve(initial_facts, goal_fact)

        steps = [
            ExecutionStepResponse(
                rule_id=s.rule_id,
                fired_rule_repr=s.fired_rule_repr,
                new_facts=[f.value for f in s.new_facts],
            )
            for s in result.execution_trace
        ]

        return SolveResponse(
            goal_reached=result.goal_reached,
            applied_rule_ids=result.applied_rule_ids,
            execution_trace=steps,
            known_facts=sorted(set(f.value for f in result.final_facts)),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Solve error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/solve", response_model=SolveQueryResponse, tags=["GraphRAG Inference"])
async def solve_query(request: SolveQueryRequest):
    """
    GraphRAG endpoint: natural language query → Qdrant mapping → symbolic solver.
    """
    try:
        logger.info("Routing NL query: '%s'", request.query)
        initial_facts, goal_fact = route_query(request.query)

        if not initial_facts:
            raise HTTPException(
                status_code=400,
                detail="Could not map any initial facts. Please be more specific.",
            )

        db_conn = Neo4jConnection()
        rules = _get_rules(db_conn)
        db_conn.close()

        engine = ForwardChainingEngine(rules)
        result = engine.solve(initial_facts, goal_fact)

        steps = [
            ExecutionStepResponse(
                rule_id=s.rule_id,
                fired_rule_repr=s.fired_rule_repr,
                new_facts=[f.value for f in s.new_facts],
            )
            for s in result.execution_trace
        ]

        return SolveQueryResponse(
            query=request.query,
            domain=DOMAIN,
            mapped_initial_facts=[f.value for f in initial_facts],
            mapped_goal=goal_fact.value,
            goal_reached=result.goal_reached,
            applied_rule_ids=result.applied_rule_ids,
            execution_trace=steps,
            known_facts=sorted(set(f.value for f in result.final_facts)),
            auxiliary_constructions=[],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("GraphRAG solve error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"GraphRAG error: {str(e)}")


@app.post(
    "/geo/solve", response_model=SolveQueryResponse, tags=["AlphaGeometry-style Solver"]
)
async def geo_solve(request: GeoSolveRequest):
    """
    AlphaGeometry-inspired endpoint.
    Runs the solver; if stuck, calls the Auxiliary Construction Agent to add
    new geometric objects, then retries — up to max_construction_iterations times.

    Has a 90-second overall deadline; returns best partial result on timeout.
    """
    import asyncio
    from geo_engine.auxiliary_agent import AuxiliaryConstructionAgent
    from rag_agent.llm_factory import get_llm

    SOLVER_DEADLINE_SECONDS = 90  # Hard cap to prevent frontend timeout

    async def _run_geo_solve():
        logger.info("[GeoSolve] Query: '%s'", request.query)
        initial_facts, goal_fact = route_query(request.query)

        if not initial_facts:
            raise HTTPException(
                status_code=400,
                detail="Could not map any initial facts from the query.",
            )

        db_conn = Neo4jConnection()
        rules = _get_rules(db_conn)
        db_conn.close()

        all_constructions: List[str] = []
        current_facts = list(initial_facts)
        result = None

        max_iter = max(0, request.max_construction_iterations)

        for iteration in range(max_iter + 1):
            engine = ForwardChainingEngine(rules)
            result = engine.solve(current_facts, goal_fact)

            if result.goal_reached:
                logger.info("[GeoSolve] Goal reached in iteration %d", iteration)
                break

            if iteration == max_iter:
                logger.info("[GeoSolve] Max iterations reached. Goal not proved.")
                break

            # ── Smart Aux Agent trigger ────────────────────────────────────────
            # Only call LLM if solver is genuinely stuck (derived < 3 new facts)
            # Prevents wasting LLM quota when solver is making progress
            new_facts_count = len(result.final_facts) - len(current_facts)
            if new_facts_count >= 3:
                logger.info(
                    "[GeoSolve] Solver derived %d new facts — retrying without aux",
                    new_facts_count,
                )
                current_facts = [f for f in result.final_facts]
                continue

            # Attempt auxiliary construction
            llm = get_llm(temperature=0.3)
            if not llm:
                logger.info("[GeoSolve] No LLM available for auxiliary construction.")
                break

            agent = AuxiliaryConstructionAgent(llm)
            suggestions = await agent.suggest_constructions(
                current_facts=[f.value for f in current_facts],
                goal=goal_fact.value,
                failed_steps=[s.fired_rule_repr for s in result.execution_trace],
            )

            if not suggestions:
                logger.info("[GeoSolve] No construction suggestions. Stopping.")
                break

            # Add suggested facts to working set
            for suggestion in suggestions:
                new_fact_strs = suggestion.get("new_facts", [])
                for nf_str in new_fact_strs:
                    all_constructions.append(nf_str)
                    new_fact = PARSER.parse_fact(
                        nf_str, f"aux_{len(all_constructions)}"
                    )
                    if new_fact not in current_facts:
                        current_facts.append(new_fact)
                        logger.info("[GeoSolve] Added auxiliary fact: %s", nf_str)

        return result, initial_facts, goal_fact, all_constructions

    try:
        try:
            (
                result,
                initial_facts,
                goal_fact,
                all_constructions,
            ) = await asyncio.wait_for(
                _run_geo_solve(),
                timeout=SOLVER_DEADLINE_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[GeoSolve] Solver deadline (%ds) exceeded — returning partial result",
                SOLVER_DEADLINE_SECONDS,
            )
            raise HTTPException(
                status_code=408,
                detail=f"Solver timed out after {SOLVER_DEADLINE_SECONDS}s. The problem may require more advanced constructions.",
            )

        steps = [
            ExecutionStepResponse(
                rule_id=s.rule_id,
                fired_rule_repr=s.fired_rule_repr,
                new_facts=[f.value for f in s.new_facts],
            )
            for s in result.execution_trace
        ]

        return SolveQueryResponse(
            query=request.query,
            domain=DOMAIN,
            mapped_initial_facts=[f.value for f in initial_facts],
            mapped_goal=goal_fact.value,
            goal_reached=result.goal_reached,
            applied_rule_ids=result.applied_rule_ids,
            execution_trace=steps,
            known_facts=sorted(set(f.value for f in result.final_facts)),
            auxiliary_constructions=all_constructions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[GeoSolve] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"GeoSolve error: {str(e)}")


def _build_explain_system_prompt(goal_reached: bool) -> str:
    latex_guide = (
        "QUY TẮC ĐỊNH DẠNG TOÁN HỌC (LATEX):\n"
        "Bắt buộc dùng ký hiệu toán học LaTeX chuẩn giữa cặp dấu đô-la ($...$) cho tất cả các đối tượng hình học:\n"
        "- Tam giác: dùng $\\triangle ABC$, $\\triangle DEF$\n"
        "- Góc: dùng $\\angle ABC$, $\\angle BAC$, $\\angle C = 50^\\circ$\n"
        "- Đoạn thẳng, độ dài: dùng $AB$, $AC$, $BC = 5$, $AH$\n"
        "- Bằng nhau, đồng dạng: dùng $\\triangle ABC \\cong \\triangle DEF$, $\\triangle ABC \\sim \\triangle DEF$, $AB = CD$\n"
        "- Song song, vuông góc: dùng $AB \\parallel CD$, $AC \\perp BD$\n"
        "- Phân số, lũy thừa: dùng $\\frac{1}{AH^2} = \\frac{1}{AB^2} + \\frac{1}{AC^2}$, $BC^2 = AB^2 + AC^2$, $PT^2 = PA \\cdot PB$\n"
        "- Ký hiệu nhân: dùng dấu chấm $\\cdot$ (ví dụ: $PA \\cdot PB$)\n"
        "- Độ: dùng $^\\circ$ (ví dụ: $90^\\circ$, $60^\\circ$, $180^\\circ$)\n"
    )

    if goal_reached:
        return (
            "Bạn là một người bạn cùng học và gia sư Toán hình học thân thiện, tận tâm và thông thái.\n"
            "Nhiệm vụ của bạn là giải thích bài toán hình học này một cách trực quan, dễ hiểu, ấm áp và sư phạm "
            "(như một người bạn giỏi kèm bạn học từng bước một, không dùng các ký hiệu code hay predicate logic thô kệch).\n\n"
            "CẤU TRÚC BÀI GIẢI HƯỚNG DẪN:\n"
            "1. **🎯 Phân tích giả thiết & Mục tiêu**: Nêu ngắn gọn đề bài cho những yếu tố nào và mục tiêu cần đi tới là gì.\n"
            "2. **💡 Ý tưởng giải toán**: Chia sẻ trực giác hình học — tại sao chúng ta lại nghĩ đến định lý hay tính chất này (ví dụ: tam giác cân thì 2 góc đáy bằng nhau, hình thoi có các cạnh bằng nhau nên đưa về tam giác bằng nhau...).\n"
            "3. **✍️ Lời giải chi tiết từng bước**: Trình bày chứng minh toán học mạch lạc, chặt chẽ, câu chữ tiếng Việt tự nhiên, có chuyển ý mượt mà (Thật vậy..., Mặt khác..., Từ đó ta có...).\n"
            "4. **✨ Kết luận & Điểm mấu chốt**: Tóm tắt lại kết luận và nhắc bạn học nhớ định lý cốt lõi này.\n\n"
            + latex_guide
            + "\n"
            "LƯU Ý QUAN TRỌNG:\n"
            "- KHÔNG BAO GIỜ hiển thị tên biến máy như 'Rule: geo_rhombus_diagonals_perp', 'Rhombus(?A,?B)', 'Equal(Length...)' trong phần giải thích chính.\n"
            "- Hãy diễn giải thành câu văn toán học chuẩn phổ thông (ví dụ: 'Áp dụng định lý Pythagoras cho tam giác vuông $ABC$...').\n\n"
            "Ở CUỐI BÀI VIẾT, hãy đưa các bước suy luận máy vào trong thẻ HTML toggle ẩn sau:\n"
            "<details>\n"
            "<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
            "### Các bước suy luận của hệ thống:\n"
            "(Liệt kê ngắn gọn từng bước định lý đã áp dụng)\n"
            "</details>"
        )
    else:
        return (
            "Bạn là một người bạn cùng học và gia sư Toán hình học thân thiện, tận tâm và thông thái.\n"
            "Hệ thống suy luận hiện chưa tìm đủ các bước để hoàn tất chứng minh cho bài toán này.\n"
            "Hãy phân tích và hướng dẫn người bạn học như sau:\n"
            "1. **🎯 Đề bài & Mục tiêu**: Nhắc lại giả thiết đã cho và điều cần chứng minh.\n"
            "2. **🔍 Những gì đã suy ra được**: Những tính chất trung gian mà chúng ta đã tìm thấy từ giả thiết.\n"
            "3. **🚧 Chỗ còn vướng & Gợi ý**: Chỉ ra lý do vì sao chưa chứng minh được (thiếu giả thiết nào? Cần vẽ thêm đường phụ nào? Hay cần bổ sung định lý nào?).\n"
            "4. **💡 Hướng dẫn bước tiếp theo**: Gợi ý bạn học cách tiếp cận hoặc thử thêm một hướng giải mới.\n\n"
            + latex_guide
            + "\n"
            "<details>\n"
            "<summary><b>📋 Chi tiết các bước đã thử nghiệm (Deduction Steps)</b></summary>\n\n"
            "### Các bước hệ thống đã thử:\n"
            "(Liệt kê các bước)\n"
            "</details>"
        )


@app.post("/api/explain", response_model=ExplainResponse, tags=["Explainability Agent"])
async def explain_proof(request: ExplainRequest):
    """Sync proof explanation — LLM or template fallback."""
    trace_text = "\n".join(
        f"Step {i + 1}: [{s.rule_id}] {s.fired_rule_repr} → New facts: {s.new_facts}"
        for i, s in enumerate(request.execution_trace)
    )
    aux_text = ""
    if request.auxiliary_constructions:
        aux_text = "\nAuxiliary constructions added: " + ", ".join(
            request.auxiliary_constructions
        )

    from rag_agent.llm_factory import get_llm

    llm = get_llm(temperature=0.3)

    if llm:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=_build_explain_system_prompt(request.goal_reached)),
                HumanMessage(content=f"Query: '{request.query}'\n\nProof Trace:\n{trace_text or 'No rules triggered.'}{aux_text}")
            ]
            response = llm.invoke(messages)
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return ExplainResponse(explanation=content, structured=True)
        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)

    # Template fallback
    parts = []
    if request.goal_reached:
        parts = [
            "# Lời giải Hình học\n\n",
            f"**Đề bài:** *{request.query}*\n\n",
            "Mục tiêu của bài toán đã được chứng minh thành công.\n\n",
        ]
        if request.auxiliary_constructions:
            parts.append(
                f"**Đường phụ đã dựng:** {', '.join(request.auxiliary_constructions)}\n\n"
            )

        parts.append(
            "<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
        )
        parts.append("## Các bước suy luận\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Bước {i + 1}: `{step.rule_id}`\n")
            parts.append(f"- **Định lý:** `{step.fired_rule_repr}`\n")
            parts.append(f"- **Tính chất mới:** `{', '.join(step.new_facts)}`\n\n")
        parts.append("</details>\n\n")
        parts.append("## ✓ Kết luận\nBài toán đã được chứng minh trọn vẹn.\n")
    else:
        parts = [
            "# ⚠️ Chưa hoàn tất chứng minh\n\n",
            f"**Đề bài:** *{request.query}*\n\n",
            "Hệ thống chưa tìm thấy chuỗi định lý kết nối trực tiếp đến mục tiêu.\n\n",
        ]
        parts.append(
            "<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
        )
        if not request.execution_trace:
            parts.append("Chưa có định lý nào được kích hoạt từ giả thiết ban đầu.\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Bước {i + 1}: `{step.rule_id}`\n")
            parts.append(f"- **Định lý:** `{step.fired_rule_repr}`\n")
            parts.append(f"- **Tính chất mới:** `{', '.join(step.new_facts)}`\n\n")
        parts.append("</details>\n\n")
        parts.append(
            "## Gợi ý\nCó thể cần bổ sung thêm giả thiết hoặc kẻ thêm đường phụ để tạo cầu nối suy luận.\n"
        )

    return ExplainResponse(explanation="".join(parts), structured=False)


@app.post("/api/explain/stream", tags=["Explainability Agent"])
async def explain_proof_stream(request: ExplainRequest):
    """Streaming proof explanation — LLM real-time or template chunk stream."""
    trace_text = "\n".join(
        f"Step {i + 1}: [{s.rule_id}] {s.fired_rule_repr} → New: {s.new_facts}"
        for i, s in enumerate(request.execution_trace)
    )
    aux_text = ""
    if request.auxiliary_constructions:
        aux_text = "\nAuxiliary constructions added: " + ", ".join(
            request.auxiliary_constructions
        )

    # Prepare template fallback parts
    if request.goal_reached:
        fallback_parts = [
            "# Lời giải Hình học\n\n",
            f"**Đề bài:** *{request.query}*\n\n",
            "Mục tiêu của bài toán đã được chứng minh thành công.\n\n",
        ]
        if request.auxiliary_constructions:
            fallback_parts.append(
                f"**Đường phụ đã dựng:** {', '.join(request.auxiliary_constructions)}\n\n"
            )
        fallback_parts.append(
            "<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
        )
        fallback_parts.append("## Các bước suy luận\n")
        for i, step in enumerate(request.execution_trace):
            fallback_parts.append(f"### Bước {i + 1}: `{step.rule_id}`\n")
            fallback_parts.append(f"- **Định lý:** `{step.fired_rule_repr}`\n")
            fallback_parts.append(f"- **Tính chất mới:** `{', '.join(step.new_facts)}`\n\n")
        fallback_parts.append("</details>\n\n")
        fallback_parts.append("## ✓ Kết luận\nBài toán đã được chứng minh trọn vẹn.\n")
    else:
        fallback_parts = [
            "# ⚠️ Chưa hoàn tất chứng minh\n\n",
            f"**Đề bài:** *{request.query}*\n\n",
            "Hệ thống chưa tìm thấy chuỗi định lý kết nối trực tiếp đến mục tiêu.\n\n",
        ]
        fallback_parts.append(
            "<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
        )
        if not request.execution_trace:
            fallback_parts.append("Chưa có định lý nào được kích hoạt từ giả thiết ban đầu.\n")
        for i, step in enumerate(request.execution_trace):
            fallback_parts.append(f"### Bước {i + 1}: `{step.rule_id}`\n")
            fallback_parts.append(f"- **Định lý:** `{step.fired_rule_repr}`\n")
            fallback_parts.append(f"- **Tính chất mới:** `{', '.join(step.new_facts)}`\n\n")
        fallback_parts.append("</details>\n\n")
        fallback_parts.append(
            "## Gợi ý\nCó thể cần bổ sung thêm giả thiết hoặc kẻ thêm đường phụ để tạo cầu nối suy luận.\n"
        )

    from rag_agent.llm_factory import get_llm

    llm = get_llm(temperature=0.3)

    async def stream_generator():
        yielded_any = False
        if llm:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage

                messages = [
                    SystemMessage(content=_build_explain_system_prompt(request.goal_reached)),
                    HumanMessage(content=f"Query: '{request.query}'\n\nProof Trace:\n{trace_text or 'No rules triggered.'}{aux_text}")
                ]
                async for chunk in llm.astream(messages):
                    content = chunk.content
                    if not content:
                        continue
                    text = content if isinstance(content, str) else str(content)
                    if text:
                        yielded_any = True
                        yield text
            except Exception as e:
                logger.warning("Streaming LLM explanation failed: %s — falling back to template", e)

        if not yielded_any:
            for part in fallback_parts:
                yield part
                await asyncio.sleep(0.01)

    return StreamingResponse(stream_generator(), media_type="text/plain")



@app.get("/rules", tags=["Knowledge Graph"])
@app.get("/api/rules", tags=["Knowledge Graph"])
async def get_rules(
    domain: Optional[str] = Query(
        None, description="Filter by domain (always geometry)"
    ),
):
    """List all geometry rules registered in the Neo4j Knowledge Graph."""
    db_conn = Neo4jConnection()
    if not db_conn.verify_connectivity():
        db_conn.close()
        raise HTTPException(status_code=503, detail="Neo4j is unreachable.")
    try:
        with db_conn.get_session() as session:
            q = "MATCH (r:Rule) "
            params = {}
            if domain:
                q += "WHERE r.domain = $domain "
                params["domain"] = domain.lower()
            q += "RETURN r.id AS id, r.name AS name, r.inputs AS inputs, r.outputs AS outputs, r.domain AS domain, r.description AS description"
            result = session.run(q, **params)
            rules = [
                {
                    "id": rec["id"],
                    "name": rec["name"],
                    "domain": rec["domain"],
                    "inputs": rec["inputs"],
                    "outputs": rec["outputs"],
                    "description": rec["description"],
                }
                for rec in result
            ]
        return {"count": len(rules), "rules": rules}
    finally:
        db_conn.close()


@app.get("/ontology/classes", tags=["Ontology"])
async def get_ontology_classes():
    """List the geometry ontology class hierarchy stored in Neo4j."""
    db_conn = Neo4jConnection()
    if not db_conn.verify_connectivity():
        db_conn.close()
        raise HTTPException(status_code=503, detail="Neo4j is unreachable.")
    try:
        with db_conn.get_session() as session:
            result = session.run(
                "MATCH (c:OntologyClass) "
                "OPTIONAL MATCH (c)-[:IS_A]->(parent:OntologyClass) "
                "RETURN c.name AS name, c.uri AS uri, parent.name AS parent "
                "ORDER BY c.name"
            )
            classes = [
                {"name": rec["name"], "uri": rec["uri"], "parent": rec["parent"]}
                for rec in result
            ]
        return {"count": len(classes), "classes": classes}
    finally:
        db_conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
