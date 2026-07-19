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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("geo-ips-api")

app = FastAPI(
    title="GeoIPS API Gateway",
    description=(
        "Plane Geometry Intelligent Problem Solver — "
        "Neuro-Symbolic & GraphRAG engine inspired by AlphaGeometry."
    ),
    version="2.0.0"
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
        domain=DOMAIN
    )
    raw_rules = []
    for record in result:
        raw_rules.append({
            "id": record["id"],
            "name": record["name"],
            "inputs": record["inputs"] or [],
            "outputs": record["outputs"] or [],
            "description": record["description"] or ""
        })
    return raw_rules


# Built-in fallback rule subset (used when Neo4j is unreachable)
FALLBACK_GEOMETRY_RULES = [
    {"id": "geo_congruence_reflexive", "name": "Congruence Reflexive", "inputs": ["Segment(AB)"], "outputs": ["Congruent(AB,AB)"], "description": "AB ≅ AB."},
    {"id": "geo_congruence_symmetric", "name": "Congruence Symmetric", "inputs": ["Congruent(AB,CD)"], "outputs": ["Congruent(CD,AB)"], "description": "If AB≅CD then CD≅AB."},
    {"id": "geo_congruence_transitive", "name": "Congruence Transitivity", "inputs": ["Congruent(AB,CD)", "Congruent(CD,EF)"], "outputs": ["Congruent(AB,EF)"], "description": "Transitivity of congruence."},
    {"id": "geo_perp_symmetry", "name": "Perpendicular Symmetry", "inputs": ["Perpendicular(AB,CD)"], "outputs": ["Perpendicular(CD,AB)"], "description": "Perpendicularity is symmetric."},
    {"id": "geo_parallel_transitive", "name": "Parallel Transitivity", "inputs": ["Parallel(a,b)", "Parallel(b,c)"], "outputs": ["Parallel(a,c)"], "description": "Transitivity of parallel lines."},
    # Variable-based parallel/perp rules (match ANY line names — uppercase AB, CD, etc.)
    {"id": "geo_parallel_transitive_var", "name": "Parallel Transitivity (general)",
     "inputs": ["Parallel(?A,?B)", "Parallel(?B,?C)"], "outputs": ["Parallel(?A,?C)"],
     "description": "Transitivity: AB∥CD ∧ CD∥EF ⇒ AB∥EF"},
    {"id": "geo_parallel_symmetric_var", "name": "Parallel Symmetric (general)",
     "inputs": ["Parallel(?A,?B)"], "outputs": ["Parallel(?B,?A)"],
     "description": "Parallel is symmetric."},
    {"id": "geo_perp_symmetry_var", "name": "Perpendicular Symmetric (general)",
     "inputs": ["Perpendicular(?A,?B)"], "outputs": ["Perpendicular(?B,?A)"],
     "description": "Perpendicularity is symmetric."},
    {"id": "geo_triangle_angle_sum", "name": "Triangle Angle Sum", "inputs": ["Triangle(A,B,C)"], "outputs": ["Equal(Add(Angle(BAC),Angle(ABC),Angle(ACB)),180)"], "description": "Angles of a triangle sum to 180°."},
    {"id": "geo_isosceles_base_angles", "name": "Isosceles Base Angles", "inputs": ["Triangle(A,B,C)", "Congruent(AB,AC)"], "outputs": ["Equal(Angle(ABC),Angle(ACB))"], "description": "Base angles of isosceles triangle are equal."},
    {"id": "geo_isosceles_reverse", "name": "Converse Isosceles", "inputs": ["Triangle(A,B,C)", "Equal(Angle(ABC),Angle(ACB))"], "outputs": ["Congruent(AB,AC)"], "description": "Equal base angles implies isosceles."},
    # ── Congruence theorems — variable-based, ANY triangle pair ────────────
    # IDs use _var suffix so they are ALWAYS merged (never overridden by Neo4j's
    # propositional geo_sas_congruence / geo_asa_congruence etc.).
    {"id": "geo_sas_congruence_var",
     "name": "SAS Congruence (general)",
     "inputs": ["Triangle(?A,?B,?C)", "Triangle(?D,?E,?F)",
                "Congruent(?A?B,?D?E)",
                "Equal(Angle(?B?A?C),Angle(?E?D?F))",
                "Congruent(?A?C,?D?F)"],
     "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
     "description": "SAS: AB=DE, ∠BAC=∠EDF, AC=DF ⇒ △ABC≅△DEF"},
    {"id": "geo_asa_congruence_var",
     "name": "ASA Congruence (general)",
     "inputs": ["Triangle(?A,?B,?C)", "Triangle(?D,?E,?F)",
                "Equal(Angle(?B?A?C),Angle(?E?D?F))",
                "Congruent(?A?B,?D?E)",
                "Equal(Angle(?A?B?C),Angle(?D?E?F))"],
     "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
     "description": "ASA: ∠BAC=∠EDF, AB=DE, ∠ABC=∠DEF ⇒ △ABC≅△DEF"},
    {"id": "geo_sss_congruence_var",
     "name": "SSS Congruence (general)",
     "inputs": ["Triangle(?A,?B,?C)", "Triangle(?D,?E,?F)",
                "Congruent(?A?B,?D?E)",
                "Congruent(?B?C,?E?F)",
                "Congruent(?A?C,?D?F)"],
     "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
     "description": "SSS: AB=DE, BC=EF, AC=DF ⇒ △ABC≅△DEF"},
    {"id": "geo_aas_congruence_var",
     "name": "AAS Congruence (general)",
     "inputs": ["Triangle(?A,?B,?C)", "Triangle(?D,?E,?F)",
                "Equal(Angle(?B?A?C),Angle(?E?D?F))",
                "Equal(Angle(?A?B?C),Angle(?D?E?F))",
                "Congruent(?B?C,?E?F)"],
     "outputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
     "description": "AAS: ∠BAC=∠EDF, ∠ABC=∠DEF, BC=EF ⇒ △ABC≅△DEF"},
    # Symmetry: CongruentTriangles(ABC,DEF) ⇔ CongruentTriangles(DEF,ABC)
    {"id": "geo_congruent_tri_symmetric",
     "name": "Congruent Triangles Symmetric",
     "inputs": ["CongruentTriangles(?ABC,?DEF)"],
     "outputs": ["CongruentTriangles(?DEF,?ABC)"],
     "description": "Triangle congruence is symmetric."},
    {"id": "geo_pythagoras_var", "name": "Pythagorean Theorem",
     "inputs": ["RightTriangle(?A,?B,?C)", "RightAngle(Angle(?B?A?C))"],
     "outputs": ["Equal(Pow(Length(?B?C),2),Add(Pow(Length(?A?B),2),Pow(Length(?A?C),2)))"],
     "description": "BC² = AB² + AC² in a right-angled triangle."},
    {"id": "geo_pythagoras_converse_var", "name": "Pythagorean Converse",
     "inputs": ["Triangle(?A,?B,?C)", "Equal(Pow(Length(?B?C),2),Add(Pow(Length(?A?B),2),Pow(Length(?A?C),2)))"],
     "outputs": ["RightTriangle(?A,?B,?C)", "RightAngle(Angle(?B?A?C))"],
     "description": "If BC² = AB² + AC² then the triangle is right-angled at A."},
    {"id": "geo_right_triangle_height_metric_var", "name": "Right Triangle Height Metric",
     "inputs": ["RightTriangle(?A,?B,?C)", "RightAngle(Angle(?B?A?C))", "Foot(?H,?A,?B?C)"],
     "outputs": ["Equal(Div(1,Pow(Length(?A?H),2)),Add(Div(1,Pow(Length(?A?B),2)),Div(1,Pow(Length(?A?C),2))))"],
     "description": "1/AH² = 1/AB² + 1/AC² in a right triangle with altitude AH."},
    {"id": "geo_thales_var", "name": "Thales Theorem",
     "inputs": ["Diameter(?A?B,Circle(?O))", "PointOnCircle(?C,Circle(?O))"],
     "outputs": ["RightAngle(Angle(?A?C?B))"],
     "description": "Angle in a semicircle is 90°."},
    {"id": "geo_congruent_tri_sides_var", "name": "Congruent Triangles → Sides",
     "inputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
     "outputs": ["Congruent(?A?B,?D?E)", "Congruent(?B?C,?E?F)", "Congruent(?A?C,?D?F)"],
     "description": "Congruent triangles have congruent sides."},
    {"id": "geo_congruent_tri_angles_var", "name": "Congruent Triangles → Angles",
     "inputs": ["CongruentTriangles(?A?B?C,?D?E?F)"],
     "outputs": ["Equal(Angle(?B?A?C),Angle(?E?D?F))", "Equal(Angle(?A?B?C),Angle(?D?E?F))", "Equal(Angle(?A?C?B),Angle(?D?F?E))"],
     "description": "Congruent triangles have equal angles."},
    {"id": "geo_midpoint_theorem_var", "name": "Midsegment Theorem",
     "inputs": ["Triangle(?A,?B,?C)", "Midpoint(?M,?A?B)", "Midpoint(?N,?A?C)"],
     "outputs": ["Parallel(?M?N,?B?C)", "Equal(Length(?M?N),Div(Length(?B?C),2))"],
     "description": "Midsegment is parallel to base and half its length."},
    {"id": "geo_right_triangle_expand_var", "name": "RightTriangle Expand",
     "inputs": ["RightTriangle(?A,?B,?C)"],
     "outputs": ["Triangle(?A,?B,?C)", "RightAngle(Angle(?B?A?C))"],
     "description": "Expand RightTriangle to Triangle + RightAngle."},
    {"id": "geo_exterior_angle_var", "name": "Exterior Angle Theorem",
     "inputs": ["Triangle(?A,?B,?C)", "ExteriorAngle(?E,?A,?B?C)"],
     "outputs": ["Equal(Angle(?E),Add(Angle(?B?A?C),Angle(?A?B?C)))"],
     "description": "Exterior angle is sum of two non-adjacent interior angles."},
    {"id": "geo_equilateral_all_60_var", "name": "Equilateral Triangle Angles",
     "inputs": ["Triangle(?A,?B,?C)", "Congruent(?A?B,?B?C)", "Congruent(?B?C,?A?C)"],
     "outputs": ["Equal(Angle(?B?A?C),60)", "Equal(Angle(?A?B?C),60)", "Equal(Angle(?A?C?B),60)"],
     "description": "All angles of equilateral triangle are 60°."},
    {"id": "geo_thales_right_angle_var", "name": "Thales: Angle in Semicircle",
     "inputs": ["Diameter(?A?B,Circle(?O))", "PointOnCircle(?C,Circle(?O))"],
     "outputs": ["RightAngle(Angle(?A?C?B))", "Equal(Angle(?A?C?B),90)"],
     "description": "Angle inscribed in a semicircle is 90°."},
    {"id": "geo_cyclic_quad_opposite_var", "name": "Cyclic Quadrilateral Opposite Angles",
     "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D,Circle(?O))"],
     "outputs": ["Equal(Add(Angle(?D?A?B),Angle(?B?C?D)),180)", "Equal(Add(Angle(?A?B?C),Angle(?C?D?A)),180)"],
     "description": "Opposite angles of cyclic quadrilateral sum to 180°."},
    {"id": "geo_aa_similarity_var", "name": "AA Similarity",
     "inputs": ["Triangle(?A,?B,?C)", "Triangle(?D,?E,?F)", "Equal(Angle(?B?A?C),Angle(?E?D?F))", "Equal(Angle(?A?B?C),Angle(?D?E?F))"],
     "outputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
     "description": "AA similarity theorem."},
    {"id": "geo_similar_tri_angles_var", "name": "Similar Triangles: Equal Angles",
     "inputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
     "outputs": ["Equal(Angle(?B?A?C),Angle(?E?D?F))", "Equal(Angle(?A?B?C),Angle(?D?E?F))", "Equal(Angle(?A?C?B),Angle(?D?F?E))"],
     "description": "Similar triangles have equal corresponding angles."},
    {"id": "geo_similar_tri_ratios_var", "name": "Similar Triangles: Proportional Sides",
     "inputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
     "outputs": ["Equal(Div(Length(?A?B),Length(?D?E)),Div(Length(?B?C),Length(?E?F)))",
                 "Equal(Div(Length(?A?B),Length(?D?E)),Div(Length(?A?C),Length(?D?F)))"],
     "description": "Similar triangles have proportional corresponding sides."},
    {"id": "geo_midpoint_halves_var", "name": "Midpoint Halves Segment",
     "inputs": ["Midpoint(?M,?A?B)"],
     "outputs": ["Equal(Length(?A?M),Length(?M?B))", "Equal(Length(?A?M),Div(Length(?A?B),2))"],
     "description": "Midpoint divides segment into two equal halves."},
 
     # ── Circle & Power of Point Theorems ──────────────────────────────────────
    {"id": "geo_chord_definition_var", "name": "Chord Definition",
     "inputs": ["PointOnCircle(?A,Circle(?O))", "PointOnCircle(?B,Circle(?O))"],
     "outputs": ["Chord(?A,?B,Circle(?O))"],
     "description": "A segment connecting two points on a circle is a chord."},
    {"id": "geo_intersecting_chords_var", "name": "Intersecting Chords Theorem",
     "inputs": ["Circle(?O)", "PointOnCircle(?A,Circle(?O))", "PointOnCircle(?B,Circle(?O))",
                "PointOnCircle(?C,Circle(?O))", "PointOnCircle(?D,Circle(?O))",
                "IntersectionPoint(?P,?A?B,?C?D)"],
     "outputs": ["Equal(Mul(Length(?A?P),Length(?P?B)),Mul(Length(?C?P),Length(?P?D)))"],
     "description": "Power of Point: PA * PB = PC * PD for intersecting chords."},
    {"id": "geo_tangent_secant_theorem_var", "name": "Tangent-Secant Theorem",
     "inputs": ["Circle(?O)", "PointOutsideCircle(?P,Circle(?O))",
                "TangentSegment(?P,?T,Circle(?O))", "SecantSegment(?P,?A,?B,Circle(?O))"],
     "outputs": ["Equal(Pow(Length(?P?T),2),Mul(Length(?P?A),Length(?P?B)))"],
     "description": "Square of tangent segment equals product of secant segments."},
    {"id": "geo_ptolemy_theorem_var", "name": "Ptolemy's Theorem",
     "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D)"],
     "outputs": ["Equal(Mul(Length(?A?C),Length(?B?D)),Add(Mul(Length(?A?B),Length(?C?D)),Mul(Length(?B?C),Length(?A?D))))"],
     "description": "Ptolemy: AC * BD = AB * CD + BC * AD."},
 
     # ── Rhombus & Olympiad Lemmas ─────────────────────────────────────────────
    {"id": "geo_rhombus_diagonals_perp_var", "name": "Rhombus Diagonals are Perpendicular",
     "inputs": ["Rhombus(?A,?B,?C,?D)"],
     "outputs": ["Perpendicular(?A?C,?B?D)"],
     "description": "The diagonals of a rhombus are perpendicular to each other."},
    {"id": "geo_ceva_theorem_var", "name": "Ceva's Theorem",
     "inputs": ["Triangle(?A,?B,?C)", "PointOnSegment(?D,?B,?C)", "PointOnSegment(?E,?A,?C)",
                "PointOnSegment(?F,?A,?B)", "Concurrent(?A?D,?B?E,?C?F)"],
     "outputs": ["Equal(Mul(Div(Length(?B?D),Length(?C?D)),Mul(Div(Length(?C?E),Length(?A?E)),Div(Length(?A?F),Length(?B?F)))),1)"],
     "description": "Condition for three cevians to be concurrent."},
    {"id": "geo_menelaus_theorem_var", "name": "Menelaus's Theorem",
     "inputs": ["Triangle(?A,?B,?C)", "PointOnLine(?D,?B,?C)", "PointOnLine(?E,?C,?A)",
                "PointOnLine(?F,?A,?B)", "Collinear(?D,?E,?F)"],
     "outputs": ["Equal(Mul(Div(Length(?B?D),Length(?C?D)),Mul(Div(Length(?C?E),Length(?A?E)),Div(Length(?A?F),Length(?B?F)))),1)"],
     "description": "Menelaus collinearity theorem."},
    {"id": "geo_simson_line_var", "name": "Simson Line Theorem",
     "inputs": ["Triangle(?A,?B,?C)", "PointOnCircle(?P,Circle(?O))",
                "Circumcircle(?O,?A,?B,?C)",
                "Foot(?X,?P,?A?B)", "Foot(?Y,?P,?B?C)", "Foot(?Z,?P,?A?C)"],
     "outputs": ["Collinear(?X,?Y,?Z)"],
     "description": "The feet of the perpendiculars from a point on the circumcircle are collinear."},
    {"id": "geo_parallel_quadrilateral_properties_var", "name": "Parallel Quadrilateral Diagonal Properties",
     "inputs": ["Parallel(?A?B,?C?D)", "Parallel(?A?D,?B?C)", "PointOnSegment(?M,?A,?C)", "PointOnSegment(?N,?A,?C)"],
     "outputs": [
         "Congruent(?A?D,?C?B)",
         "Congruent(?A?B,?C?D)",
         "Triangle(?A,?D,?M)",
         "Triangle(?C,?B,?N)",
         "Equal(Angle(?D?A?M),Angle(?B?C?N))"
     ],
     "description": "If opposite sides of a quadrilateral are parallel, it is a parallelogram, yielding congruent opposite sides, diagonal alternate angles, and triangle existence."},
    {"id": "geo_parallelogram_properties_var", "name": "Parallelogram Properties",
     "inputs": ["Parallelogram(?A,?B,?C,?D)", "PointOnSegment(?M,?A,?C)", "PointOnSegment(?N,?A,?C)"],
     "outputs": [
         "Parallel(?A?B,?C?D)",
         "Parallel(?A?D,?B?C)",
         "Congruent(?A?D,?C?B)",
         "Congruent(?A?B,?C?D)",
         "Triangle(?A,?D,?M)",
         "Triangle(?C,?B,?N)",
         "Equal(Angle(?D?A?M),Angle(?B?C?N))"
     ],
     "description": "Properties of a parallelogram with points on diagonal AC."},
    {"id": "geo_isosceles_midpoint_perp_var", "name": "Isosceles Triangle Midpoint Altitude",
     "inputs": ["Triangle(?A,?B,?C)", "Congruent(?A?B,?A?C)", "Midpoint(?M,?B?C)"],
     "outputs": ["Perpendicular(?A?M,?B?C)", "RightAngle(Angle(?B?M?A))", "RightAngle(Angle(?A?M?B))", "RightAngle(Angle(?C?M?A))"],
     "description": "In an isosceles triangle AB=AC, the median AM is perpendicular to BC."},
    {"id": "geo_isosceles_midpoint_perp_length_var", "name": "Isosceles Triangle Midpoint Altitude (Length)",
     "inputs": ["Triangle(?A,?B,?C)", "Equal(Length(?A?B),Length(?A?C))", "Midpoint(?M,?B?C)"],
     "outputs": ["Perpendicular(?A?M,?B?C)", "RightAngle(Angle(?B?M?A))", "RightAngle(Angle(?A?M?B))", "RightAngle(Angle(?C?M?A))"],
     "description": "In an isosceles triangle AB=AC, the median AM is perpendicular to BC."},
    {"id": "geo_circle_radii_equal_var", "name": "Circle Radii Length Equality",
     "inputs": ["PointOnCircle(?M,Circle(?O))", "PointOnCircle(?N,Circle(?O))"],
     "outputs": ["Equal(Length(?O?M),Length(?O?N))", "Congruent(?M?O,?N?O)", "Congruent(?O?M,?O?N)", "Congruent(?M?O,?O?N)", "Congruent(?O?M,?N?O)", "Congruent(?M?O,?N?O)", "Congruent(MO,NO)"],
     "description": "All radii from the center of a circle to points on the circle are equal in length."},
    {"id": "geo_point_on_circle_implies_circle_var", "name": "Point On Circle Implies Circle Existence",
     "inputs": ["PointOnCircle(?M,Circle(?O))"],
     "outputs": ["Circle(?O)"],
     "description": "If M is on Circle(O), then Circle(O) exists."},
    {"id": "geo_circle_midpoint_radius_var", "name": "Circle Midpoint Diameter Definition",
     "inputs": ["Midpoint(?O,?A?B)", "Circle(?O)", "PointOnCircle(?C,Circle(?O))"],
     "outputs": ["PointOnCircle(?A,Circle(?O))", "PointOnCircle(?B,Circle(?O))", "Diameter(?A?B)", "RightAngle(Angle(?A?C?B))", "RightAngle(Angle(?C))"],
     "description": "If O is midpoint of AB and C is on Circle(O), then AB is a diameter and angle ACB is a right angle."},
    {"id": "geo_midpoint_congruence_var", "name": "Midpoint Congruence Definition",
     "inputs": ["Midpoint(?M,?B?C)"],
     "outputs": ["Equal(Length(?B?M),Length(?M?C))", "Congruent(?B?M,?C?M)", "Congruent(?B?M,?M?C)"],
     "description": "Midpoint M of BC divides segment BC into congruent halves BM and MC."},
    {"id": "geo_circumcenter_midpoint_perp_var", "name": "Circumcenter Midpoint Perpendicular Bisector",
     "inputs": ["Triangle(?A,?B,?C)", "Circumcircle(?O,?A,?B,?C)", "Midpoint(?M,?B?C)"],
     "outputs": ["Perpendicular(?O?M,?B?C)", "Perpendicular(?M?O,?B?C)", "RightAngle(Angle(?O?M?B))"],
     "description": "The segment from circumcenter O to midpoint M of BC is perpendicular to BC."},
    {"id": "geo_circle_points_midpoint_perp_var", "name": "Circle Points Midpoint Perpendicular Bisector",
     "inputs": ["Triangle(?A,?B,?C)", "Circle(?O)", "PointOnCircle(?A,Circle(?O))", "PointOnCircle(?B,Circle(?O))", "PointOnCircle(?C,Circle(?O))", "Midpoint(?M,?B?C)"],
     "outputs": ["Perpendicular(?O?M,?B?C)", "Perpendicular(?M?O,?B?C)", "RightAngle(Angle(?O?M?B))"],
     "description": "The segment from center O of circumcircle to midpoint M of BC is perpendicular to BC."},
    {"id": "geo_congruent_radii_is_circumcenter_var", "name": "Congruent Radii form Circumcenter Midpoint Perpendicular",
     "inputs": ["Triangle(?A,?B,?C)", "Congruent(?A?O,?B?O)", "Congruent(?B?O,?C?O)", "Midpoint(?M,?B?C)"],
     "outputs": ["Perpendicular(?O?M,?B?C)", "Perpendicular(?M?O,?B?C)", "RightAngle(Angle(?O?M?B))"],
     "description": "If O is equidistant from vertices A, B, C of triangle ABC, then OM is perpendicular to BC."},
    {"id": "geo_angle_bisector_theorem_var", "name": "Angle Bisector Theorem",
     "inputs": ["Triangle(?A,?B,?C)", "AngleBisector(?A?D,Angle(?B?A?C))", "PointOnSegment(?D,?B,?C)"],
     "outputs": [
         "Equal(Div(Length(?B?D),Length(?C?D)),Div(Length(?A?B),Length(?A?C)))",
         "Equal(Div(Length(?B?D),Length(?D?C)),Div(Length(?A?B),Length(?A?C)))"
     ],
     "description": "An angle bisector of a triangle divides the opposite side into segments proportional to the adjacent sides."},
    {"id": "geo_angle_equal_is_bisector_var", "name": "Equal Angles form Angle Bisector",
     "inputs": ["Triangle(?A,?B,?C)", "Equal(Angle(?B?A?D),Angle(?C?A?D))", "PointOnSegment(?D,?B,?C)"],
     "outputs": ["AngleBisector(?A?D,Angle(?B?A?C))"],
     "description": "If D is on segment BC and angle BAD equals angle CAD, then AD is the angle bisector of angle BAC."},
    {"id": "geo_inscribed_angle_theorem_var", "name": "Inscribed Angle Theorem (Circle Points)",
     "inputs": ["Circle(?O)", "PointOnCircle(?X,Circle(?O))", "PointOnCircle(?Y,Circle(?O))", "PointOnCircle(?Z,Circle(?O))", "PointOnCircle(?W,Circle(?O))"],
     "outputs": ["Equal(Angle(?Y?X?Z),Angle(?Y?W?Z))"],
     "description": "Angles subtended by the same segment in cyclic points are equal."},
    {"id": "geo_inscribed_angle_cyclic_quad_var", "name": "Inscribed Angle Theorem (Cyclic Quadrilateral)",
     "inputs": ["CyclicQuadrilateral(?X,?Y,?Z,?W)"],
     "outputs": ["Equal(Angle(?Y?X?Z),Angle(?Y?W?Z))"],
     "description": "Angles subtended by the same segment in a cyclic quadrilateral are equal."},
    {"id": "geo_inscribed_angle_cyclic_points_var", "name": "Inscribed Angle Theorem (Cyclic Points)",
     "inputs": ["CyclicPoints(?X,?Y,?Z,?W)"],
     "outputs": ["Equal(Angle(?Y?X?Z),Angle(?Y?W?Z))"],
     "description": "Angles subtended by the same segment in cyclic points are equal."},
    {"id": "geo_points_on_circle_is_circumcircle_var", "name": "Circle Points form Circumcircle",
     "inputs": ["Circle(?O)", "PointOnCircle(?A,Circle(?O))", "PointOnCircle(?B,Circle(?O))", "PointOnCircle(?C,Circle(?O))"],
     "outputs": ["Circumcircle(?O,?A,?B,?C)"],
     "description": "If points A, B, C lie on Circle(O), O is the circumcircle of triangle ABC."},
    {"id": "geo_collinear_center_is_diameter_var", "name": "Collinear Points with Center form Diameter",
     "inputs": ["Circle(?O)", "PointOnCircle(?A,Circle(?O))", "PointOnCircle(?D,Circle(?O))", "Collinear(?A,?D,?O)"],
     "outputs": ["Diameter(?A?D,Circle(?O))"],
     "description": "If A and D lie on Circle(O) and are collinear with center O, AD is a diameter."},
    {"id": "geo_midpoint_center_is_diameter_var", "name": "Midpoint Center forms Diameter",
     "inputs": ["Circle(?O)", "PointOnCircle(?A,Circle(?O))", "PointOnCircle(?D,Circle(?O))", "Midpoint(?O,?A?D)"],
     "outputs": ["Diameter(?A?D,Circle(?O))"],
     "description": "If O is the midpoint of segment AD and A, D lie on Circle(O), AD is a diameter."},
    {"id": "geo_isogonal_altitude_circumcenter_var", "name": "Altitude and Circumcenter Isogonal Relation",
     "inputs": ["Triangle(?A,?B,?C)", "Foot(?H,?A,?B?C)", "Circumcircle(?O,?A,?B,?C)"],
     "outputs": ["Equal(Angle(?B?A?H),Angle(?C?A?O))", "Equal(Angle(?B?A?H),Angle(?O?A?C))"],
     "description": "The altitude from A and the circumradius AO form equal angles with adjacent sides AB and AC."},
    {"id": "geo_thales_diameter_angle_var", "name": "Thales' Theorem on Diameter",
     "inputs": ["Circle(?O)", "Diameter(?A?D)", "PointOnCircle(?C,Circle(?O))"],
     "outputs": ["RightAngle(Angle(?A?C?D))"],
     "description": "An angle inscribed in a semicircle is a right angle (90 degrees)."},
    {"id": "geo_thales_diameter_angle_2arg_var", "name": "Thales' Theorem on Diameter (2-arg)",
     "inputs": ["Circle(?O)", "Diameter(?A?D,Circle(?O))", "PointOnCircle(?C,Circle(?O))"],
     "outputs": ["RightAngle(Angle(?A?C?D))"],
     "description": "An angle inscribed in a semicircle is a right angle (90 degrees)."},
    {"id": "geo_thales_three_points_circle_var", "name": "Thales' Semicircle Angle Theorem",
     "inputs": ["Circle(?O)", "PointOnCircle(?A,Circle(?O))", "PointOnCircle(?D,Circle(?O))", "PointOnCircle(?C,Circle(?O))"],
     "outputs": ["RightAngle(Angle(?A?C?D))"],
     "description": "Any three points on a circle with diameter AD form a right angle ACD."},
    {"id": "geo_perp_angle_transfer_var", "name": "Perpendicular Line Angle Transfer",
     "inputs": ["Perpendicular(?A?B,?D?E)", "Perpendicular(?A?C,?D?F)"],
     "outputs": ["Equal(Angle(?B?A?C),Angle(?E?D?F))"],
     "description": "If two pairs of lines are mutually perpendicular, the angles between them are equal."},
    {"id": "geo_cyclic_quad_opposite_angles_var", "name": "Cyclic Quadrilateral Opposite Angle Sum",
     "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D)"],
     "outputs": [
         "Equal(Add(Angle(?D?A?B),Angle(?B?C?D)),180)",
         "Equal(Angle(?D?A?B)+Angle(?B?C?D),180)",
         "Equal(Angle(?A?B?C)+Angle(?C?D?A),180)"
     ],
     "description": "Opposite angles of a cyclic quadrilateral sum to 180 degrees."},
    {"id": "geo_similar_triangles_angle_equality_var", "name": "Similar Triangles Angle Equality",
     "inputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
     "outputs": ["Equal(Angle(?A?B?C),Angle(?D?E?F))", "Equal(Angle(?B?C?A),Angle(?E?F?D))", "Equal(Angle(?C?A?B),Angle(?F?D?E))"],
     "description": "Corresponding angles of similar triangles are equal."},
    {"id": "geo_similar_triangles_ratio_var", "name": "Similar Triangles Side Ratios",
     "inputs": ["SimilarTriangles(?A?B?C,?D?E?F)"],
     "outputs": ["Equal(Div(Length(?A?B),Length(?D?E)),Div(Length(?B?C),Length(?E?F)))"],
     "description": "Corresponding sides of similar triangles are proportional."}
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
        logger.info("Merged rules: %d Neo4j + %d fallback additions = %d total",
                    len(neo4j_rules), added_fallback, len(merged))
    return merged



# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class SolveRequest(BaseModel):
    domain: str = Field("geometry", description="Domain (always 'geometry' in GeoIPS)")
    facts: List[str] = Field(..., description="List of raw geometry predicates as initial facts")
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
        3, description="Max times the auxiliary agent can add constructions (0 = disabled)"
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
        initial_facts = [PARSER.parse_fact(f, f"init_{i}") for i, f in enumerate(request.facts)]
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
            raise HTTPException(status_code=400, detail=f"Unknown strategy: '{strategy}'")

        result = engine.solve(initial_facts, goal_fact)

        steps = [
            ExecutionStepResponse(
                rule_id=s.rule_id,
                fired_rule_repr=s.fired_rule_repr,
                new_facts=[f.value for f in s.new_facts]
            ) for s in result.execution_trace
        ]

        return SolveResponse(
            goal_reached=result.goal_reached,
            applied_rule_ids=result.applied_rule_ids,
            execution_trace=steps,
            known_facts=sorted(set(f.value for f in result.final_facts))
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
                detail="Could not map any initial facts. Please be more specific."
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
                new_facts=[f.value for f in s.new_facts]
            ) for s in result.execution_trace
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


@app.post("/geo/solve", response_model=SolveQueryResponse, tags=["AlphaGeometry-style Solver"])
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
                detail="Could not map any initial facts from the query."
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
                logger.info("[GeoSolve] Solver derived %d new facts — retrying without aux", new_facts_count)
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
                    new_fact = PARSER.parse_fact(nf_str, f"aux_{len(all_constructions)}")
                    if new_fact not in current_facts:
                        current_facts.append(new_fact)
                        logger.info("[GeoSolve] Added auxiliary fact: %s", nf_str)

        return result, initial_facts, goal_fact, all_constructions

    try:
        try:
            result, initial_facts, goal_fact, all_constructions = await asyncio.wait_for(
                _run_geo_solve(),
                timeout=SOLVER_DEADLINE_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("[GeoSolve] Solver deadline (%ds) exceeded — returning partial result", SOLVER_DEADLINE_SECONDS)
            # Return a partial result indicating timeout rather than crashing
            raise HTTPException(
                status_code=408,
                detail=f"Solver timed out after {SOLVER_DEADLINE_SECONDS}s. The problem may require more advanced constructions. Try simplifying the query or check the problem statement."
            )

        steps = [
            ExecutionStepResponse(
                rule_id=s.rule_id,
                fired_rule_repr=s.fired_rule_repr,
                new_facts=[f.value for f in s.new_facts]
            ) for s in result.execution_trace
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
        "CRITICAL FOR LATEX MATHEMATICAL NOTATION:\n"
        "Ensure all mathematical symbols, equations, segments, angles, and triangles are beautifully and consistently formatted in LaTeX using single dollar signs ($...$):\n"
        "- Triangles: use $\\triangle ABC$ instead of 'Triangle(ABC)' or 'tam giác ABC'.\n"
        "- Angles: use $\\angle ABC$ instead of 'Angle(ABC)' or 'góc ABC'.\n"
        "- Segments: use $AB$, $CD$ instead of raw segment strings.\n"
        "- Congruence: use $\\cong$ (e.g., $\\triangle ABC \\cong \\triangle DEF$, $AB \\cong CD$).\n"
        "- Parallelism: use $\\parallel$ (e.g., $AB \\parallel CD$).\n"
        "- Equality: use $=$ (e.g., $AB = CD$ or $\\angle ABC = \\angle DEF$).\n"
        "- Degree: use $^\\circ$ (e.g., $90^\\circ$, $50^\\circ$) instead of the degree symbol (°) or word 'độ'.\n"
        "- Fractions/Division: use $\\frac{{a}}{{b}}$ (e.g., $\\frac{{1}}{{AH^2}} = \\frac{{1}}{{AB^2}} + \\frac{{1}}{{AC^2}}$) instead of slashes or raw division predicates.\n"
        "- Products/Multiplication: use $\\cdot$ (e.g., $PA \\cdot PB = PC \\cdot PD$) or standard multiplication formatting.\n"
        "All explanations must be in Vietnamese, highly professional, standard, and easy to read.\n"
    )

    if goal_reached:
        return (
            "You are an expert plane geometry tutor for GeoIPS.\n"
            "Your task is to write a beautiful, natural language geometry proof (lời giải tự nhiên chuẩn học sinh đang thi IMO) "
            "in Vietnamese that reads elegantly, starts with the given facts, uses logical transition words, and ends with the goal.\n"
            "Make it read like an elegant proof written by a human mathematician. Do NOT write a raw list of steps like 'Step 1: ...' in the main body. "
            "Instead, integrate all the logical steps from the proof trace into a cohesive, paragraph-based mathematical proof.\n"
            "Reference specific Euclidean theorems and axioms by name where relevant (e.g., định lý Thales, định lý Pythagore, tam giác đồng dạng, hai tam giác bằng nhau theo trường hợp cạnh-góc-cạnh (c-g-c)).\n"
            + latex_guide +
            "\n"
            "CRITICAL: At the very end of your response, you MUST output the raw deduction steps wrapped inside an HTML `<details>` toggle like this:\n"
            "<details>\n"
            "<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n"
            "\n"
            "### Các bước suy luận hệ thống:\n"
            "For each step in the trace, write a short bullet point explaining what rule fired and what was derived (e.g., *Bước 1: Áp dụng định lý... để suy ra...*).\n"
            "</details>\n"
            "\n"
            "STRICT: Only explain logical connections that appear in the trace. Do not hallucinate extra steps."
        )
    else:
        return (
            "You are an expert plane geometry tutor for GeoIPS.\n"
            "The symbolic solver could NOT prove the goal from the given facts.\n"
            "Explain clearly to a high-school student in Vietnamese:\n"
            "1. What facts were given and what the goal was.\n"
            "2. What intermediate facts (if any) were deduced before the solver got stuck.\n"
            "3. WHY the proof failed — is a hypothesis missing? Is there a theorem needed that isn't in the KB?\n"
            "4. What additional information or constructions might help.\n"
            + latex_guide +
            "\n"
            "CRITICAL: Wrap the list of attempted/deduced steps inside an HTML `<details>` toggle at the end of your response like this:\n"
            "<details>\n"
            "<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n"
            "\n"
            "### Các bước đã thử nghiệm:\n"
            "...\n"
            "</details>\n"
            "\n"
            "STRICT: Do NOT claim the goal was proved. Start by clearly stating it is UNPROVED."
        )


@app.post("/api/explain", response_model=ExplainResponse, tags=["Explainability Agent"])
async def explain_proof(request: ExplainRequest):
    """Sync proof explanation — LLM or template fallback."""
    trace_text = "\n".join(
        f"Step {i+1}: [{s.rule_id}] {s.fired_rule_repr} → New facts: {s.new_facts}"
        for i, s in enumerate(request.execution_trace)
    )
    aux_text = ""
    if request.auxiliary_constructions:
        aux_text = "\nAuxiliary constructions added: " + ", ".join(request.auxiliary_constructions)

    from rag_agent.llm_factory import get_llm
    llm = get_llm(temperature=0.3)

    if llm:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", _build_explain_system_prompt(request.goal_reached)),
                ("human", "Query: '{query}'\n\nProof Trace:\n{trace}{aux}")
            ])
            chain = prompt | llm
            response = chain.invoke({
                "query": request.query,
                "trace": trace_text or "No rules triggered.",
                "aux": aux_text,
            })
            content = response.content if isinstance(response.content, str) else str(response.content)
            return ExplainResponse(explanation=content, structured=True)
        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)

    # Template fallback
    parts = []
    if request.goal_reached:
        parts = [
            "# Geometry Proof Explanation\n",
            f"**Query:** *{request.query}*\n",
            "The symbolic engine successfully proved the goal.\n\n",
        ]
        if request.auxiliary_constructions:
            parts.append(f"**Auxiliary Constructions Used:** {', '.join(request.auxiliary_constructions)}\n\n")
        
        parts.append("<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n")
        parts.append("## Deduction Steps\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Step {i+1}: `{step.rule_id}`")
            parts.append(f"- **Rule:** `{step.fired_rule_repr}`")
            parts.append(f"- **New facts:** `{', '.join(step.new_facts)}`\n")
        parts.append("\n</details>\n")
        parts.append("\n## Conclusion\nThe goal has been formally proved by the symbolic engine. ✓")
    else:
        parts = [
            "# ⚠️ Proof Attempt — Goal Not Reached\n",
            f"**Query:** *{request.query}*\n",
            "The solver could not establish the goal from the given facts.\n\n",
        ]
        parts.append("<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n")
        parts.append("## Attempted Steps\n")
        if not request.execution_trace:
            parts.append("No rules were triggered — the given facts do not satisfy any theorem preconditions.\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Step {i+1}: `{step.rule_id}`")
            parts.append(f"- **Rule:** `{step.fired_rule_repr}`")
            parts.append(f"- **New facts:** `{', '.join(step.new_facts)}`\n")
        parts.append("\n</details>\n")
        parts.append("\n## Analysis\n⚠️ **Logical gap detected.** Either the initial conditions are insufficient, or the Knowledge Base is missing a bridging theorem.")

    return ExplainResponse(explanation="\n".join(parts), structured=False)


@app.post("/api/explain/stream", tags=["Explainability Agent"])
async def explain_proof_stream(request: ExplainRequest):
    """Streaming proof explanation — LLM real-time or template chunk stream."""
    trace_text = "\n".join(
        f"Step {i+1}: [{s.rule_id}] {s.fired_rule_repr} → New: {s.new_facts}"
        for i, s in enumerate(request.execution_trace)
    )
    aux_text = ""
    if request.auxiliary_constructions:
        aux_text = "\nAuxiliary constructions added: " + ", ".join(request.auxiliary_constructions)

    from rag_agent.llm_factory import get_llm
    llm = get_llm(temperature=0.3)

    if llm:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", _build_explain_system_prompt(request.goal_reached)),
                ("human", "Query: '{query}'\n\nProof Trace:\n{trace}{aux}")
            ])

            async def generate_llm():
                chain = prompt | llm
                async for chunk in chain.astream({
                    "query": request.query,
                    "trace": trace_text or "No rules triggered.",
                    "aux": aux_text,
                }):
                    content = chunk.content
                    if not content:
                        continue
                    yield content if isinstance(content, str) else str(content)

            return StreamingResponse(generate_llm(), media_type="text/plain")
        except Exception as e:
            logger.warning("Streaming LLM explanation failed: %s", e)

    # Template stream fallback
    if request.goal_reached:
        parts = [
            "# Geometry Proof Explanation\n\n",
            f"**Query:** *{request.query}*\n\n",
        ]
        if request.auxiliary_constructions:
            parts.append(f"**Auxiliary Constructions:** {', '.join(request.auxiliary_constructions)}\n\n")
        
        parts.append("<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n")
        parts.append("## Deduction Steps\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Step {i+1}: `{step.rule_id}`\n")
            parts.append(f"- **Rule:** `{step.fired_rule_repr}`\n")
            parts.append(f"- **New facts:** `{', '.join(step.new_facts)}`\n\n")
        parts.append("</details>\n\n")
        parts.append("## ✓ Conclusion\nGoal formally proved by the symbolic engine.\n")
    else:
        parts = [
            "# ⚠️ Proof Attempt — Goal Not Reached\n\n",
            f"**Query:** *{request.query}*\n\n",
            "The solver could not establish the goal from the given facts.\n\n",
        ]
        parts.append("<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n")
        if not request.execution_trace:
            parts.append("No rules were triggered.\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Step {i+1}: `{step.rule_id}`\n")
            parts.append(f"- **Rule:** `{step.fired_rule_repr}`\n")
            parts.append(f"- **New facts:** `{', '.join(step.new_facts)}`\n\n")
        parts.append("</details>\n\n")
        parts.append("## Analysis\n⚠️ Logical gap detected — KB may be missing a bridging theorem.\n")

    async def generate_template():
        for chunk in parts:
            yield chunk
            await asyncio.sleep(0.03)

    return StreamingResponse(generate_template(), media_type="text/plain")


@app.get("/rules", tags=["Knowledge Graph"])
@app.get("/api/rules", tags=["Knowledge Graph"])
async def get_rules(domain: Optional[str] = Query(None, description="Filter by domain (always geometry)")):
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
                    "id": rec["id"], "name": rec["name"], "domain": rec["domain"],
                    "inputs": rec["inputs"], "outputs": rec["outputs"],
                    "description": rec["description"]
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
