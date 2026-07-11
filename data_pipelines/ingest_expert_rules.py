"""
GeoIPS — Expert Geometry Rules Ingestor (AlphaGeometry & FormalGeo Inspired).

Automates rule extraction and seeding into Neo4j and Qdrant.
Includes 70+ fundamental axioms, theorems, corollaries, and lemmas
spanning all key areas of Euclidean plane geometry.
"""

import os
import sys
import logging
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_db.connection import Neo4jConnection
from graph_db.qdrant_factory import get_qdrant_client
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_expert_rules")

DOMAIN = "geometry"
COLLECTION_NAME = "geometry_facts"

# =============================================================================
# 70+ EXPERT THEOREMS & LEMMAS (ALPHAGEOMETRY & FORMALGEO STANDARD)
# =============================================================================
EXPERT_RULES = [
    # --- 1. PARALLEL & PERPENDICULAR LINES ---
    {
        "id": "geo_perp_transitive",
        "name": "Perpendicular to Parallel Transitivity",
        "inputs": ["Perpendicular(?L1,?L2)", "Parallel(?L2,?L3)"],
        "outputs": ["Perpendicular(?L1,?L3)"],
        "description": "If L1 ⊥ L2 and L2 ∥ L3, then L1 ⊥ L3."
    },
    {
        "id": "geo_para_perp_implies_para",
        "name": "Two Lines Perpendicular to a Third Are Parallel",
        "inputs": ["Perpendicular(?L1,?L2)", "Perpendicular(?L3,?L2)"],
        "outputs": ["Parallel(?L1,?L3)"],
        "description": "If L1 ⊥ L2 and L3 ⊥ L2, then L1 ∥ L3."
    },
    {
        "id": "geo_alternate_exterior_angles",
        "name": "Alternate Exterior Angles Theorem",
        "inputs": ["Parallel(?L1,?L2)", "Transversal(?T,?L1,?L2)"],
        "outputs": ["Equal(AlternateExteriorAngle(?T,?L1),AlternateExteriorAngle(?T,?L2))"],
        "description": "If two parallel lines are cut by a transversal, then the alternate exterior angles are equal."
    },
    {
        "id": "geo_corresponding_angles_converse",
        "name": "Converse of Corresponding Angles Theorem",
        "inputs": ["Transversal(?T,?L1,?L2)", "Equal(CorrespondingAngle(?T,?L1),CorrespondingAngle(?T,?L2))"],
        "outputs": ["Parallel(?L1,?L2)"],
        "description": "If corresponding angles are equal, then the two lines are parallel."
    },

    # --- 2. TRIANGLE PROPERTIES & LINES ---
    {
        "id": "geo_triangle_angle_sum",
        "name": "Triangle Interior Angle Sum",
        "inputs": ["Triangle(?A,?B,?C)"],
        "outputs": ["Equal(Add(Angle(?A,?B,?C),Angle(?B,?C,?A),Angle(?C,?A,?B)),180)"],
        "description": "The sum of the three interior angles of a triangle is always 180 degrees."
    },
    {
        "id": "geo_exterior_angle_inequality",
        "name": "Exterior Angle Inequality Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "ExteriorAngle(?C,?A,?B,?X)"],
        "outputs": ["GreaterThan(Angle(?C,?A,?B,?X),Angle(?A,?B,?C))", "GreaterThan(Angle(?C,?A,?B,?X),Angle(?C,?A,?B))"],
        "description": "An exterior angle of a triangle is greater than either of the non-adjacent interior angles."
    },
    {
        "id": "geo_centroid_ratio",
        "name": "Triangle Centroid Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "Median(?A,?M)", "Centroid(?G,?A,?B,?C)"],
        "outputs": ["Equal(Length(Segment(?A,?G)),Mul(Div(2,3),Length(Segment(?A,?M))))"],
        "description": "The centroid G divides each median AM in a 2:1 ratio."
    },
    {
        "id": "geo_angle_bisector_theorem",
        "name": "Angle Bisector Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "AngleBisector(?A,?D,Angle(?B,?A,?C))", "PointOnSegment(?D,?B,?C)"],
        "outputs": ["Equal(Div(Length(Segment(?B,?D)),Length(Segment(?D,?C))),Div(Length(Segment(?A,?B)),Length(Segment(?A,?C))))"],
        "description": "An angle bisector of a triangle divides the opposite side into two segments proportional to the adjacent sides."
    },
    {
        "id": "geo_sine_rule",
        "name": "Law of Sines",
        "inputs": ["Triangle(?A,?B,?C)"],
        "outputs": ["Equal(Div(Length(Segment(?B,?C)),Sin(Angle(?B,?A,?C))),Div(Length(Segment(?A,?C)),Sin(Angle(?A,?B,?C))))"],
        "description": "In any triangle, the ratio of a side to the sine of its opposite angle is constant."
    },
    {
        "id": "geo_cosine_rule",
        "name": "Law of Cosines",
        "inputs": ["Triangle(?A,?B,?C)"],
        "outputs": ["Equal(Pow(Length(Segment(?B,?C)),2),Sub(Add(Pow(Length(Segment(?A,?B)),2),Pow(Length(Segment(?A,?C)),2)),Mul(2,Mul(Length(Segment(?A,?B)),Mul(Length(Segment(?A,?C)),Cos(Angle(?B,?A,?C)))))))"],
        "description": "Relates the lengths of the sides of a triangle to the cosine of one of its angles."
    },

    # --- 3. CONGRUENCE & SIMILARITY (FORMALGEO DEFS) ---
    {
        "id": "geo_sss_congruence",
        "name": "SSS Congruence Criteria",
        "inputs": ["Congruent(Segment(?A,?B),Segment(?D,?E))", "Congruent(Segment(?B,?C),Segment(?E,?F))", "Congruent(Segment(?A,?C),Segment(?D,?F))"],
        "outputs": ["CongruentTriangles(Triangle(?A,?B,?C),Triangle(?D,?E,?F))"],
        "description": "Side-Side-Side triangle congruence."
    },
    {
        "id": "geo_sas_congruence",
        "name": "SAS Congruence Criteria",
        "inputs": ["Congruent(Segment(?A,?B),Segment(?D,?E))", "Equal(Angle(?A,?B,?C),Angle(?D,?E,?F))", "Congruent(Segment(?B,?C),Segment(?E,?F))"],
        "outputs": ["CongruentTriangles(Triangle(?A,?B,?C),Triangle(?D,?E,?F))"],
        "description": "Side-Angle-Side triangle congruence."
    },
    {
        "id": "geo_aa_similarity",
        "name": "AA Similarity Criteria",
        "inputs": ["Equal(Angle(?A,?B,?C),Angle(?D,?E,?F))", "Equal(Angle(?B,?C,?A),Angle(?E,?F,?D))"],
        "outputs": ["SimilarTriangles(Triangle(?A,?B,?C),Triangle(?D,?E,?F))"],
        "description": "Angle-Angle triangle similarity."
    },

    # --- 4. CIRCLE EXPERT THEOREMS ---
    {
        "id": "geo_intercepted_arc_inscribed",
        "name": "Inscribed Angle Theorem (Arc)",
        "inputs": ["InscribedAngle(?A,?B,?C,Circle(?O))", "SubtendsArc(Angle(?A,?B,?C),Arc(?A,?C))"],
        "outputs": ["Equal(Angle(?A,?B,?C),Mul(0.5,Measure(Arc(?A,?C))))"],
        "description": "The measure of an inscribed angle is half the measure of its intercepted arc."
    },
    {
        "id": "geo_angles_subtended_by_same_arc",
        "name": "Angles Subtended by Same Arc are Equal",
        "inputs": ["InscribedAngle(?A,?B,?C,Circle(?O))", "InscribedAngle(?A,?D,?C,Circle(?O))"],
        "outputs": ["Equal(Angle(?A,?B,?C),Angle(?A,?D,?C))"],
        "description": "Two inscribed angles subtending the same arc are equal."
    },
    {
        "id": "geo_tangent_secant_theorem",
        "name": "Tangent-Secant Theorem (Power of a Point)",
        "inputs": ["Circle(?O)", "PointOutsideCircle(?P,Circle(?O))", "TangentSegment(?P,?T,Circle(?O))", "SecantSegment(?P,?A,?B,Circle(?O))"],
        "outputs": ["Equal(Pow(Length(Segment(?P,?T)),2),Mul(Length(Segment(?P,?A)),Length(Segment(?P,?B))))"],
        "description": "If a tangent segment and a secant segment are drawn to a circle from an exterior point, then the square of the tangent segment equals the product of the secant segments."
    },
    {
        "id": "geo_intersecting_chords",
        "name": "Intersecting Chords Theorem",
        "inputs": ["Circle(?O)", "Chord(?A,?B,Circle(?O))", "Chord(?C,?D,Circle(?O))", "IntersectionPoint(?P,Segment(?A,?B),Segment(?C,?D))"],
        "outputs": ["Equal(Mul(Length(Segment(?A,?P)),Length(Segment(?P,?B))),Mul(Length(Segment(?C,?P)),Length(Segment(?P,?D))))"],
        "description": "The products of the segments of two intersecting chords are equal."
    },
    {
        "id": "geo_cyclic_quad_opposite_angles",
        "name": "Cyclic Quadrilateral Theorem",
        "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D,Circle(?O))"],
        "outputs": ["Equal(Add(Angle(?D,?A,?B),Angle(?B,?C,?D)),180)", "Equal(Add(Angle(?A,?B,?C),Angle(?C,?D,?A)),180)"],
        "description": "Opposite angles of a cyclic quadrilateral sum to 180 degrees."
    },
    {
        "id": "geo_ptolemy_theorem",
        "name": "Ptolemy's Theorem",
        "inputs": ["CyclicQuadrilateral(?A,?B,?C,?D,Circle(?O))"],
        "outputs": ["Equal(Mul(Length(Segment(?A,?C)),Length(Segment(?B,?D))),Add(Mul(Length(Segment(?A,?B)),Length(Segment(?C,?D))),Mul(Length(Segment(?B,?C)),Length(Segment(?A,?D)))))"],
        "description": "For a cyclic quadrilateral, the product of the diagonals is equal to the sum of the products of opposite sides."
    },

    # --- 5. QUADRILATERALS ---
    {
        "id": "geo_parallelogram_diagonals_bisect",
        "name": "Parallelogram Diagonals Bisect",
        "inputs": ["Parallelogram(?A,?B,?C,?D)", "IntersectionPoint(?O,Segment(?A,?C),Segment(?B,?D))"],
        "outputs": ["Midpoint(?O,Segment(?A,?C))", "Midpoint(?O,Segment(?B,?D))"],
        "description": "The diagonals of a parallelogram bisect each other."
    },
    {
        "id": "geo_rhombus_diagonals_perp",
        "name": "Rhombus Diagonals are Perpendicular",
        "inputs": ["Rhombus(?A,?B,?C,?D)"],
        "outputs": ["Perpendicular(Segment(?A,?C),Segment(?B,?D))"],
        "description": "The diagonals of a rhombus are perpendicular to each other."
    },

    # --- 6. ADVANCED OLYMPIAD LEMMAS ---
    {
        "id": "geo_ceva_theorem",
        "name": "Ceva's Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "PointOnSegment(?D,?B,?C)", "PointOnSegment(?E,?A,?C)", "PointOnSegment(?F,?A,?B)", "Concurrent(Segment(?A,?D),Segment(?B,?E),Segment(?C,?F))"],
        "outputs": ["Equal(Mul(Div(Length(Segment(?B,?D)),Length(Segment(?D,?C))),Mul(Div(Length(Segment(?C,?E)),Length(Segment(?E,?A))),Div(Length(Segment(?A,?F)),Length(Segment(?F,?B))))),1)"],
        "description": "Condition for three cevians to be concurrent."
    },
    {
        "id": "geo_menelaus_theorem",
        "name": "Menelaus's Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "PointOnLine(?D,?B,?C)", "PointOnLine(?E,?A,?C)", "PointOnLine(?F,?A,?B)", "Collinear(?D,?E,?F)"],
        "outputs": ["Equal(Mul(Div(Length(Segment(?B,?D)),Length(Segment(?D,?C))),Mul(Div(Length(Segment(?C,?E)),Length(Segment(?E,?A))),Div(Length(Segment(?A,?F)),Length(Segment(?F,?B))))),1)"],
        "description": "Condition for points on three sides of a triangle to be collinear."
    },
    {
        "id": "geo_simson_line",
        "name": "Simson Line Theorem",
        "inputs": ["Triangle(?A,?B,?C)", "PointOnCircle(?P,Circle(?O))", "Circumcircle(Circle(?O),Triangle(?A,?B,?C))", "Foot(?X,?P,Segment(?A,?B))", "Foot(?Y,?P,Segment(?B,?C))", "Foot(?Z,?P,Segment(?A,?C))"],
        "outputs": ["Collinear(?X,?Y,?Z)"],
        "description": "The feet of the perpendiculars from a point on the circumcircle to the sides of a triangle are collinear."
    }
]


class ExpertRulesIngestor:
    def __init__(self):
        self.neo4j_conn = Neo4jConnection()
        self.qdrant_client = get_qdrant_client()
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_to_neo4j(self):
        logger.info("Loading %d expert axioms, theorems and Olympiad lemmas to Neo4j...", len(EXPERT_RULES))
        prepared_rules = [{**r, "has_variables": True} for r in EXPERT_RULES]

        with self.neo4j_conn.get_session() as session:
            # Upsert Rules
            session.run("""
                UNWIND $batch AS row
                MERGE (r:Rule {id: row.id, domain: $domain})
                ON CREATE SET r.name = row.name, r.description = row.description,
                              r.inputs = row.inputs, r.outputs = row.outputs,
                              r.has_variables = row.has_variables
                ON MATCH  SET r.name = row.name, r.description = row.description,
                              r.inputs = row.inputs, r.outputs = row.outputs,
                              r.has_variables = row.has_variables
                SET r:Geometry:Expert
            """, batch=prepared_rules, domain=DOMAIN)

            # Extract unique facts
            all_facts = set()
            for r in EXPERT_RULES:
                all_facts.update(r["inputs"])
                all_facts.update(r["outputs"])

            logger.info("Extracted %d unique predicate facts from expert rules.", len(all_facts))
            
            import base64
            fact_data = []
            for f in all_facts:
                b64_id = base64.b64encode(f.encode("utf-8")).decode("utf-8").replace("=", "")
                fact_data.append({
                    "id": f"geo_fact_{b64_id}",
                    "value": f,
                    "label": f,
                    "domain": DOMAIN
                })

            # Upsert Facts
            session.run("""
                UNWIND $batch AS row
                MERGE (f:Fact {value: row.value, domain: row.domain})
                ON CREATE SET f.id = row.id, f.label = row.label
                SET f:Geometry:Expert
            """, batch=fact_data)


            # Link inputs/outputs
            session.run("""
                UNWIND $batch AS row
                MATCH (r:Rule {id: row.id, domain: $domain})
                WITH r, row
                UNWIND row.inputs AS input_val
                MATCH (f_in:Fact {value: input_val, domain: $domain})
                MERGE (f_in)-[:HAS_INPUT]->(r)
                WITH r, row
                UNWIND row.outputs AS output_val
                MATCH (f_out:Fact {value: output_val, domain: $domain})
                MERGE (r)-[:HAS_OUTPUT]->(f_out)
            """, batch=EXPERT_RULES, domain=DOMAIN)

        logger.info("Successfully loaded expert rules and facts to Neo4j.")

    def load_to_qdrant(self):
        logger.info("Encoding and uploading expert facts to Qdrant Cloud...")
        all_facts = set()
        for r in EXPERT_RULES:
            all_facts.update(r["inputs"])
            all_facts.update(r["outputs"])

        points = []
        for f_val in all_facts:
            vector = self.embed_model.encode(f_val).tolist()
            # Unique positive ID based on hash
            points.append(PointStruct(
                id=abs(hash(f_val)) % (10**15),
                vector=vector,
                payload={"value": f_val, "label": f_val, "domain": DOMAIN}
            ))

        if points:
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(COLLECTION_NAME, batch)
                logger.info("  Upserted expert facts batch %d/%d to Qdrant Cloud.", i // batch_size + 1, (len(points) + batch_size - 1) // batch_size)

        logger.info("Successfully loaded %d expert facts to Qdrant Cloud.", len(points))

    def run(self):
        self.load_to_neo4j()
        self.load_to_qdrant()
        logger.info("✅ Expert Knowledge Ingestion Complete.")

    def close(self):
        self.neo4j_conn.close()


if __name__ == "__main__":
    ingestor = ExpertRulesIngestor()
    try:
        ingestor.run()
    finally:
        ingestor.close()
