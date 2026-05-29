
import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_db.connection import Neo4jConnection
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_geometry")

DOMAIN = "geometry"
COLLECTION_NAME = "geometry_facts"

# =============================================================================
# COMPREHENSIVE EUCLIDEAN PLANE GEOMETRY KNOWLEDGE BASE
# Covers the full high-school curriculum of classical theorems and axioms.
# =============================================================================

GEOMETRY_KNOWLEDGE = [
    # =========================================================================
    # A. CONGRUENCE & EQUALITY PROPERTIES (7 rules)
    # =========================================================================
    {
        "id": "geo_congruence_reflexive",
        "name": "Congruence Reflexive Property",
        "inputs": ["Segment(AB)"],
        "outputs": ["Congruent(AB,AB)"],
        "description": "Any segment is congruent to itself. AB ≅ AB. This is Euclid's Common Notion 4."
    },
    {
        "id": "geo_congruence_symmetric",
        "name": "Congruence Symmetric Property",
        "inputs": ["Congruent(AB,CD)"],
        "outputs": ["Congruent(CD,AB)"],
        "description": "If AB ≅ CD, then CD ≅ AB. Congruence is a symmetric relation."
    },
    {
        "id": "geo_congruence_transitive",
        "name": "Congruence Transitivity",
        "inputs": ["Congruent(AB,CD)", "Congruent(CD,EF)"],
        "outputs": ["Congruent(AB,EF)"],
        "description": "If segment AB is congruent to CD, and CD is congruent to EF, then AB is congruent to EF. Euclid's Common Notion 1."
    },
    {
        "id": "geo_sas_congruence",
        "name": "SAS Congruence (Side-Angle-Side)",
        "inputs": ["Congruent(AB,DE)", "Equal(Angle(BAC),Angle(EDF))", "Congruent(AC,DF)"],
        "outputs": ["CongruentTriangles(ABC,DEF)"],
        "description": "If two sides and the included angle of one triangle are congruent to two sides and the included angle of another, the triangles are congruent. Euclid Proposition I.4."
    },
    {
        "id": "geo_asa_congruence",
        "name": "ASA Congruence (Angle-Side-Angle)",
        "inputs": ["Equal(Angle(BAC),Angle(EDF))", "Congruent(AB,DE)", "Equal(Angle(ABC),Angle(DEF))"],
        "outputs": ["CongruentTriangles(ABC,DEF)"],
        "description": "If two angles and the included side of one triangle are congruent to two angles and the included side of another, the triangles are congruent. Euclid Proposition I.26."
    },
    {
        "id": "geo_sss_congruence",
        "name": "SSS Congruence (Side-Side-Side)",
        "inputs": ["Congruent(AB,DE)", "Congruent(BC,EF)", "Congruent(AC,DF)"],
        "outputs": ["CongruentTriangles(ABC,DEF)"],
        "description": "If three sides of one triangle are congruent to three sides of another, the triangles are congruent. Euclid Proposition I.8."
    },
    {
        "id": "geo_aas_congruence",
        "name": "AAS Congruence (Angle-Angle-Side)",
        "inputs": ["Equal(Angle(BAC),Angle(EDF))", "Equal(Angle(ABC),Angle(DEF))", "Congruent(BC,EF)"],
        "outputs": ["CongruentTriangles(ABC,DEF)"],
        "description": "If two angles and a non-included side of one triangle are congruent to the corresponding parts of another, the triangles are congruent."
    },

    # =========================================================================
    # B. TRIANGLE THEOREMS (8 rules)
    # =========================================================================
    {
        "id": "geo_triangle_angle_sum",
        "name": "Triangle Angle Sum Theorem",
        "inputs": ["Triangle(A,B,C)"],
        "outputs": ["Equal(Add(Angle(BAC),Angle(ABC),Angle(ACB)),180)"],
        "description": "The sum of interior angles of any triangle equals 180°. Euclid Proposition I.32."
    },
    {
        "id": "geo_isosceles_base_angles",
        "name": "Isosceles Triangle Base Angles",
        "inputs": ["Triangle(A,B,C)", "Congruent(AB,AC)"],
        "outputs": ["Equal(Angle(ABC),Angle(ACB))"],
        "description": "In an isosceles triangle, the base angles are equal. If AB ≅ AC then ∠B = ∠C. Euclid Proposition I.5 (Pons Asinorum)."
    },
    {
        "id": "geo_isosceles_reverse",
        "name": "Converse of Isosceles Triangle Theorem",
        "inputs": ["Triangle(A,B,C)", "Equal(Angle(ABC),Angle(ACB))"],
        "outputs": ["Congruent(AB,AC)"],
        "description": "If two angles of a triangle are equal, the sides opposite them are congruent. Euclid Proposition I.6."
    },
    {
        "id": "geo_exterior_angle",
        "name": "Exterior Angle Theorem",
        "inputs": ["Triangle(A,B,C)", "ExteriorAngle(ACD,C)"],
        "outputs": ["Equal(ExteriorAngle(ACD,C),Add(Angle(BAC),Angle(ABC)))"],
        "description": "An exterior angle of a triangle equals the sum of the two remote interior angles. Euclid Proposition I.32."
    },
    {
        "id": "geo_pythagoras",
        "name": "Pythagorean Theorem",
        "inputs": ["RightTriangle(A,B,C)", "RightAngle(Angle(BAC))"],
        "outputs": [
            "Equal(Pow(BC,2),Add(Pow(AB,2),Pow(AC,2)))",
            "BC^2=AB^2+AC^2",
            "Equal(BC^2,Add(AB^2,AC^2))"
        ],
        "description": "In right triangle ABC, right-angled at A, the square of the hypotenuse BC is equal to the sum of squares of the other two sides: BC² = AB² + AC²."
    },
    {
        "id": "geo_pythagoras_converse",
        "name": "Converse of Pythagorean Theorem",
        "inputs": ["Triangle(A,B,C)", "BC^2=AB^2+AC^2"],
        "outputs": ["RightTriangle(A,B,C)", "RightAngle(Angle(BAC))"],
        "description": "If in triangle ABC, BC² = AB² + AC², then the triangle is a right triangle with the right angle at A."
    },
    {
        "id": "geo_midpoint_theorem",
        "name": "Triangle Midpoint (Midsegment) Theorem",
        "inputs": ["Triangle(A,B,C)", "Midpoint(M,AB)", "Midpoint(N,AC)"],
        "outputs": ["Parallel(MN,BC)", "Equal(Length(MN),Div(Length(BC),2))"],
        "description": "The segment connecting the midpoints of two sides of a triangle is parallel to the third side and half its length."
    },
    {
        "id": "geo_triangle_inequality",
        "name": "Triangle Inequality Theorem",
        "inputs": ["Triangle(A,B,C)"],
        "outputs": ["GreaterThan(Add(Length(AB),Length(BC)),Length(AC))", "GreaterThan(Add(Length(AB),Length(AC)),Length(BC))", "GreaterThan(Add(Length(BC),Length(AC)),Length(AB))"],
        "description": "The sum of any two sides of a triangle is greater than the third side."
    },

    # =========================================================================
    # C. SIMILARITY (5 rules)
    # =========================================================================
    {
        "id": "geo_aa_similarity",
        "name": "AA Similarity Criterion",
        "inputs": ["Triangle(A,B,C)", "Triangle(D,E,F)", "Equal(Angle(BAC),Angle(EDF))", "Equal(Angle(ABC),Angle(DEF))"],
        "outputs": ["SimilarTriangles(ABC,DEF)"],
        "description": "If two angles of one triangle are equal to two angles of another, the triangles are similar. (The third angles are automatically equal since angles sum to 180°.)"
    },
    {
        "id": "geo_sas_similarity",
        "name": "SAS Similarity Criterion",
        "inputs": ["Triangle(A,B,C)", "Triangle(D,E,F)", "Equal(Div(Length(AB),Length(DE)),Div(Length(AC),Length(DF)))", "Equal(Angle(BAC),Angle(EDF))"],
        "outputs": ["SimilarTriangles(ABC,DEF)"],
        "description": "If two sides of one triangle are proportional to two sides of another and the included angles are equal, the triangles are similar."
    },
    {
        "id": "geo_sss_similarity",
        "name": "SSS Similarity Criterion",
        "inputs": ["Triangle(A,B,C)", "Triangle(D,E,F)", "Equal(Div(Length(AB),Length(DE)),Div(Length(BC),Length(EF)))", "Equal(Div(Length(BC),Length(EF)),Div(Length(AC),Length(DF)))"],
        "outputs": ["SimilarTriangles(ABC,DEF)"],
        "description": "If three sides of one triangle are proportional to three sides of another, the triangles are similar."
    },
    {
        "id": "geo_similar_corresponding_sides",
        "name": "Similar Triangles Proportional Sides",
        "inputs": ["SimilarTriangles(ABC,DEF)"],
        "outputs": ["Equal(Div(Length(AB),Length(DE)),Div(Length(BC),Length(EF)))", "Equal(Div(Length(BC),Length(EF)),Div(Length(AC),Length(DF)))"],
        "description": "If two triangles are similar, their corresponding sides are proportional: AB/DE = BC/EF = AC/DF."
    },
    {
        "id": "geo_similar_corresponding_angles",
        "name": "Similar Triangles Equal Angles",
        "inputs": ["SimilarTriangles(ABC,DEF)"],
        "outputs": ["Equal(Angle(BAC),Angle(EDF))", "Equal(Angle(ABC),Angle(DEF))", "Equal(Angle(ACB),Angle(DFE))"],
        "description": "If two triangles are similar, their corresponding angles are equal."
    },

    # =========================================================================
    # D. PARALLEL & PERPENDICULAR LINES (7 rules)
    # =========================================================================
    {
        "id": "geo_parallel_transitive",
        "name": "Parallel Transitivity",
        "inputs": ["Parallel(a,b)", "Parallel(b,c)"],
        "outputs": ["Parallel(a,c)"],
        "description": "If line a is parallel to line b, and line b is parallel to line c, then line a is parallel to line c."
    },
    {
        "id": "geo_parallel_alt_int_angles",
        "name": "Parallel Lines Alternate Interior Angles",
        "inputs": ["Parallel(L1,L2)", "Transversal(T,L1,L2)"],
        "outputs": ["Equal(AlternateInteriorAngle(T,L1),AlternateInteriorAngle(T,L2))"],
        "description": "If two parallel lines are cut by a transversal, the alternate interior angles are equal. Euclid Proposition I.29."
    },
    {
        "id": "geo_parallel_corresponding_angles",
        "name": "Parallel Lines Corresponding Angles",
        "inputs": ["Parallel(L1,L2)", "Transversal(T,L1,L2)"],
        "outputs": ["Equal(CorrespondingAngle(T,L1),CorrespondingAngle(T,L2))"],
        "description": "If two parallel lines are cut by a transversal, the corresponding angles are equal."
    },
    {
        "id": "geo_parallel_co_interior",
        "name": "Parallel Lines Co-Interior (Same-Side Interior) Angles",
        "inputs": ["Parallel(L1,L2)", "Transversal(T,L1,L2)"],
        "outputs": ["Equal(Add(CoInteriorAngle(T,L1),CoInteriorAngle(T,L2)),180)"],
        "description": "If two parallel lines are cut by a transversal, the co-interior angles are supplementary (sum to 180°)."
    },
    {
        "id": "geo_perp_symmetry",
        "name": "Perpendicular Symmetry",
        "inputs": ["Perpendicular(AB,CD)"],
        "outputs": ["Perpendicular(CD,AB)"],
        "description": "If line AB is perpendicular to CD, then CD is perpendicular to AB."
    },
    {
        "id": "geo_perp_to_parallel",
        "name": "Perpendicular to One Parallel Line",
        "inputs": ["Perpendicular(L,a)", "Parallel(a,b)"],
        "outputs": ["Perpendicular(L,b)"],
        "description": "If a line is perpendicular to one of two parallel lines, it is perpendicular to the other."
    },
    {
        "id": "geo_parallel_from_perp",
        "name": "Two Lines Perpendicular to Same Line",
        "inputs": ["Perpendicular(L,a)", "Perpendicular(L,b)"],
        "outputs": ["Parallel(a,b)"],
        "description": "If two lines are both perpendicular to the same transversal, they are parallel to each other."
    },

    # =========================================================================
    # E. CIRCLE THEOREMS (7 rules)
    # =========================================================================
    {
        "id": "geo_thales",
        "name": "Thales' Theorem",
        "inputs": ["Diameter(AB,Circle(O))", "PointOnCircle(C,Circle(O))"],
        "outputs": ["RightAngle(Angle(ACB))"],
        "description": "An angle inscribed in a semicircle is a right angle (90°). If AB is a diameter and C is on the circle, then ∠ACB = 90°."
    },
    {
        "id": "geo_inscribed_angle",
        "name": "Inscribed Angle Theorem",
        "inputs": ["InscribedAngle(ABC,Circle(O))", "CentralAngle(AOC,Circle(O))"],
        "outputs": ["Equal(CentralAngle(AOC),Mul(2,InscribedAngle(ABC)))"],
        "description": "A central angle is twice the inscribed angle that subtends the same arc."
    },
    {
        "id": "geo_inscribed_same_arc",
        "name": "Inscribed Angles on Same Arc",
        "inputs": ["InscribedAngle(ABC,Circle(O))", "InscribedAngle(ADC,Circle(O))", "SameArc(AC,ABC,ADC)"],
        "outputs": ["Equal(InscribedAngle(ABC),InscribedAngle(ADC))"],
        "description": "Inscribed angles subtending the same arc are equal."
    },
    {
        "id": "geo_tangent_radius",
        "name": "Tangent-Radius Perpendicularity",
        "inputs": ["Tangent(T,Circle(O),P)"],
        "outputs": ["Perpendicular(T,Radius(OP))"],
        "description": "A tangent to a circle is perpendicular to the radius at the point of tangency."
    },
    {
        "id": "geo_tangent_lengths",
        "name": "Tangent Segments from External Point",
        "inputs": ["Tangent(T1,Circle(O),A)", "Tangent(T2,Circle(O),B)", "ExternalPoint(P,T1,T2)"],
        "outputs": ["Congruent(PA,PB)"],
        "description": "Two tangent segments drawn from the same external point to a circle are congruent."
    },
    {
        "id": "geo_cyclic_quad_opp",
        "name": "Cyclic Quadrilateral Opposite Angles",
        "inputs": ["CyclicQuadrilateral(ABCD,Circle(O))"],
        "outputs": ["Equal(Add(Angle(DAB),Angle(BCD)),180)", "Equal(Add(Angle(ABC),Angle(CDA)),180)"],
        "description": "Opposite angles of a cyclic quadrilateral are supplementary (sum to 180°)."
    },
    {
        "id": "geo_chord_bisector",
        "name": "Perpendicular from Center Bisects Chord",
        "inputs": ["Circle(O)", "Chord(AB,Circle(O))", "Perpendicular(OM,AB)"],
        "outputs": ["Midpoint(M,AB)"],
        "description": "A perpendicular from the center of a circle to a chord bisects the chord."
    },

    # =========================================================================
    # F. AREA & MEASUREMENT (6 rules)
    # =========================================================================
    {
        "id": "geo_triangle_area",
        "name": "Triangle Area Formula",
        "inputs": ["Triangle(A,B,C)", "Base(BC,b)", "Height(AH,h)"],
        "outputs": ["Equal(Area(Triangle(A,B,C)),Mul(Div(1,2),Mul(b,h)))"],
        "description": "The area of a triangle is one-half the product of its base and corresponding height: A = ½bh."
    },
    {
        "id": "geo_parallelogram_area",
        "name": "Parallelogram Area Formula",
        "inputs": ["Parallelogram(ABCD)", "Base(AB,b)", "Height(h)"],
        "outputs": ["Equal(Area(Parallelogram(ABCD)),Mul(b,h))"],
        "description": "The area of a parallelogram equals base times height: A = bh."
    },
    {
        "id": "geo_circle_area",
        "name": "Circle Area Formula",
        "inputs": ["Circle(O)", "Radius(r)"],
        "outputs": ["Equal(Area(Circle(O)),Mul(Pi,Pow(r,2)))"],
        "description": "The area of a circle is π times the square of its radius: A = πr²."
    },
    {
        "id": "geo_circle_circumference",
        "name": "Circle Circumference Formula",
        "inputs": ["Circle(O)", "Radius(r)"],
        "outputs": ["Equal(Circumference(Circle(O)),Mul(2,Mul(Pi,r)))"],
        "description": "The circumference of a circle is 2π times its radius: C = 2πr."
    },
    {
        "id": "geo_trapezoid_area",
        "name": "Trapezoid Area Formula",
        "inputs": ["Trapezoid(ABCD)", "Parallel(AB,CD)", "Base(AB,a)", "Base(CD,b)", "Height(h)"],
        "outputs": ["Equal(Area(Trapezoid(ABCD)),Mul(Div(1,2),Mul(Add(a,b),h)))"],
        "description": "The area of a trapezoid is half the sum of the parallel bases times the height: A = ½(a+b)h."
    },
    {
        "id": "geo_sector_area",
        "name": "Sector Area Formula",
        "inputs": ["Sector(O,A,B)", "Radius(r)", "CentralAngle(theta)"],
        "outputs": ["Equal(Area(Sector(O,A,B)),Mul(Div(theta,360),Mul(Pi,Pow(r,2))))"],
        "description": "The area of a sector is (θ/360) × πr², where θ is the central angle in degrees."
    },

    # =========================================================================
    # G. CONGRUENT TRIANGLES CONSEQUENCES (3 rules)
    # =========================================================================
    {
        "id": "geo_congruent_tri_sides",
        "name": "Congruent Triangles Have Congruent Sides",
        "inputs": ["CongruentTriangles(ABC,DEF)"],
        "outputs": ["Congruent(AB,DE)", "Congruent(BC,EF)", "Congruent(AC,DF)"],
        "description": "If two triangles are congruent, all corresponding sides are congruent: AB≅DE, BC≅EF, AC≅DF."
    },
    {
        "id": "geo_congruent_tri_angles",
        "name": "Congruent Triangles Have Equal Angles",
        "inputs": ["CongruentTriangles(ABC,DEF)"],
        "outputs": ["Equal(Angle(BAC),Angle(EDF))", "Equal(Angle(ABC),Angle(DEF))", "Equal(Angle(ACB),Angle(DFE))"],
        "description": "If two triangles are congruent, all corresponding angles are equal: ∠A=∠D, ∠B=∠E, ∠C=∠F."
    },
    {
        "id": "geo_equilateral_angles",
        "name": "Equilateral Triangle Angles",
        "inputs": ["Triangle(A,B,C)", "Congruent(AB,BC)", "Congruent(BC,AC)"],
        "outputs": ["Equal(Angle(BAC),60)", "Equal(Angle(ABC),60)", "Equal(Angle(ACB),60)"],
        "description": "An equilateral triangle has all angles equal to 60°."
    },
]


class GeometryIngest:
    def __init__(self):
        self.neo4j_conn = Neo4jConnection()
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_to_neo4j(self):
        logger.info("Loading %d Geometry axioms/theorems to Neo4j...", len(GEOMETRY_KNOWLEDGE))
        with self.neo4j_conn.get_session() as session:
            session.run("""
                UNWIND $batch AS row
                MERGE (r:Rule {id: row.id, domain: $domain})
                ON CREATE SET r.name = row.name, r.description = row.description,
                              r.inputs = row.inputs, r.outputs = row.outputs
                ON MATCH SET r.name = row.name, r.description = row.description,
                             r.inputs = row.inputs, r.outputs = row.outputs
                SET r:Geometry
            """, batch=GEOMETRY_KNOWLEDGE, domain=DOMAIN)
            
            all_facts = set()
            for r in GEOMETRY_KNOWLEDGE:
                all_facts.update(r["inputs"])
                all_facts.update(r["outputs"])
            
            logger.info("Extracted %d unique facts from geometry rules.", len(all_facts))
            fact_data = [{"value": f, "label": f, "domain": DOMAIN} for f in all_facts]
            
            session.run("""
                UNWIND $batch AS row
                MERGE (f:Fact {value: row.value, domain: row.domain})
                ON CREATE SET f.id = 'geo_fact_' + row.value, f.label = row.label
                SET f:Geometry
            """, batch=fact_data)
            
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
            """, batch=GEOMETRY_KNOWLEDGE, domain=DOMAIN)

    def load_to_qdrant(self):
        logger.info("Loading Geometry facts to Qdrant...")
        all_facts = set()
        for r in GEOMETRY_KNOWLEDGE:
            all_facts.update(r["inputs"])
            all_facts.update(r["outputs"])
            
        try:
            from qdrant_client.models import Distance, VectorParams
            self.qdrant_client.delete_collection(COLLECTION_NAME)
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info("Recreated collection '%s' to clear stale vectors.", COLLECTION_NAME)
        except Exception as e:
            logger.warning("Failed to recreate collection '%s': %s", COLLECTION_NAME, e)

        points = []
        for f_val in all_facts:
            vector = self.embed_model.encode(f_val).tolist()
            points.append(PointStruct(
                id=abs(hash(f_val)) % (10**15),
                vector=vector,
                payload={"value": f_val, "label": f_val, "domain": DOMAIN}
            ))
            
        if points:
            # Batch upsert in chunks of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(COLLECTION_NAME, batch)
                logger.info("  Upserted batch %d/%d (%d points)", i // batch_size + 1, (len(points) + batch_size - 1) // batch_size, len(batch))
        
        logger.info("Total geometry facts loaded to Qdrant: %d", len(points))

    def run(self):
        self.load_to_neo4j()
        self.load_to_qdrant()
        logger.info("✅ Geometry ingestion complete. %d rules, %d unique facts.",
                     len(GEOMETRY_KNOWLEDGE),
                     len(set(f for r in GEOMETRY_KNOWLEDGE for f in r["inputs"] + r["outputs"])))

if __name__ == "__main__":
    GeometryIngest().run()
