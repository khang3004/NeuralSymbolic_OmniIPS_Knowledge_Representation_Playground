"""
GeoIPS — Ontology Ingestion Pipeline.

Reads the OWL Turtle file (ontology/geometry_ontology.ttl) and populates
Neo4j with OntologyClass nodes and IS_A relationships.
Also links existing geometry Fact nodes to their corresponding OntologyClass
via INSTANCE_OF relationships.
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from graph_db.connection import Neo4jConnection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_ontology")

# ---------------------------------------------------------------------------
# Manual OWL class definitions (mirrors geometry_ontology.ttl)
# Avoids requiring rdflib as a hard dependency.
# ---------------------------------------------------------------------------

ONTOLOGY_CLASSES = [
    # (name, uri, parent_name, label, comment)
    ("GeometricObject", "geo:GeometricObject", None, "Geometric Object", "Root of all geometric entities."),
    ("Shape", "geo:Shape", "GeometricObject", "Shape", "A geometric shape."),
    ("Polygon", "geo:Polygon", "Shape", "Polygon", "A closed plane figure bounded by segments."),
    ("Triangle", "geo:Triangle", "Polygon", "Triangle", "A three-sided polygon. Angles sum to 180°."),
    ("RightTriangle", "geo:RightTriangle", "Triangle", "Right Triangle", "A triangle with one 90° angle."),
    ("IsoscelesTriangle", "geo:IsoscelesTriangle", "Triangle", "Isosceles Triangle", "A triangle with two congruent sides."),
    ("EquilateralTriangle", "geo:EquilateralTriangle", "IsoscelesTriangle", "Equilateral Triangle", "All sides congruent, all angles 60°."),
    ("ScaleneTriangle", "geo:ScaleneTriangle", "Triangle", "Scalene Triangle", "No congruent sides."),
    ("Quadrilateral", "geo:Quadrilateral", "Polygon", "Quadrilateral", "A four-sided polygon."),
    ("Trapezoid", "geo:Trapezoid", "Quadrilateral", "Trapezoid", "One pair of parallel sides."),
    ("IsoscelesTrapezoid", "geo:IsoscelesTrapezoid", "Trapezoid", "Isosceles Trapezoid", "Trapezoid with congruent legs."),
    ("Parallelogram", "geo:Parallelogram", "Quadrilateral", "Parallelogram", "Two pairs of parallel sides."),
    ("Rectangle", "geo:Rectangle", "Parallelogram", "Rectangle", "Parallelogram with four right angles."),
    ("Rhombus", "geo:Rhombus", "Parallelogram", "Rhombus", "Parallelogram with four congruent sides."),
    ("Square", "geo:Square", "Rectangle", "Square", "Rectangle with all sides congruent."),
    ("Circle", "geo:Circle", "Shape", "Circle", "All points equidistant from center."),
    ("Line", "geo:Line", "GeometricObject", "Line", "Infinite straight figure."),
    ("Ray", "geo:Ray", "Line", "Ray", "Half-line from an endpoint."),
    ("Segment", "geo:Segment", "Line", "Segment", "Finite portion of a line."),
    ("Angle", "geo:Angle", "GeometricObject", "Angle", "Figure formed by two rays from a vertex."),
    ("AcuteAngle", "geo:AcuteAngle", "Angle", "Acute Angle", "0° < angle < 90°."),
    ("RightAngle", "geo:RightAngle", "Angle", "Right Angle", "Angle = 90°."),
    ("ObtuseAngle", "geo:ObtuseAngle", "Angle", "Obtuse Angle", "90° < angle < 180°."),
    ("StraightAngle", "geo:StraightAngle", "Angle", "Straight Angle", "Angle = 180°."),
    ("ReflexAngle", "geo:ReflexAngle", "Angle", "Reflex Angle", "180° < angle < 360°."),
    ("Point", "geo:Point", "GeometricObject", "Point", "A zero-dimensional location."),
    ("Midpoint", "geo:Midpoint", "Point", "Midpoint", "Divides a segment into two equal halves."),
    ("Centroid", "geo:Centroid", "Point", "Centroid", "Intersection of the three medians."),
    ("Circumcenter", "geo:Circumcenter", "Point", "Circumcenter", "Center of the circumscribed circle."),
    ("Incenter", "geo:Incenter", "Point", "Incenter", "Center of the inscribed circle."),
    ("Orthocenter", "geo:Orthocenter", "Point", "Orthocenter", "Intersection of the three altitudes."),
]

# Fact-predicate functor → OntologyClass name mapping
# When a Fact with value "Triangle(A,B,C)" is ingested, it gets linked
# to the "Triangle" OntologyClass via INSTANCE_OF.
FUNCTOR_TO_CLASS = {
    "Triangle": "Triangle",
    "RightTriangle": "RightTriangle",
    "CongruentTriangles": "Triangle",
    "SimilarTriangles": "Triangle",
    "Quadrilateral": "Quadrilateral",
    "Parallelogram": "Parallelogram",
    "Rectangle": "Rectangle",
    "Rhombus": "Rhombus",
    "Square": "Square",
    "Trapezoid": "Trapezoid",
    "Circle": "Circle",
    "Segment": "Segment",
    "Angle": "Angle",
    "RightAngle": "RightAngle",
    "AcuteAngle": "AcuteAngle",
    "ObtuseAngle": "ObtuseAngle",
    "Perpendicular": "Line",
    "Parallel": "Line",
    "Diameter": "Segment",
    "Chord": "Segment",
    "Midpoint": "Midpoint",
    "Centroid": "Centroid",
}


class OntologyIngest:
    def __init__(self):
        self.neo4j_conn = Neo4jConnection()

    def load_classes(self):
        """Create OntologyClass nodes and IS_A relationships in Neo4j."""
        logger.info("Loading %d ontology classes into Neo4j...", len(ONTOLOGY_CLASSES))
        with self.neo4j_conn.get_session() as session:
            # Create class nodes
            session.run("""
                UNWIND $classes AS row
                MERGE (c:OntologyClass {name: row.name})
                ON CREATE SET c.uri = row.uri, c.label = row.label, c.comment = row.comment
                ON MATCH  SET c.uri = row.uri, c.label = row.label, c.comment = row.comment
            """, classes=[
                {"name": n, "uri": u, "label": l, "comment": c}
                for n, u, _, l, c in ONTOLOGY_CLASSES
            ])
            logger.info("OntologyClass nodes created/updated.")

            # Create IS_A relationships
            isa_pairs = [
                {"child": n, "parent": p}
                for n, _, p, _, _ in ONTOLOGY_CLASSES
                if p is not None
            ]
            session.run("""
                UNWIND $pairs AS row
                MATCH (child:OntologyClass {name: row.child})
                MATCH (parent:OntologyClass {name: row.parent})
                MERGE (child)-[:IS_A]->(parent)
            """, pairs=isa_pairs)
            logger.info("IS_A relationships created: %d", len(isa_pairs))

    def link_facts_to_classes(self):
        """
        Link existing geometry Fact nodes to OntologyClass nodes via INSTANCE_OF.
        Matches based on the predicate functor (first word before '(').
        """
        logger.info("Linking geometry Facts to OntologyClass nodes via INSTANCE_OF...")
        with self.neo4j_conn.get_session() as session:
            result = session.run(
                "MATCH (f:Fact {domain: 'geometry'}) RETURN f.value AS value, elementId(f) AS eid"
            )
            facts = [(rec["value"], rec["eid"]) for rec in result]
            logger.info("Found %d geometry facts to link.", len(facts))

            links_created = 0
            for value, eid in facts:
                # Extract functor from predicate value
                import re
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", value)
                if not m:
                    continue
                functor = m.group(1)
                class_name = FUNCTOR_TO_CLASS.get(functor)
                if not class_name:
                    continue

                session.run("""
                    MATCH (f:Fact) WHERE elementId(f) = $eid
                    MATCH (c:OntologyClass {name: $class_name})
                    MERGE (f)-[:INSTANCE_OF]->(c)
                """, eid=eid, class_name=class_name)
                links_created += 1

        logger.info("INSTANCE_OF links created: %d", links_created)


    def run(self):
        self.load_classes()
        self.link_facts_to_classes()
        logger.info("✅ Ontology ingestion complete.")

    def close(self):
        self.neo4j_conn.close()


if __name__ == "__main__":
    ingestor = OntologyIngest()
    try:
        ingestor.run()
    finally:
        ingestor.close()
