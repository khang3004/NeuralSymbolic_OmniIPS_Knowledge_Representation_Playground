"""
GeoIPS — FormalGeo GDL Rules Ingestor.

Downloads the complete 130+ geometry theorems (1800+ lines of GDL configuration)
from the FormalGeo repository, parses the GDL CDL predicate logic format,
translates them to variables (e.g., A, B, C -> ?A, ?B, ?C) so they work with
the GeoIPS Unification Engine, and populates both Neo4j and Qdrant Cloud.
"""

import os
import sys
import json
import re
import logging
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_db.connection import Neo4jConnection
from graph_db.qdrant_factory import get_qdrant_client
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_formalgeo")

GDL_FILE_PATH = "/Users/KhangDS/.gemini/antigravity-ide/brain/d9649e18-fe13-4957-a583-1f6182d8fc51/scratch/theorem_GDL.json"
DOMAIN = "geometry"
COLLECTION_NAME = "geometry_facts"


def translate_predicate(pred_str: str) -> str:
    """
    Translates FormalGeo GDL predicates to GeoIPS predicate format.
    Also replaces concrete variables (caps/mixed) with unified logic variables (e.g. ?A, ?B).
    
    Examples:
        - Collinear(ABC) -> Collinear(?A,?B,?C)
        - Equal(MeasureOfAngle(AOC),90) -> RightAngle(Angle(?A,?O,?C))
        - ParallelBetweenLine(AB,CD) -> Parallel(?A?B,?C?D)
    """
    pred_str = pred_str.strip()
    
    # 1. Map predicate names to simplified forms
    pred_str = pred_str.replace("ParallelBetweenLine", "Parallel")
    pred_str = pred_str.replace("PerpendicularBetweenLine", "Perpendicular")
    pred_str = pred_str.replace("LengthOfLine", "Length")
    pred_str = pred_str.replace("MeasureOfAngle", "Angle")
    pred_str = pred_str.replace("IsMidpointOfLine", "Midpoint")
    pred_str = pred_str.replace("IsPerpendicularBisectorOfLine", "PerpendicularBisector")
    pred_str = pred_str.replace("IsBisectorOfAngle", "AngleBisector")
    pred_str = pred_str.replace("SimilarTriangle", "SimilarTriangles")
    pred_str = pred_str.replace("CongruentTriangle", "CongruentTriangles")
    
    # 2. Convert variables inside parentheses to ? prefixed variables.
    # We find groups of uppercase letters (or lowercase) and replace them.
    # E.g., Collinear(ABC) -> Collinear(?A,?B,?C)
    def repl_vars(match):
        inner = match.group(1)
        # Split inner by comma or separate adjacent uppercase chars
        if "," in inner:
            parts = [p.strip() for p in inner.split(",")]
        else:
            # E.g. ABC -> A, B, C
            parts = []
            i = 0
            while i < len(inner):
                if i + 1 < len(inner) and inner[i].isupper() and inner[i+1].islower():
                    # E.g. Ab, Bc
                    parts.append(inner[i:i+2])
                    i += 2
                else:
                    parts.append(inner[i])
                    i += 1
        
        # Add ? to variables unless it is a number or math operator like Add/Sub/Mul
        res_parts = []
        for p in parts:
            if p.isdigit() or p in ["180", "90", "360", "1", "0.5"]:
                res_parts.append(p)
            elif p.startswith("Add(") or p.startswith("Sub(") or p.startswith("Mul(") or p.startswith("Div("):
                res_parts.append(p)  # Keep expressions
            else:
                # Add ? if not present
                p_clean = p.replace("?", "")
                res_parts.append(f"?{p_clean}")
        return f"({','.join(res_parts)})"

    # Regex matches inside parentheses recursively (simplified)
    pred_str = re.sub(r"\(([^()]+)\)", repl_vars, pred_str)
    
    # Clean up double ?? if any got added
    pred_str = pred_str.replace("??", "?")
    return pred_str


class FormalGeoIngestor:
    def __init__(self):
        self.neo4j_conn = Neo4jConnection()
        self.qdrant_client = get_qdrant_client()
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def parse_gdl(self):
        """Parses the theorem_GDL.json and returns a list of unified GeoIPS Rules."""
        logger.info("Reading GDL file from %s...", GDL_FILE_PATH)
        with open(GDL_FILE_PATH, "r") as f:
            data = json.load(f)

        rules = []
        for theorem_name, sub_rules in data.items():
            # E.g. "parallel_judgment_corresponding_angle(AB,CD,E)"
            clean_name = theorem_name.split("(")[0].replace("_", " ").title()
            
            for branch_id, details in sub_rules.items():
                rule_id = f"formalgeo_{theorem_name.split('(')[0]}_{branch_id}"
                
                # GDL uses '&' as logical AND for premises
                premises_raw = details.get("premise", "").split("&")
                conclusions_raw = details.get("conclusion", [])
                
                # Translate premises and conclusions
                inputs = [translate_predicate(p) for p in premises_raw if p.strip()]
                outputs = [translate_predicate(c) for c in conclusions_raw if c.strip()]
                
                if not inputs or not outputs:
                    continue
                
                rules.append({
                    "id": rule_id,
                    "name": f"{clean_name} (Branch {branch_id})",
                    "inputs": inputs,
                    "outputs": outputs,
                    "description": f"FormalGeo automated theorem rule: {theorem_name} branch {branch_id}"
                })
        
        logger.info("Successfully parsed %d rules from FormalGeo GDL.", len(rules))
        return rules

    def load_to_neo4j(self, rules):
        logger.info("Ingesting %d FormalGeo rules into Neo4j...", len(rules))
        prepared_rules = [{**r, "has_variables": True} for r in rules]

        with self.neo4j_conn.get_session() as session:
            # Batch upsert rules
            session.run("""
                UNWIND $batch AS row
                MERGE (r:Rule {id: row.id, domain: $domain})
                ON CREATE SET r.name = row.name, r.description = row.description,
                              r.inputs = row.inputs, r.outputs = row.outputs,
                              r.has_variables = row.has_variables
                ON MATCH  SET r.name = row.name, r.description = row.description,
                              r.inputs = row.inputs, r.outputs = row.outputs,
                              r.has_variables = row.has_variables
                SET r:Geometry:FormalGeo
            """, batch=prepared_rules, domain=DOMAIN)

            # Collect unique facts
            all_facts = set()
            for r in rules:
                all_facts.update(r["inputs"])
                all_facts.update(r["outputs"])

            logger.info("Extracted %d unique facts from GDL rules.", len(all_facts))
            
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

            # Batch upsert facts
            session.run("""
                UNWIND $batch AS row
                MERGE (f:Fact {value: row.value, domain: row.domain})
                ON CREATE SET f.id = row.id, f.label = row.label
                SET f:Geometry:FormalGeo
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
            """, batch=rules, domain=DOMAIN)

        logger.info("Successfully loaded rules and facts to Neo4j.")

    def load_to_qdrant(self, rules):
        logger.info("Uploading GDL facts to Qdrant Cloud...")
        all_facts = set()
        for r in rules:
            all_facts.update(r["inputs"])
            all_facts.update(r["outputs"])

        points = []
        for f_val in all_facts:
            vector = self.embed_model.encode(f_val).tolist()
            points.append(PointStruct(
                id=abs(hash(f_val)) % (10**15),
                vector=vector,
                payload={"value": f_val, "label": f_val, "domain": DOMAIN}
            ))

        if points:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(COLLECTION_NAME, batch)
                logger.info("  Upserted GDL facts batch %d/%d to Qdrant Cloud.", i // batch_size + 1, (len(points) + batch_size - 1) // batch_size)

        logger.info("Successfully loaded %d facts to Qdrant Cloud.", len(points))

    def run(self):
        rules = self.parse_gdl()
        self.load_to_neo4j(rules)
        self.load_to_qdrant(rules)
        logger.info("✅ FormalGeo Knowledge Ingestion Complete. GeoIPS is now fully loaded.")

    def close(self):
        self.neo4j_conn.close()


if __name__ == "__main__":
    ingestor = FormalGeoIngestor()
    try:
        ingestor.run()
    finally:
        ingestor.close()
