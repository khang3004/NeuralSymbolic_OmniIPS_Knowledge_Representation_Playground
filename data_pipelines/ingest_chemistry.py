
import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_db.connection import Neo4jConnection
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_chemistry")

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
DOMAIN = "chemistry"
BATCH_SIZE = 500
COLLECTION_NAME = "chemistry_facts"

# Massive SPARQL query for chemical reactions
# Q187939: chemical reaction
# P828: has reactant
# P1542: has product
SPARQL_QUERY_REACTIONS = """
SELECT DISTINCT ?reaction ?reactionLabel ?reactant ?reactantLabel ?reactantFormula ?product ?productLabel ?productFormula WHERE {
  ?reaction wdt:P31 wd:Q187939 .
  
  ?reaction p:P828 ?reactant_statement .
  ?reactant_statement ps:P828 ?reactant .
  ?reactant wdt:P274 ?reactantFormula .
  
  ?reaction p:P1542 ?product_statement .
  ?product_statement ps:P1542 ?product .
  ?product wdt:P274 ?productFormula .
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,vi" . }
}
LIMIT 2000
"""

# =============================================================================
# CURATED HIGH-SCHOOL CHEMISTRY REACTIONS
# Essential textbook reactions that may not be reliably covered by Wikidata.
# =============================================================================

CURATED_REACTIONS = [
    # =========================================================================
    # A. SYNTHESIS / COMBINATION REACTIONS (8 reactions)
    # =========================================================================
    {
        "id": "rxn_synth_water",
        "name": "Synthesis of Water",
        "inputs": ["H2", "O2"],
        "outputs": ["H2O"],
        "description": "2H₂ + O₂ → 2H₂O. Combustion of hydrogen to produce water."
    },
    {
        "id": "rxn_synth_nacl",
        "name": "Synthesis of Sodium Chloride",
        "inputs": ["Na", "Cl2"],
        "outputs": ["NaCl"],
        "description": "2Na + Cl₂ → 2NaCl. Sodium reacts with chlorine gas to form table salt."
    },
    {
        "id": "rxn_synth_mgo",
        "name": "Synthesis of Magnesium Oxide",
        "inputs": ["Mg", "O2"],
        "outputs": ["MgO"],
        "description": "2Mg + O₂ → 2MgO. Magnesium burns in oxygen with a brilliant white flame."
    },
    {
        "id": "rxn_synth_fe2o3",
        "name": "Synthesis of Iron(III) Oxide (Rusting)",
        "inputs": ["Fe", "O2"],
        "outputs": ["Fe2O3"],
        "description": "4Fe + 3O₂ → 2Fe₂O₃. Iron reacts with oxygen to form rust."
    },
    {
        "id": "rxn_synth_nh3",
        "name": "Haber Process (Ammonia Synthesis)",
        "inputs": ["N2", "H2"],
        "outputs": ["NH3"],
        "description": "N₂ + 3H₂ → 2NH₃. Industrial synthesis of ammonia via the Haber-Bosch process."
    },
    {
        "id": "rxn_synth_so3",
        "name": "Synthesis of Sulfur Trioxide",
        "inputs": ["SO2", "O2"],
        "outputs": ["SO3"],
        "description": "2SO₂ + O₂ → 2SO₃. Oxidation of sulfur dioxide (Contact process step)."
    },
    {
        "id": "rxn_synth_al2o3",
        "name": "Synthesis of Aluminum Oxide",
        "inputs": ["Al", "O2"],
        "outputs": ["Al2O3"],
        "description": "4Al + 3O₂ → 2Al₂O₃. Aluminum burns in oxygen to form alumina."
    },
    {
        "id": "rxn_synth_hcl",
        "name": "Synthesis of Hydrogen Chloride",
        "inputs": ["H2", "Cl2"],
        "outputs": ["HCl"],
        "description": "H₂ + Cl₂ → 2HCl. Hydrogen and chlorine combine to form hydrochloric acid gas."
    },

    # =========================================================================
    # B. DECOMPOSITION REACTIONS (6 reactions)
    # =========================================================================
    {
        "id": "rxn_decomp_caco3",
        "name": "Decomposition of Calcium Carbonate",
        "inputs": ["CaCO3"],
        "outputs": ["CaO", "CO2"],
        "description": "CaCO₃ → CaO + CO₂. Thermal decomposition of limestone to quicklime."
    },
    {
        "id": "rxn_decomp_h2o",
        "name": "Electrolysis of Water",
        "inputs": ["H2O"],
        "outputs": ["H2", "O2"],
        "description": "2H₂O → 2H₂ + O₂. Water decomposition via electrolysis."
    },
    {
        "id": "rxn_decomp_h2o2",
        "name": "Decomposition of Hydrogen Peroxide",
        "inputs": ["H2O2"],
        "outputs": ["H2O", "O2"],
        "description": "2H₂O₂ → 2H₂O + O₂. Hydrogen peroxide decomposes into water and oxygen."
    },
    {
        "id": "rxn_decomp_kclo3",
        "name": "Decomposition of Potassium Chlorate",
        "inputs": ["KClO3"],
        "outputs": ["KCl", "O2"],
        "description": "2KClO₃ → 2KCl + 3O₂. Thermal decomposition; classic source of oxygen gas."
    },
    {
        "id": "rxn_decomp_nahco3",
        "name": "Decomposition of Sodium Bicarbonate",
        "inputs": ["NaHCO3"],
        "outputs": ["Na2CO3", "H2O", "CO2"],
        "description": "2NaHCO₃ → Na₂CO₃ + H₂O + CO₂. Baking soda decomposes when heated."
    },
    {
        "id": "rxn_decomp_kmno4",
        "name": "Decomposition of Potassium Permanganate",
        "inputs": ["KMnO4"],
        "outputs": ["K2MnO4", "MnO2", "O2"],
        "description": "2KMnO₄ → K₂MnO₄ + MnO₂ + O₂. Lab preparation of oxygen gas."
    },

    # =========================================================================
    # C. SINGLE REPLACEMENT REACTIONS (7 reactions)
    # =========================================================================
    {
        "id": "rxn_single_na_h2o",
        "name": "Sodium Reacting with Water",
        "inputs": ["Na", "H2O"],
        "outputs": ["NaOH", "H2"],
        "description": "2Na + 2H₂O → 2NaOH + H₂. Vigorous alkali metal reaction with water."
    },
    {
        "id": "rxn_single_zn_hcl",
        "name": "Zinc Reacting with Hydrochloric Acid",
        "inputs": ["Zn", "HCl"],
        "outputs": ["ZnCl2", "H2"],
        "description": "Zn + 2HCl → ZnCl₂ + H₂. Classic lab preparation of hydrogen gas."
    },
    {
        "id": "rxn_single_fe_cuso4",
        "name": "Iron Displacing Copper",
        "inputs": ["Fe", "CuSO4"],
        "outputs": ["FeSO4", "Cu"],
        "description": "Fe + CuSO₄ → FeSO₄ + Cu. Iron displaces copper from copper sulfate."
    },
    {
        "id": "rxn_single_mg_hcl",
        "name": "Magnesium Reacting with HCl",
        "inputs": ["Mg", "HCl"],
        "outputs": ["MgCl2", "H2"],
        "description": "Mg + 2HCl → MgCl₂ + H₂. Magnesium dissolves in hydrochloric acid."
    },
    {
        "id": "rxn_single_al_hcl",
        "name": "Aluminum Reacting with HCl",
        "inputs": ["Al", "HCl"],
        "outputs": ["AlCl3", "H2"],
        "description": "2Al + 6HCl → 2AlCl₃ + 3H₂. Aluminum dissolves in hydrochloric acid."
    },
    {
        "id": "rxn_single_ca_h2o",
        "name": "Calcium Reacting with Water",
        "inputs": ["Ca", "H2O"],
        "outputs": ["Ca(OH)2", "H2"],
        "description": "Ca + 2H₂O → Ca(OH)₂ + H₂. Calcium reacts with water to form slaked lime."
    },
    {
        "id": "rxn_single_k_h2o",
        "name": "Potassium Reacting with Water",
        "inputs": ["K", "H2O"],
        "outputs": ["KOH", "H2"],
        "description": "2K + 2H₂O → 2KOH + H₂. Highly reactive alkali metal in water."
    },

    # =========================================================================
    # D. DOUBLE REPLACEMENT / NEUTRALIZATION (10 reactions)
    # =========================================================================
    {
        "id": "rxn_double_naoh_hcl",
        "name": "Neutralization: NaOH + HCl",
        "inputs": ["NaOH", "HCl"],
        "outputs": ["NaCl", "H2O"],
        "description": "NaOH + HCl → NaCl + H₂O. Classic acid-base neutralization."
    },
    {
        "id": "rxn_double_naoh_h2so4",
        "name": "Neutralization: NaOH + H₂SO₄",
        "inputs": ["NaOH", "H2SO4"],
        "outputs": ["Na2SO4", "H2O"],
        "description": "2NaOH + H₂SO₄ → Na₂SO₄ + 2H₂O. Strong base neutralizes strong acid."
    },
    {
        "id": "rxn_double_koh_hcl",
        "name": "Neutralization: KOH + HCl",
        "inputs": ["KOH", "HCl"],
        "outputs": ["KCl", "H2O"],
        "description": "KOH + HCl → KCl + H₂O. Potassium hydroxide neutralizes hydrochloric acid."
    },
    {
        "id": "rxn_double_ca_oh2_hcl",
        "name": "Neutralization: Ca(OH)₂ + HCl",
        "inputs": ["Ca(OH)2", "HCl"],
        "outputs": ["CaCl2", "H2O"],
        "description": "Ca(OH)₂ + 2HCl → CaCl₂ + 2H₂O. Slaked lime neutralizes hydrochloric acid."
    },
    {
        "id": "rxn_double_agno3_nacl",
        "name": "Precipitation: AgNO₃ + NaCl",
        "inputs": ["AgNO3", "NaCl"],
        "outputs": ["AgCl", "NaNO3"],
        "description": "AgNO₃ + NaCl → AgCl↓ + NaNO₃. Classic precipitation of silver chloride."
    },
    {
        "id": "rxn_double_bacl2_na2so4",
        "name": "Precipitation: BaCl₂ + Na₂SO₄",
        "inputs": ["BaCl2", "Na2SO4"],
        "outputs": ["BaSO4", "NaCl"],
        "description": "BaCl₂ + Na₂SO₄ → BaSO₄↓ + 2NaCl. Barium sulfate precipitate formation."
    },
    {
        "id": "rxn_double_na2co3_hcl",
        "name": "Carbonate Reaction: Na₂CO₃ + HCl",
        "inputs": ["Na2CO3", "HCl"],
        "outputs": ["NaCl", "H2O", "CO2"],
        "description": "Na₂CO₃ + 2HCl → 2NaCl + H₂O + CO₂↑. Carbonate reacts with acid producing gas."
    },
    {
        "id": "rxn_double_naoh_co2",
        "name": "CO₂ Absorption by NaOH",
        "inputs": ["NaOH", "CO2"],
        "outputs": ["Na2CO3", "H2O"],
        "description": "2NaOH + CO₂ → Na₂CO₃ + H₂O. Carbon dioxide absorbed by sodium hydroxide."
    },
    {
        "id": "rxn_double_ca_oh2_co2",
        "name": "CO₂ Test: Limewater",
        "inputs": ["Ca(OH)2", "CO2"],
        "outputs": ["CaCO3", "H2O"],
        "description": "Ca(OH)₂ + CO₂ → CaCO₃↓ + H₂O. The classic limewater test for CO₂ — milky white precipitate."
    },
    {
        "id": "rxn_double_na_hcl",
        "name": "Sodium Acid Reaction",
        "inputs": ["Na", "HCl"],
        "outputs": ["NaCl", "H2"],
        "description": "2Na + 2HCl → 2NaCl + H₂. Direct reaction of sodium metal with hydrochloric acid."
    },

    # =========================================================================
    # E. COMBUSTION REACTIONS (6 reactions)
    # =========================================================================
    {
        "id": "rxn_combust_ch4",
        "name": "Combustion of Methane",
        "inputs": ["CH4", "O2"],
        "outputs": ["CO2", "H2O"],
        "description": "CH₄ + 2O₂ → CO₂ + 2H₂O. Complete combustion of methane (natural gas)."
    },
    {
        "id": "rxn_combust_c2h5oh",
        "name": "Combustion of Ethanol",
        "inputs": ["C2H5OH", "O2"],
        "outputs": ["CO2", "H2O"],
        "description": "C₂H₅OH + 3O₂ → 2CO₂ + 3H₂O. Complete combustion of ethanol."
    },
    {
        "id": "rxn_combust_c3h8",
        "name": "Combustion of Propane",
        "inputs": ["C3H8", "O2"],
        "outputs": ["CO2", "H2O"],
        "description": "C₃H₈ + 5O₂ → 3CO₂ + 4H₂O. Complete combustion of propane."
    },
    {
        "id": "rxn_combust_c",
        "name": "Combustion of Carbon",
        "inputs": ["C", "O2"],
        "outputs": ["CO2"],
        "description": "C + O₂ → CO₂. Complete combustion of carbon."
    },
    {
        "id": "rxn_combust_s",
        "name": "Combustion of Sulfur",
        "inputs": ["S", "O2"],
        "outputs": ["SO2"],
        "description": "S + O₂ → SO₂. Combustion of sulfur producing sulfur dioxide."
    },
    {
        "id": "rxn_combust_c_incomplete",
        "name": "Incomplete Combustion of Carbon",
        "inputs": ["C", "O2"],
        "outputs": ["CO"],
        "description": "2C + O₂ → 2CO. Incomplete combustion of carbon produces toxic carbon monoxide."
    },

    # =========================================================================
    # F. OXIDE + WATER REACTIONS (6 reactions)
    # =========================================================================
    {
        "id": "rxn_cao_h2o",
        "name": "Quicklime + Water",
        "inputs": ["CaO", "H2O"],
        "outputs": ["Ca(OH)2"],
        "description": "CaO + H₂O → Ca(OH)₂. Quicklime reacts exothermically with water to form slaked lime."
    },
    {
        "id": "rxn_na2o_h2o",
        "name": "Sodium Oxide + Water",
        "inputs": ["Na2O", "H2O"],
        "outputs": ["NaOH"],
        "description": "Na₂O + H₂O → 2NaOH. Basic oxide dissolves in water to form sodium hydroxide."
    },
    {
        "id": "rxn_so3_h2o",
        "name": "Sulfur Trioxide + Water",
        "inputs": ["SO3", "H2O"],
        "outputs": ["H2SO4"],
        "description": "SO₃ + H₂O → H₂SO₄. Acidic oxide reacts with water to form sulfuric acid."
    },
    {
        "id": "rxn_co2_h2o",
        "name": "Carbon Dioxide + Water",
        "inputs": ["CO2", "H2O"],
        "outputs": ["H2CO3"],
        "description": "CO₂ + H₂O → H₂CO₃. CO₂ dissolves in water to form weak carbonic acid."
    },
    {
        "id": "rxn_so2_h2o",
        "name": "Sulfur Dioxide + Water",
        "inputs": ["SO2", "H2O"],
        "outputs": ["H2SO3"],
        "description": "SO₂ + H₂O → H₂SO₃. SO₂ dissolves in water to form sulfurous acid (acid rain)."
    },
    {
        "id": "rxn_p2o5_h2o",
        "name": "Phosphorus Pentoxide + Water",
        "inputs": ["P2O5", "H2O"],
        "outputs": ["H3PO4"],
        "description": "P₂O₅ + 3H₂O → 2H₃PO₄. Phosphorus pentoxide reacts with water to form phosphoric acid."
    },
]


class ChemistryIngest:
    def __init__(self):
        self.neo4j_conn = Neo4jConnection()
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.stats = {"facts": 0, "rules": 0}

    def fetch_wikidata_reactions(self) -> List[Dict[str, Any]]:
        logger.info("Fetching chemical reactions from Wikidata...")
        sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
        sparql.setQuery(SPARQL_QUERY_REACTIONS)
        sparql.setReturnFormat(JSON)
        sparql.addCustomHttpHeader("User-Agent", "Omni-IPS/2.0")
        
        try:
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            logger.info(f"Retrieved {len(bindings)} reaction records from Wikidata.")
            return bindings
        except Exception as e:
            logger.warning(f"SPARQL query failed (Wikidata may be unreachable): {e}")
            logger.info("Continuing with curated reactions only.")
            return []

    def process_wikidata(self, bindings: List[Dict[str, Any]]):
        reactions = {}
        facts = {}
        
        for b in bindings:
            rid = b["reaction"]["value"].split("/")[-1]
            r_name = b["reactionLabel"]["value"]
            
            if rid not in reactions:
                reactions[rid] = {
                    "id": rid,
                    "name": r_name,
                    "inputs": set(),
                    "outputs": set(),
                    "domain": DOMAIN
                }
            
            # Process Reactants
            react_formula = b["reactantFormula"]["value"]
            react_label = b["reactantLabel"]["value"]
            reactions[rid]["inputs"].add(react_formula)
            facts[react_formula] = {"value": react_formula, "label": react_label, "domain": DOMAIN}
            
            # Process Products
            prod_formula = b["productFormula"]["value"]
            prod_label = b["productLabel"]["value"]
            reactions[rid]["outputs"].add(prod_formula)
            facts[prod_formula] = {"value": prod_formula, "label": prod_label, "domain": DOMAIN}

        # Convert sets to lists for JSON/Neo4j
        for r in reactions.values():
            r["inputs"] = list(r["inputs"])
            r["outputs"] = list(r["outputs"])
            
        return list(facts.values()), list(reactions.values())

    def process_curated(self):
        """Process the curated high-school reactions into facts and rules."""
        facts = {}
        rules = []
        
        for rxn in CURATED_REACTIONS:
            for formula in rxn["inputs"] + rxn["outputs"]:
                if formula not in facts:
                    facts[formula] = {
                        "value": formula,
                        "label": formula,
                        "domain": DOMAIN
                    }
            rules.append({
                "id": rxn["id"],
                "name": rxn["name"],
                "inputs": rxn["inputs"],
                "outputs": rxn["outputs"],
                "domain": DOMAIN,
                "description": rxn["description"]
            })
        
        return list(facts.values()), rules

    def load_to_neo4j(self, facts: List[Dict], rules: List[Dict]):
        logger.info(f"Loading {len(facts)} facts and {len(rules)} rules to Neo4j...")
        with self.neo4j_conn.get_session() as session:
            # Batch Load Facts with domain labels
            session.run("""
                UNWIND $batch AS row
                MERGE (f:Fact {value: row.value, domain: row.domain})
                ON CREATE SET f.id = 'chem_fact_' + row.value, f.label = row.label
                SET f:Chemistry
            """, batch=facts)
            
            # Batch Load Rules with domain labels
            session.run("""
                UNWIND $batch AS row
                MERGE (r:Rule {id: row.id, domain: row.domain})
                ON CREATE SET r.name = row.name, r.description = row.description,
                              r.inputs = row.inputs, r.outputs = row.outputs
                ON MATCH SET r.name = row.name, r.description = row.description,
                             r.inputs = row.inputs, r.outputs = row.outputs
                SET r:Chemistry
            """, batch=rules)
            
            # Create Relationships
            session.run("""
                UNWIND $batch AS row
                MATCH (r:Rule {id: row.id, domain: row.domain})
                WITH r, row
                UNWIND row.inputs AS input_val
                MATCH (f_in:Fact {value: input_val, domain: row.domain})
                MERGE (f_in)-[:HAS_INPUT]->(r)
                WITH r, row
                UNWIND row.outputs AS output_val
                MATCH (f_out:Fact {value: output_val, domain: row.domain})
                MERGE (r)-[:HAS_OUTPUT]->(f_out)
            """, batch=rules)
            
        self.stats["facts"] += len(facts)
        self.stats["rules"] += len(rules)

    def load_to_qdrant(self, facts: List[Dict]):
        logger.info(f"Loading {len(facts)} facts to Qdrant...")
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
        for i, f in enumerate(facts):
            vector = self.embed_model.encode(f"{f['label']} ({f['value']})").tolist()
            points.append(PointStruct(
                id=abs(hash(f['value'])) % (10**15),
                vector=vector,
                payload={"value": f['value'], "label": f['label'], "domain": DOMAIN}
            ))
            
            if len(points) >= BATCH_SIZE:
                self.qdrant_client.upsert(COLLECTION_NAME, points)
                points = []
        
        if points:
            self.qdrant_client.upsert(COLLECTION_NAME, points)

    def run(self):
        start_time = time.time()
        
        # 1. Process curated high-school reactions (always available)
        logger.info("=" * 60)
        logger.info("PHASE 1: Loading curated high-school chemistry reactions...")
        logger.info("=" * 60)
        curated_facts, curated_rules = self.process_curated()
        self.load_to_neo4j(curated_facts, curated_rules)
        self.load_to_qdrant(curated_facts)
        logger.info("Curated reactions loaded: %d facts, %d rules", len(curated_facts), len(curated_rules))
        
        # 2. Fetch and process Wikidata reactions (network-dependent)
        logger.info("=" * 60)
        logger.info("PHASE 2: Fetching reactions from Wikidata SPARQL...")
        logger.info("=" * 60)
        bindings = self.fetch_wikidata_reactions()
        if bindings:
            wiki_facts, wiki_rules = self.process_wikidata(bindings)
            self.load_to_neo4j(wiki_facts, wiki_rules)
            self.load_to_qdrant(wiki_facts)
            logger.info("Wikidata reactions loaded: %d facts, %d rules", len(wiki_facts), len(wiki_rules))
        else:
            logger.info("No Wikidata reactions fetched. Curated reactions are sufficient.")
        
        duration = time.time() - start_time
        logger.info("✅ Chemistry ETL Complete in %.2fs. Total Stats: %s", duration, self.stats)

if __name__ == "__main__":
    ChemistryIngest().run()
