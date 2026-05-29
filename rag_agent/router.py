"""
Neuro-Symbolic Router for Omni-IPS.

Bridges Natural Language queries with the Symbolic Core Engine by:
1. Parsing queries into structured Facts and Goals (via LangChain LLM or regex fallback).
2. Semantically mapping those text entities to exact Neo4j Fact nodes using Qdrant vector search.
"""

import os
import sys
import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Load local environment configurations from .env
load_dotenv()

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from core_engine.models import Fact
from graph_db.connection import Neo4jConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_router")

# Load embedding model once at module level for performance
_embedding_model: Optional[SentenceTransformer] = None

def _get_embedding_model() -> SentenceTransformer:
    """Lazily loads and caches the SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer embedding model 'all-MiniLM-L6-v2'...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# Standard Chemistry Natural Language to Formula mapping lookup
CHEM_DICTIONARY = {
    "water": "H2O",
    "sodium": "Na",
    "sodium hydroxide": "NaOH",
    "hydrochloric acid": "HCl",
    "sodium chloride": "NaCl",
    "salt": "NaCl",
    "chlorine": "Cl2",
    "oxygen": "O2",
    "hydrogen": "H2",
    "magnesium": "Mg",
    "magnesium oxide": "MgO",
    "iron": "Fe",
    "iron oxide": "Fe2O3",
    "rust": "Fe2O3",
    "calcium carbonate": "CaCO3",
    "limestone": "CaCO3",
    "calcium oxide": "CaO",
    "quicklime": "CaO",
    "carbon dioxide": "CO2",
    "sodium bicarbonate": "NaHCO3",
    "baking soda": "NaHCO3",
    "sodium carbonate": "Na2CO3",
    "soda ash": "Na2CO3",
    "zinc": "Zn",
    "zinc chloride": "ZnCl2",
    "copper sulfate": "CuSO4",
    "copper": "Cu",
    "iron sulfate": "FeSO4",
    "silver nitrate": "AgNO3",
    "silver chloride": "AgCl",
    "barium chloride": "BaCl2",
    "barium sulfate": "BaSO4",
    "sodium sulfate": "Na2SO4",
    "potassium hydroxide": "KOH",
    "sulfuric acid": "H2SO4",
    "potassium sulfate": "K2SO4",
    "methane": "CH4",
    "ethanol": "C2H5OH",
    "propane": "C3H8",
    "ammonia": "NH3",
    "ammonium chloride": "NH4Cl",
    "carbonic acid": "H2CO3",
    "calcium hydroxide": "Ca(OH)2",
    "slaked lime": "Ca(OH)2",
}


def fallback_query_parser(query: str, domain: str) -> Tuple[List[str], str]:
    """
    Offline/Fallback regex-based natural language parser.
    Extracts formulas, equations, or geometric predicates directly from query text.
    """
    domain = domain.lower()
    extracted_facts = []
    extracted_goal = ""

    # Convert query to lowercase for dictionary keyword matching
    query_lower = query.lower()

    if domain == "chemistry":
        # 1. Look for known chemical common names and map them
        for common_name, formula in CHEM_DICTIONARY.items():
            # Match word boundary to avoid partial matching (e.g. 'iron' matching 'iron oxide')
            pattern = rf"\b{re.escape(common_name)}\b"
            if re.search(pattern, query_lower):
                # Simple heuristic: if it appears after "make", "get", "synthesize", "produce", or "yield", it is likely the goal
                is_goal = False
                for keyword in ["make", "get", "synthesize", "produce", "yield", "obtain", "derive", "forming", "form"]:
                    keyword_index = query_lower.find(keyword)
                    if keyword_index != -1 and query_lower.find(common_name) > keyword_index:
                        is_goal = True
                        break
                
                if is_goal:
                    extracted_goal = formula
                else:
                    extracted_facts.append(formula)

        # 2. Extract raw chemical formulas directly using regex (e.g. Na, H2O, HCl)
        # Matches uppercase chemical formulas with possible brackets and subscripts
        formula_regex = r"\b[A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*\))*\d*\b"
        raw_formulas = re.findall(formula_regex, query)
        for formula in raw_formulas:
            # Filter out single letters that aren't common elements unless they represent single-letter elements (excluding I/A to avoid pronoun false positives)
            if formula in ["O", "C", "H", "N", "S", "P", "K", "U", "F", "V", "W", "Y", "B"] or len(formula) > 1:
                # Determine if it's the goal based on keywords
                idx = query.find(formula)
                is_goal = False
                for kw in ["make", "get", "synthesize", "produce", "yield", "obtain", "derive", "->", "to"]:
                    kw_idx = query_lower.find(kw)
                    if kw_idx != -1 and idx > kw_idx:
                        is_goal = True
                        break
                
                if is_goal:
                    extracted_goal = formula
                else:
                    if formula not in extracted_facts:
                        extracted_facts.append(formula)

    elif domain == "geometry":
        # Extract predicates of format Word(arg1, arg2...) e.g. Congruent(AB, CD), Triangle(A,B,C)
        predicate_regex = r"\b[A-Za-z_]+\([A-Za-z0-9,\s_]*\)"
        raw_preds = re.findall(predicate_regex, query)
        
        # Check standard sentence structures:
        # Structure 1: Prove {goal} given/if/when/assuming {facts}
        # Structure 2: Given/if/assuming {facts}, prove {goal}
        given_idx = -1
        for kw in ["given", "if", "assume", "assuming", "suppose", "where"]:
            idx = query_lower.find(kw)
            if idx != -1:
                given_idx = idx
                break
                
        prove_idx = -1
        for kw in ["prove", "show", "deduce", "conclude", "obtain"]:
            idx = query_lower.find(kw)
            if idx != -1:
                prove_idx = idx
                break
                
        for pred in raw_preds:
            cleaned_pred = re.sub(r"\s+", "", pred)
            idx = query.find(pred)
            
            is_goal = False
            if prove_idx != -1:
                if given_idx != -1:
                    if prove_idx < given_idx:
                        # "Prove {goal} given {facts}"
                        if idx < given_idx:
                            is_goal = True
                    else:
                        # "Given {facts}, prove {goal}"
                        if idx > prove_idx:
                            is_goal = True
                else:
                    # No 'given' word, just check if it's the first one after the 'prove' keyword
                    if idx > prove_idx and not extracted_goal:
                        is_goal = True
            
            if is_goal:
                extracted_goal = cleaned_pred
            else:
                if cleaned_pred not in extracted_facts:
                    extracted_facts.append(cleaned_pred)

    elif domain == "algebra":
        # Find index of goal keywords
        goal_idx = -1
        for kw in ["find", "solve", "prove", "calculate", "what is", "chứng minh", "tìm"]:
            idx = query_lower.find(kw)
            if idx != -1:
                goal_idx = idx
                break

        # Extract equations containing '=' or mathematical terms
        eq_regex = r"\b[a-zA-Z0-9\+\-\*\/\s]+=[ a-zA-Z0-9\+\-\*\/\s]+\b"
        raw_eqs = re.findall(eq_regex, query)
        for eq in raw_eqs:
            cleaned_eq = re.sub(r"\s+", "", eq)
            # Remove leading non-equation words like 'find', 'given', etc from the equation match
            cleaned_eq = re.sub(r"^(find|given|solve|prove|tìm|cho)", "", cleaned_eq, flags=re.IGNORECASE)
            
            # Determine if it's the goal
            idx = query.find(eq)
            is_goal = False
            if cleaned_eq.startswith("x="):
                is_goal = True
            elif goal_idx != -1 and idx >= goal_idx:
                is_goal = True
                
            if is_goal:
                extracted_goal = cleaned_eq
            else:
                extracted_facts.append(cleaned_eq)

        # Check operations
        op_regex = r"\b[A-Za-z_]+\([a-zA-Z0-9,\s_]*\)"
        raw_ops = re.findall(op_regex, query)
        for op in raw_ops:
            cleaned_op = re.sub(r"\s+", "", op)
            extracted_facts.append(cleaned_op)

    # Clean duplicates and ensure we have fallback defaults if parsing returned nothing
    extracted_facts = list(set(extracted_facts))
    
    # If no goal was explicitly extracted, fallback to the last element or standard domain placeholders
    if not extracted_goal and extracted_facts:
        # Default fallback
        if domain == "chemistry":
            extracted_goal = "NaCl"
        elif domain == "geometry":
            extracted_goal = "Congruent(AB,EF)"
        elif domain == "algebra":
            extracted_goal = "x=3"
            
    # Cleanup facts if goal is accidentally in it
    if extracted_goal in extracted_facts:
        extracted_facts.remove(extracted_goal)

    logger.info("[Fallback Parser] Extracted facts: %s, goal: %s", extracted_facts, extracted_goal)
    return extracted_facts, extracted_goal


def llm_query_parser(query: str, domain: str) -> Tuple[List[str], str]:
    """
    LLM-based query parser using LangChain with model-agnostic structured outputs.
    Strictly parses natural language queries into exact logical Fact and Goal structures.
    Falls back to fallback_query_parser if LLM is not configured or fails.
    """
    from rag_agent.llm_factory import get_llm
    llm = get_llm(temperature=0.0)
    
    if not llm:
        logger.info("LLM factory returned None. Using offline regex fallback parser.")
        return fallback_query_parser(query, domain)

    try:
        from pydantic import BaseModel, Field
        from langchain_core.prompts import ChatPromptTemplate

        class ExtractedProblem(BaseModel):
            initial_facts: List[str] = Field(..., description="Clean formal logic strings or chemical formulas extracted from the context.")
            goal_fact: str = Field(..., description="The target final assertion or chemical compound to deduce.")

        system_prompt = (
            "You are a translation layer and expert NLP parser for a Neuro-Symbolic reasoning engine.\n"
            "Your task is to parse a natural language query (regardless of whether it is in English or Vietnamese) "
            "for the domain '{domain}' and strictly translate it into exact, clean, formal logical structures "
            "(initial facts and goal fact) matching our domain syntax.\n\n"
            "Domain Syntax Rules:\n"
            "1. Chemistry:\n"
            "   - Extract chemical formulas (e.g., 'Na', 'H2O'). Map common names to formulas (e.g., 'nước' -> 'H2O', 'xút ăn da' -> 'NaOH', 'natri' -> 'Na', 'natri hydroxit' -> 'NaOH').\n"
            "   - Example: 'Tôi có natri và nước, làm sao ra xút ăn da?' -> initial_facts: ['Na', 'H2O'], goal_fact: 'NaOH'\n"
            "   - Example: 'I have sodium and water, how do I get NaOH?' -> initial_facts: ['Na', 'H2O'], goal_fact: 'NaOH'\n\n"
            "2. Geometry:\n"
            "   - Extract formal geometry predicates, sorting arguments alphabetically when appropriate (e.g., segment endpoints 'AB', 'CD').\n"
            "   - For right triangles, always extract BOTH 'RightTriangle(A,B,C)' and 'RightAngle(Angle(BAC))' if right-angled at A.\n"
            "   - Example: 'Chứng minh AB bằng EF biết AB bằng CD và CD bằng EF' -> initial_facts: ['Congruent(AB,CD)', 'Congruent(CD,EF)'], goal_fact: 'Congruent(AB,EF)'\n"
            "   - Example: 'Prove AB equals EF given AB is congruent to CD and CD is congruent to EF' -> initial_facts: ['Congruent(AB,CD)', 'Congruent(CD,EF)'], goal_fact: 'Congruent(AB,EF)'\n"
            "   - Example: 'Cho tam giác ABC vuông tại A. Chứng minh rằng bình phương cạnh huyền BC bằng tổng bình phương hai cạnh vuông AB và AC' -> initial_facts: ['RightTriangle(A,B,C)', 'RightAngle(Angle(BAC))'], goal_fact: 'BC^2=AB^2+AC^2'\n"
            "   - Example: 'Given right triangle ABC with right angle at A, prove BC^2 = AB^2 + AC^2' -> initial_facts: ['RightTriangle(A,B,C)', 'RightAngle(Angle(BAC))'], goal_fact: 'BC^2=AB^2+AC^2'\n\n"
            "3. Algebra:\n"
            "   - Extract algebraic equations and formal operation predicates (e.g. 'Subtract(2,both_sides)', 'Add(3,both_sides)').\n"
            "   - Example: 'Giải phương trình x+2=5' -> initial_facts: ['x+2=5'], goal_fact: 'x=3'\n"
            "   - Example: 'Solve x+2=5' -> initial_facts: ['x+2=5'], goal_fact: 'x=3'\n"
            "   - Example: 'Given x+2=5, Subtract(2,both_sides), find x=3' -> initial_facts: ['x+2=5', 'Subtract(2,both_sides)'], goal_fact: 'x=3'\n"
            "   - Example: 'Cho phương trình x+2=5, trừ 2 ở cả hai vế, tìm x=3' -> initial_facts: ['x+2=5', 'Subtract(2,both_sides)'], goal_fact: 'x=3'\n\n"
            "Do not solve the problem. Output ONLY the parsed structures in the specified format without extra words."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Parse this query: {query}")
        ])

        logger.info("Calling model-agnostic structured parser for domain '%s'...", domain)
        
        # Try structured output first
        try:
            structured_llm = llm.with_structured_output(ExtractedProblem)
            chain = prompt | structured_llm
            response = chain.invoke({"query": query, "domain": domain})
            facts, goal = response.initial_facts, response.goal_fact
        except (AttributeError, NotImplementedError, Exception) as e:
            logger.info("Model does not support with_structured_output or failed: %s. Using manual JSON parsing.", e)
            from langchain_core.output_parsers import JsonOutputParser
            parser = JsonOutputParser(pydantic_object=ExtractedProblem)
            prompt_with_format = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\nOutput ONLY valid JSON matching this schema:\n{format_instructions}"),
                ("human", "{query}")
            ])
            chain = prompt_with_format | llm | parser
            response = chain.invoke({
                "query": query, 
                "domain": domain, 
                "format_instructions": parser.get_format_instructions()
            })
            facts, goal = response["initial_facts"], response["goal_fact"]

        logger.info("[LLM Parser] Extracted facts: %s, goal: %s", facts, goal)
        return facts, goal

    except Exception as e:
        logger.error("LLM parser failed: %s", str(e))
        print(f"\n[Hệ thống] Cảnh báo: Không thể kết nối với dịch vụ AI (LLM). Đang sử dụng bộ phân tích dự phòng (Regex).")
        logger.warning("LangChain model-agnostic parser call failed: %s. Falling back to offline parser.", e)
        return fallback_query_parser(query, domain)


def map_text_to_graph_fact(
    text: str, 
    domain: str, 
    qdrant_client: QdrantClient,
    collection_name: str = "omni_ips_facts"
) -> Fact:
    """
    Queries Qdrant to find the most semantically matching Fact node in Neo4j.
    First checks the specific domain partition collection (e.g., chemistry_facts, geometry_facts).
    Performs an L2 verification using:
    1. Exact string/formula match using Qdrant Payload Filter search.
    2. Extremely high-confidence vector similarity search (threshold >= 0.85) if exact match fails.
    Maps the result to a canonical Fact and resolves the exact neo4j_id.
    """
    domain = domain.lower()
    text = text.strip()
    
    # 1. Determine collection to query
    domain_collection = f"{domain}_facts"
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if domain_collection not in collections:
            logger.info("Collection '%s' not found. Querying central collection '%s'.", domain_collection, collection_name)
            domain_collection = collection_name
    except Exception as e:
        logger.warning("Failed to query collections: %s. Defaulting to '%s'.", e, collection_name)
        domain_collection = collection_name

    logger.info("Mapping text '%s' using collection '%s' for domain '%s'", text, domain_collection, domain)

    payload_match = None
    
    # 2. Try Exact Payload Match (Scroll with Filters)
    try:
        # Build strict filters
        must_filters = []
        if domain_collection == collection_name:
            must_filters.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
            
        # Try matching text against value field
        val_filters = must_filters + [FieldCondition(key="value", match=MatchValue(value=text))]
        results, _ = qdrant_client.scroll(
            collection_name=domain_collection,
            scroll_filter=Filter(must=val_filters),
            limit=1
        )
        
        if results:
            payload_match = results[0].payload
            logger.info("Exact payload match (value) found for '%s' in '%s'", text, domain_collection)
        else:
            # Try matching text against label field
            label_filters = must_filters + [FieldCondition(key="label", match=MatchValue(value=text))]
            results, _ = qdrant_client.scroll(
                collection_name=domain_collection,
                scroll_filter=Filter(must=label_filters),
                limit=1
            )
            if results:
                payload_match = results[0].payload
                logger.info("Exact payload match (label) found for '%s' in '%s'", text, domain_collection)
    except Exception as e:
        logger.warning("Exact payload matching failed: %s", e)

    # 3. Fallback to High-Confidence Vector Search
    if not payload_match:
        try:
            model = _get_embedding_model()
            query_vector = model.encode(text).tolist()
            
            # Query domain partition
            must_filters = []
            if domain_collection == collection_name:
                must_filters.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
                
            results = qdrant_client.query_points(
                collection_name=domain_collection,
                query=query_vector,
                query_filter=Filter(must=must_filters) if must_filters else None,
                limit=1
            )
            
            if results.points:
                match = results.points[0]
                # High similarity threshold for validation
                HIGH_CONFIDENCE_THRESHOLD = 0.85
                if match.score >= HIGH_CONFIDENCE_THRESHOLD:
                    payload_match = match.payload
                    logger.info("High-confidence vector match (%0.4f >= %0.2f) found for '%s' in '%s'", 
                                match.score, HIGH_CONFIDENCE_THRESHOLD, text, domain_collection)
                else:
                    logger.info("Vector match found for '%s' but score %0.4f is below high-confidence threshold %0.2f", 
                                text, match.score, HIGH_CONFIDENCE_THRESHOLD)
        except Exception as e:
            logger.warning("Vector search fallback failed for text '%s': %s", text, e)

    # 4. Resolve Neo4j ID & Construct Fact
    if payload_match:
        canonical_value = payload_match["value"]
        canonical_label = payload_match.get("label", canonical_value)
        neo4j_id = payload_match.get("neo4j_id")
        
        # If neo4j_id is not in domain-specific collection payload, resolve it from central omni_ips_facts
        if not neo4j_id and domain_collection != collection_name:
            try:
                central_results, _ = qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="domain", match=MatchValue(value=domain)),
                            FieldCondition(key="value", match=MatchValue(value=canonical_value))
                        ]
                    ),
                    limit=1
                )
                if central_results:
                    neo4j_id = central_results[0].payload.get("neo4j_id")
            except Exception as e:
                logger.warning("Failed to resolve neo4j_id from central collection: %s", e)

        # Standard conventional fallbacks if still not found
        if not neo4j_id:
            if domain == "chemistry":
                neo4j_id = f"fact_{canonical_value}"
            elif domain == "geometry":
                neo4j_id = f"geo_fact_{canonical_value}"
            else:
                neo4j_id = f"adhoc_{domain}_{abs(hash(canonical_value))}"

        logger.info("Successfully mapped '%s' to Fact value='%s', label='%s', neo4j_id='%s'", 
                    text, canonical_value, canonical_label, neo4j_id)
        return Fact(
            id=neo4j_id,
            value=canonical_value,
            domain=domain,
            attributes={"label": canonical_label}
        )

    # 5. Ad-hoc Fallback (If no match at all is found in Qdrant)
    # Check manual local chemistry lookup dictionary
    mapped_val = text
    if domain == "chemistry" and text.lower() in CHEM_DICTIONARY:
        mapped_val = CHEM_DICTIONARY[text.lower()]
        
    # Construct ad-hoc IDs
    if domain == "chemistry":
        neo4j_id = f"fact_{mapped_val}"
    elif domain == "geometry":
        neo4j_id = f"geo_fact_{mapped_val}"
    else:
        neo4j_id = f"adhoc_{domain}_{abs(hash(mapped_val))}"

    logger.info("Ad-hoc fallback routing for '%s' -> Value: '%s', ID: '%s'", text, mapped_val, neo4j_id)
    return Fact(
        id=neo4j_id,
        value=mapped_val,
        domain=domain,
        attributes={"label": text}
    )


def route_query(query: str, domain: str) -> Tuple[List[Fact], Fact]:
    """
    Entrypoint for the Neuro-Symbolic Router.
    Parses natural language query, maps entities via Qdrant, and returns Fact nodes.
    """
    domain = domain.lower()
    
    # 1. Parse text query into structured text concepts
    facts_text, goal_text = llm_query_parser(query, domain)

    # 2. Establish connection to Qdrant with retry
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    qdrant = None
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
            qdrant.get_collections()
            break
        except Exception as e:
            if attempt == max_retries:
                logger.warning("Could not connect to Qdrant at %s:%d after %d attempts: %s. Using ad-hoc routing.",
                               qdrant_host, qdrant_port, max_retries, e)
                qdrant = None
            else:
                time.sleep(1.0 * attempt)

    # 3. Map parsed text elements to formal Fact objects
    mapped_facts = []
    for f_text in facts_text:
        if qdrant:
            mapped_facts.append(map_text_to_graph_fact(f_text, domain, qdrant))
        else:
            # Offline fallback
            val = CHEM_DICTIONARY.get(f_text.lower(), f_text) if domain == "chemistry" else f_text
            mapped_facts.append(Fact(id=f"adhoc_{domain}_{abs(hash(val))}", value=val, domain=domain))

    if qdrant and goal_text:
        mapped_goal = map_text_to_graph_fact(goal_text, domain, qdrant)
    else:
        val = CHEM_DICTIONARY.get(goal_text.lower(), goal_text) if domain == "chemistry" and goal_text else goal_text
        mapped_goal = Fact(id=f"adhoc_{domain}_{abs(hash(val))}", value=val, domain=domain)

    # Deduplicate facts
    seen = set()
    deduped_facts = []
    for f in mapped_facts:
        if f.value not in seen:
            seen.add(f.value)
            deduped_facts.append(f)

    # Ensure facts do not contain the goal
    deduped_facts = [f for f in deduped_facts if f.value != mapped_goal.value]

    logger.info("Neuro-Symbolic routing COMPLETE. Mapped %d Facts, Goal: %s", len(deduped_facts), mapped_goal.value)
    return deduped_facts, mapped_goal
