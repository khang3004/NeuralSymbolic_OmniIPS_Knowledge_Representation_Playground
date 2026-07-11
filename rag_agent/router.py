"""
GeoIPS — Neuro-Symbolic Router.

Bridges Natural Language geometry queries with the Symbolic Core Engine by:
1. Parsing queries into structured geometric Facts and Goals (via LangChain LLM or regex fallback).
2. Semantically mapping those text entities to exact Neo4j Fact nodes using Qdrant vector search.
"""

import os
import re
import time
import logging
from typing import List, Tuple, Optional

from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from core_engine.models import Fact
from graph_db.qdrant_factory import get_qdrant_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("geo_router")

# Lazy-loaded embedding model
_embedding_model: Optional[SentenceTransformer] = None

DOMAIN = "geometry"
COLLECTION_NAME = "geometry_facts"


def _get_embedding_model() -> SentenceTransformer:
    """Lazily loads and caches the SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer embedding model 'all-MiniLM-L6-v2'...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# Fallback regex-based parser (geometry only)
# ---------------------------------------------------------------------------

def fallback_query_parser(query: str) -> Tuple[List[str], str]:
    """
    Offline regex-based geometry query parser.
    Extracts predicates of the form Word(arg1, arg2, ...) from the query.
    Uses keyword position heuristics to split facts vs goal.
    """
    predicate_regex = r"\b[A-Za-z_]+\([A-Za-z0-9,\s_()\^\.]*\)"
    raw_preds = re.findall(predicate_regex, query)

    query_lower = query.lower()

    # Detect keyword positions
    given_idx = -1
    for kw in ["given", "if", "assume", "assuming", "suppose", "where", "cho", "biết"]:
        idx = query_lower.find(kw)
        if idx != -1:
            given_idx = idx
            break

    prove_idx = -1
    for kw in ["prove", "show", "deduce", "conclude", "obtain", "chứng minh", "tìm"]:
        idx = query_lower.find(kw)
        if idx != -1:
            prove_idx = idx
            break

    extracted_facts: List[str] = []
    extracted_goal: str = ""

    for pred in raw_preds:
        cleaned = re.sub(r"\s+", "", pred)
        idx = query.find(pred)

        is_goal = False
        if prove_idx != -1:
            if given_idx != -1:
                if prove_idx < given_idx:
                    if idx < given_idx:
                        is_goal = True
                else:
                    if idx > prove_idx:
                        is_goal = True
            else:
                if idx > prove_idx and not extracted_goal:
                    is_goal = True

        if is_goal:
            extracted_goal = cleaned
        elif cleaned not in extracted_facts:
            extracted_facts.append(cleaned)

    # Also catch things like "BC^2=AB^2+AC^2" (Pythagorean equality)
    eq_regex = r"\b[A-Za-z0-9\^\+\-]+=[A-Za-z0-9\^\+\-\*]+\b"
    for eq in re.findall(eq_regex, query):
        if "=" in eq and eq not in extracted_facts and eq != extracted_goal:
            if prove_idx != -1 and query.find(eq) > prove_idx and not extracted_goal:
                extracted_goal = eq
            elif eq not in extracted_facts:
                extracted_facts.append(eq)

    if not extracted_goal and extracted_facts:
        extracted_goal = "Congruent(AB,EF)"

    if extracted_goal in extracted_facts:
        extracted_facts.remove(extracted_goal)

    logger.info(
        "[Fallback Parser] Extracted facts: %s, goal: %s",
        extracted_facts,
        extracted_goal,
    )
    return extracted_facts, extracted_goal


# ---------------------------------------------------------------------------
# LLM-based structured parser
# ---------------------------------------------------------------------------

def llm_query_parser(query: str) -> Tuple[List[str], str]:
    """
    LLM-based geometry query parser.
    Uses LangChain with structured output to extract formal predicates.
    Falls back to regex parser if LLM is unavailable.
    """
    from rag_agent.llm_factory import get_llm
    from typing import List as TypingList

    llm = get_llm(temperature=0.0)

    if not llm:
        logger.info("LLM unavailable. Using offline regex fallback parser.")
        return fallback_query_parser(query)

    try:
        from pydantic import BaseModel, Field
        from langchain_core.prompts import ChatPromptTemplate

        class ExtractedGeoProblem(BaseModel):
            initial_facts: TypingList[str] = Field(
                ...,
                description=(
                    "List of formal geometry predicates representing known conditions. "
                    "E.g. ['Triangle(A,B,C)', 'Congruent(AB,AC)', 'RightAngle(Angle(BAC))']"
                ),
            )
            goal_fact: str = Field(
                ...,
                description=(
                    "The single target fact to prove or deduce. "
                    "E.g. 'Equal(Angle(ABC),Angle(ACB))' or 'BC^2=AB^2+AC^2'"
                ),
            )

        system_prompt = (
            "You are a translation layer for a plane geometry Neuro-Symbolic reasoning engine (GeoIPS).\n"
            "Parse natural language geometry problems (in English or Vietnamese) into exact formal predicates.\n\n"
            "Predicate Syntax Rules:\n"
            "- Points are uppercase single letters: A, B, C, ...\n"
            "- Segments: AB, BC, AC (two adjacent uppercase letters)\n"
            "- Angles: Angle(BAC) means ∠BAC with vertex at A\n"
            "- Triangle: Triangle(A,B,C)\n"
            "- Right Triangle: RightTriangle(A,B,C) + RightAngle(Angle(BAC)) for right angle at A\n"
            "- Congruence: Congruent(AB,CD) — sort args alphabetically for commutativity\n"
            "- Equality: Equal(Angle(ABC),Angle(DEF)) or Equal(x,60)\n"
            "- Parallel: Parallel(AB,CD)\n"
            "- Perpendicular: Perpendicular(AB,CD)\n"
            "- Isosceles: Triangle(A,B,C) + Congruent(AB,AC) for isosceles with apex A\n"
            "- Pythagorean equality: BC^2=AB^2+AC^2\n\n"
            "Examples:\n"
            "  'Cho tam giác ABC cân tại A. Chứng minh góc B bằng góc C'\n"
            "  → initial_facts: ['Triangle(A,B,C)', 'Congruent(AB,AC)'], goal_fact: 'Equal(Angle(ABC),Angle(ACB))'\n\n"
            "  'Given right triangle ABC with right angle at A, prove BC^2=AB^2+AC^2'\n"
            "  → initial_facts: ['RightTriangle(A,B,C)', 'RightAngle(Angle(BAC))'], goal_fact: 'BC^2=AB^2+AC^2'\n\n"
            "  'If Parallel(AB,CD) and Parallel(CD,EF), prove Parallel(AB,EF)'\n"
            "  → initial_facts: ['Parallel(AB,CD)', 'Parallel(CD,EF)'], goal_fact: 'Parallel(AB,EF)'\n\n"
            "Output ONLY the parsed structures. Do not solve the problem."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Parse this geometry problem: {query}")
        ])

        logger.info("Calling LLM geometry parser...")

        try:
            structured_llm = llm.with_structured_output(ExtractedGeoProblem)
            chain = prompt | structured_llm
            response = chain.invoke({"query": query})
            facts, goal = response.initial_facts, response.goal_fact
        except Exception as e:
            logger.info("Structured output failed (%s). Using JSON parser.", e)
            from langchain_core.output_parsers import JsonOutputParser
            parser = JsonOutputParser(pydantic_object=ExtractedGeoProblem)
            prompt_with_fmt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\nOutput ONLY valid JSON:\n{format_instructions}"),
                ("human", "{query}")
            ])
            chain = prompt_with_fmt | llm | parser
            response = chain.invoke({
                "query": query,
                "format_instructions": parser.get_format_instructions()
            })
            facts, goal = response["initial_facts"], response["goal_fact"]

        logger.info("[LLM Parser] Extracted facts: %s, goal: %s", facts, goal)
        return facts, goal

    except Exception as e:
        logger.error("LLM parser failed: %s. Falling back to regex.", e)
        return fallback_query_parser(query)


# ---------------------------------------------------------------------------
# Qdrant semantic mapping
# ---------------------------------------------------------------------------

def map_text_to_graph_fact(
    text: str,
    qdrant_client,
    collection_name: str = COLLECTION_NAME,
) -> Fact:
    """
    Maps a raw text predicate to a canonical Fact using 2-tier Qdrant lookup:
    1. Exact payload scroll (zero overhead, perfect match).
    2. High-confidence vector search (score >= 0.85 threshold).
    Falls back to ad-hoc Fact construction if Qdrant is unreachable.
    """
    text = text.strip()
    payload_match = None

    # Tier 1: Exact payload match
    try:
        val_results, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="value", match=MatchValue(value=text))
            ]),
            limit=1,
        )
        if val_results:
            payload_match = val_results[0].payload
            logger.info("Exact value match found for '%s'", text)
        else:
            lbl_results, _ = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="label", match=MatchValue(value=text))
                ]),
                limit=1,
            )
            if lbl_results:
                payload_match = lbl_results[0].payload
                logger.info("Exact label match found for '%s'", text)
    except Exception as e:
        logger.warning("Exact payload matching failed for '%s': %s", text, e)

    # Tier 2: High-confidence vector search
    if not payload_match:
        try:
            model = _get_embedding_model()
            query_vector = model.encode(text).tolist()
            results = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=1,
            )
            if results.points:
                match = results.points[0]
                HIGH_CONFIDENCE_THRESHOLD = 0.85
                if match.score >= HIGH_CONFIDENCE_THRESHOLD:
                    payload_match = match.payload
                    logger.info(
                        "High-confidence vector match (%.4f) found for '%s'",
                        match.score, text
                    )
                else:
                    logger.info(
                        "Vector match score %.4f below threshold %.2f for '%s'",
                        match.score, HIGH_CONFIDENCE_THRESHOLD, text
                    )
        except Exception as e:
            logger.warning("Vector search failed for '%s': %s", text, e)

    # Build Fact from match or ad-hoc fallback
    if payload_match:
        canonical_value = payload_match["value"]
        canonical_label = payload_match.get("label", canonical_value)
        neo4j_id = payload_match.get("neo4j_id", f"geo_fact_{canonical_value}")
        logger.info("Mapped '%s' → '%s'", text, canonical_value)
    else:
        canonical_value = text
        canonical_label = text
        neo4j_id = f"geo_fact_{abs(hash(text))}"
        logger.info("Ad-hoc fallback for '%s'", text)

    return Fact(
        id=neo4j_id,
        value=canonical_value,
        domain=DOMAIN,
        attributes={"label": canonical_label},
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def route_query(query: str) -> Tuple[List[Fact], Fact]:
    """
    Main entrypoint for the GeoIPS Neuro-Symbolic Router.
    Parses a natural language geometry query and maps entities to formal Facts.

    Args:
        query: Natural language geometry problem (English or Vietnamese).

    Returns:
        (initial_facts, goal_fact) — lists of Fact objects ready for the solver.
    """
    # Step 1: Parse text → structured predicates
    facts_text, goal_text = llm_query_parser(query)

    # Step 2: Connect to Qdrant (cloud or local via factory)
    qdrant = None
    for attempt in range(1, 4):
        try:
            qdrant = get_qdrant_client()
            qdrant.get_collections()
            break
        except Exception as e:
            if attempt == 3:
                logger.warning(
                    "Could not connect to Qdrant after 3 attempts: %s. Using ad-hoc routing.", e
                )
                qdrant = None
            else:
                time.sleep(1.0 * attempt)

    # Step 3: Map text predicates → Fact objects
    mapped_facts: List[Fact] = []
    for f_text in facts_text:
        if qdrant:
            mapped_facts.append(map_text_to_graph_fact(f_text, qdrant))
        else:
            mapped_facts.append(
                Fact(id=f"geo_fact_{abs(hash(f_text))}", value=f_text, domain=DOMAIN)
            )

    if qdrant and goal_text:
        mapped_goal = map_text_to_graph_fact(goal_text, qdrant)
    else:
        mapped_goal = Fact(
            id=f"geo_fact_{abs(hash(goal_text))}", value=goal_text, domain=DOMAIN
        )

    # Deduplicate facts, remove goal if it slipped in
    seen: set = set()
    deduped: List[Fact] = []
    for f in mapped_facts:
        if f.value not in seen and f.value != mapped_goal.value:
            seen.add(f.value)
            deduped.append(f)

    logger.info(
        "Routing complete. %d initial facts, goal: %s", len(deduped), mapped_goal.value
    )
    return deduped, mapped_goal
