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
    {"id": "geo_triangle_angle_sum", "name": "Triangle Angle Sum", "inputs": ["Triangle(A,B,C)"], "outputs": ["Equal(Add(Angle(BAC),Angle(ABC),Angle(ACB)),180)"], "description": "Angles of a triangle sum to 180°."},
    {"id": "geo_isosceles_base_angles", "name": "Isosceles Base Angles", "inputs": ["Triangle(A,B,C)", "Congruent(AB,AC)"], "outputs": ["Equal(Angle(ABC),Angle(ACB))"], "description": "Base angles of isosceles triangle are equal."},
    {"id": "geo_isosceles_reverse", "name": "Converse Isosceles", "inputs": ["Triangle(A,B,C)", "Equal(Angle(ABC),Angle(ACB))"], "outputs": ["Congruent(AB,AC)"], "description": "Equal base angles implies isosceles."},
    {"id": "geo_sas_congruence", "name": "SAS Congruence", "inputs": ["Congruent(AB,DE)", "Equal(Angle(BAC),Angle(EDF))", "Congruent(AC,DF)"], "outputs": ["CongruentTriangles(ABC,DEF)"], "description": "Side-Angle-Side congruence."},
    {"id": "geo_asa_congruence", "name": "ASA Congruence", "inputs": ["Equal(Angle(BAC),Angle(EDF))", "Congruent(AB,DE)", "Equal(Angle(ABC),Angle(DEF))"], "outputs": ["CongruentTriangles(ABC,DEF)"], "description": "Angle-Side-Angle congruence."},
    {"id": "geo_sss_congruence", "name": "SSS Congruence", "inputs": ["Congruent(AB,DE)", "Congruent(BC,EF)", "Congruent(AC,DF)"], "outputs": ["CongruentTriangles(ABC,DEF)"], "description": "Side-Side-Side congruence."},
    {"id": "geo_pythagoras", "name": "Pythagorean Theorem", "inputs": ["RightTriangle(A,B,C)", "RightAngle(Angle(BAC))"], "outputs": ["Equal(Pow(BC,2),Add(Pow(AB,2),Pow(AC,2)))", "BC^2=AB^2+AC^2"], "description": "In right triangle ABC at A: BC²=AB²+AC²."},
    {"id": "geo_pythagoras_converse", "name": "Pythagorean Converse", "inputs": ["Triangle(A,B,C)", "BC^2=AB^2+AC^2"], "outputs": ["RightTriangle(A,B,C)", "RightAngle(Angle(BAC))"], "description": "If BC²=AB²+AC² then right-angled at A."},
    {"id": "geo_perp_to_parallel", "name": "Perp to Parallel", "inputs": ["Perpendicular(L,a)", "Parallel(a,b)"], "outputs": ["Perpendicular(L,b)"], "description": "Line perp to one parallel is perp to the other."},
    {"id": "geo_parallel_from_perp", "name": "Two Lines Perp to Same", "inputs": ["Perpendicular(L,a)", "Perpendicular(L,b)"], "outputs": ["Parallel(a,b)"], "description": "Two lines perp to same line are parallel."},
    {"id": "geo_thales", "name": "Thales Theorem", "inputs": ["Diameter(AB,Circle(O))", "PointOnCircle(C,Circle(O))"], "outputs": ["RightAngle(Angle(ACB))"], "description": "Angle in a semicircle is 90°."},
    {"id": "geo_congruent_tri_sides", "name": "Congruent Triangles → Sides", "inputs": ["CongruentTriangles(ABC,DEF)"], "outputs": ["Congruent(AB,DE)", "Congruent(BC,EF)", "Congruent(AC,DF)"], "description": "Congruent triangles have congruent sides."},
    {"id": "geo_congruent_tri_angles", "name": "Congruent Triangles → Angles", "inputs": ["CongruentTriangles(ABC,DEF)"], "outputs": ["Equal(Angle(BAC),Angle(EDF))", "Equal(Angle(ABC),Angle(DEF))", "Equal(Angle(ACB),Angle(DFE))"], "description": "Congruent triangles have equal angles."},
    {"id": "geo_midpoint_theorem", "name": "Midsegment Theorem", "inputs": ["Triangle(A,B,C)", "Midpoint(M,AB)", "Midpoint(N,AC)"], "outputs": ["Parallel(MN,BC)", "Equal(Length(MN),Div(Length(BC),2))"], "description": "Midsegment is parallel to base and half its length."},

    # ── Equality algebra ──────────────────────────────────────────────────────
    {"id": "geo_equal_symmetric", "name": "Equality Symmetric",
     "inputs": ["Equal(?X,?Y)"], "outputs": ["Equal(?Y,?X)"],
     "description": "If a=b then b=a."},
    {"id": "geo_equal_transitive", "name": "Equality Transitive",
     "inputs": ["Equal(?X,?Y)", "Equal(?Y,?Z)"], "outputs": ["Equal(?X,?Z)"],
     "description": "If a=b and b=c then a=c."},

    # ── Right-angle facts ─────────────────────────────────────────────────────
    {"id": "geo_right_angle_90", "name": "Right Angle is 90",
     "inputs": ["RightAngle(Angle(?X))"],
     "outputs": ["Equal(Angle(?X),90)"],
     "description": "A right angle equals 90 degrees."},
    {"id": "geo_right_triangle_expand", "name": "RightTriangle Expand",
     "inputs": ["RightTriangle(A,B,C)"],
     "outputs": ["Triangle(A,B,C)", "RightAngle(Angle(BAC))"],
     "description": "Right triangle at A implies Triangle + RightAngle at A."},

    # ── Exterior angle theorem ────────────────────────────────────────────────
    {"id": "geo_exterior_angle", "name": "Exterior Angle Theorem",
     "inputs": ["Triangle(A,B,C)", "ExteriorAngle(?E,A,BC)"],
     "outputs": ["Equal(Angle(?E),Add(Angle(BAC),Angle(ABC)))"],
     "description": "Exterior angle = sum of two non-adjacent interior angles."},

    # ── Equilateral triangle ──────────────────────────────────────────────────
    {"id": "geo_equilateral_all_60", "name": "Equilateral Triangle Angles",
     "inputs": ["Triangle(A,B,C)", "Congruent(AB,BC)", "Congruent(BC,AC)"],
     "outputs": ["Equal(Angle(BAC),60)", "Equal(Angle(ABC),60)", "Equal(Angle(ACB),60)"],
     "description": "All angles of equilateral triangle = 60 degrees."},

    # ── Circle theorems ───────────────────────────────────────────────────────
    {"id": "geo_thales_right_angle", "name": "Thales: Angle in Semicircle",
     "inputs": ["Diameter(AB,Circle(O))", "PointOnCircle(C,Circle(O))"],
     "outputs": ["RightAngle(Angle(ACB))", "Equal(Angle(ACB),90)"],
     "description": "Angle inscribed in a semicircle is 90 degrees."},
    {"id": "geo_cyclic_quad_opposite", "name": "Cyclic Quadrilateral Opposite Angles",
     "inputs": ["CyclicQuadrilateral(A,B,C,D,Circle(O))"],
     "outputs": ["Equal(Add(Angle(DAB),Angle(BCD)),180)",
                 "Equal(Add(Angle(ABC),Angle(CDA)),180)"],
     "description": "Opposite angles of a cyclic quadrilateral sum to 180 degrees."},

    # ── AA Similarity ─────────────────────────────────────────────────────────
    {"id": "geo_aa_similarity", "name": "AA Similarity",
     "inputs": ["Triangle(A,B,C)", "Triangle(D,E,F)",
                "Equal(Angle(BAC),Angle(EDF))", "Equal(Angle(ABC),Angle(DEF))"],
     "outputs": ["SimilarTriangles(ABC,DEF)"],
     "description": "Two triangles with two equal angles are similar (AA)."},
    {"id": "geo_similar_tri_angles", "name": "Similar Triangles: Equal Angles",
     "inputs": ["SimilarTriangles(ABC,DEF)"],
     "outputs": ["Equal(Angle(BAC),Angle(EDF))", "Equal(Angle(ABC),Angle(DEF))",
                 "Equal(Angle(ACB),Angle(DFE))"],
     "description": "Similar triangles have equal corresponding angles."},
    {"id": "geo_similar_tri_ratios", "name": "Similar Triangles: Proportional Sides",
     "inputs": ["SimilarTriangles(ABC,DEF)"],
     "outputs": ["Equal(Div(Length(AB),Length(DE)),Div(Length(BC),Length(EF)))",
                 "Equal(Div(Length(AB),Length(DE)),Div(Length(AC),Length(DF)))"],
     "description": "Similar triangles have proportional corresponding sides."},

    # ── Midpoint ──────────────────────────────────────────────────────────────
    {"id": "geo_midpoint_halves", "name": "Midpoint Halves Segment",
     "inputs": ["Midpoint(M,AB)"],
     "outputs": ["Equal(Length(AM),Length(MB))",
                 "Equal(Length(AM),Div(Length(AB),2))"],
     "description": "Midpoint divides segment into two equal halves."},
]


def _get_rules(neo4j_conn: Neo4jConnection) -> list:
    """Load rules from Neo4j, fallback to built-in set."""
    rules = []
    try:
        if neo4j_conn.verify_connectivity():
            with neo4j_conn.get_session() as session:
                raw = _load_rules_from_neo4j(session)
                rules = [PARSER.parse_rule(r) for r in raw]
                logger.info("Loaded %d rules from Neo4j", len(rules))
    except Exception as e:
        logger.warning("Neo4j rule load failed: %s — using fallback rules", e)
    if not rules:
        logger.info("Using built-in fallback geometry rules (%d)", len(FALLBACK_GEOMETRY_RULES))
        rules = [PARSER.parse_rule(r) for r in FALLBACK_GEOMETRY_RULES]
    return rules


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
    """
    from geo_engine.auxiliary_agent import AuxiliaryConstructionAgent
    from rag_agent.llm_factory import get_llm

    try:
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
    if goal_reached:
        return (
            "You are an expert plane geometry tutor and proof explainer for GeoIPS.\n"
            "Your task is to translate a symbolic geometry proof trace into a clear, engaging, "
            "pedagogically rich explanation suitable for a high-school student.\n"
            "Reference specific Euclidean theorems and axioms by name (e.g., Thales' Theorem, SAS Congruence).\n"
            "Use LaTeX notation for mathematical expressions where helpful (e.g., $\\angle ABC = 90°$).\n"
            "Use markdown headers, bullet points, and clear step-by-step structure.\n"
            "STRICT: Only explain steps that appear in the trace. Do not hallucinate extra steps."
        )
    else:
        return (
            "You are an expert plane geometry tutor for GeoIPS.\n"
            "The symbolic solver could NOT prove the goal from the given facts.\n"
            "Explain clearly to a high-school student:\n"
            "1. What facts were given and what the goal was.\n"
            "2. What intermediate facts (if any) were deduced before the solver got stuck.\n"
            "3. WHY the proof failed — is a hypothesis missing? Is there a theorem needed that isn't in the KB?\n"
            "4. What additional information or constructions might help.\n"
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
            "The symbolic engine successfully proved the goal. Here is the step-by-step breakdown:\n\n",
        ]
        if request.auxiliary_constructions:
            parts.append(f"**Auxiliary Constructions Used:** {', '.join(request.auxiliary_constructions)}\n\n")
        parts.append("## Deduction Steps\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Step {i+1}: `{step.rule_id}`")
            parts.append(f"- **Rule:** `{step.fired_rule_repr}`")
            parts.append(f"- **New facts:** `{', '.join(step.new_facts)}`\n")
        parts.append("## Conclusion\nThe goal has been formally proved by the symbolic engine. ✓")
    else:
        parts = [
            "# ⚠️ Proof Attempt — Goal Not Reached\n",
            f"**Query:** *{request.query}*\n",
            "The solver could not establish the goal from the given facts.\n\n",
            "## Attempted Steps\n",
        ]
        if not request.execution_trace:
            parts.append("No rules were triggered — the given facts do not satisfy any theorem preconditions.\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Step {i+1}: `{step.rule_id}`")
            parts.append(f"- **Rule:** `{step.fired_rule_repr}`")
            parts.append(f"- **New facts:** `{', '.join(step.new_facts)}`\n")
        parts.append("## Analysis\n⚠️ **Logical gap detected.** Either the initial conditions are insufficient, or the Knowledge Base is missing a bridging theorem.")

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
        parts.append("## Deduction Steps\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Step {i+1}: `{step.rule_id}`\n")
            parts.append(f"- **Rule:** `{step.fired_rule_repr}`\n")
            parts.append(f"- **New facts:** `{', '.join(step.new_facts)}`\n\n")
        parts.append("## ✓ Conclusion\nGoal formally proved by the symbolic engine.\n")
    else:
        parts = [
            "# ⚠️ Proof Attempt — Goal Not Reached\n\n",
            f"**Query:** *{request.query}*\n\n",
            "The solver could not establish the goal from the given facts.\n\n",
        ]
        if not request.execution_trace:
            parts.append("No rules were triggered.\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Step {i+1}: `{step.rule_id}`\n")
            parts.append(f"- **Rule:** `{step.fired_rule_repr}`\n")
            parts.append(f"- **New facts:** `{', '.join(step.new_facts)}`\n\n")
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
