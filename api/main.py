import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Core engine and domain imports
from core_engine import ForwardChainingEngine, BackwardChainingEngine
from domains.chemistry import ChemistryParser
from domains.geometry import GeometryParser
from domains.algebra import AlgebraParser
from graph_db.connection import Neo4jConnection

# RAG agent imports
from rag_agent.router import route_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("omni-ips-api")

app = FastAPI(
    title="Omni-IPS API Gateway",
    description="Production-ready Neuro-Symbolic & GraphRAG Multi-domain Intelligent Problem Solver.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Domain Parser registry
PARSERS = {
    "chemistry": ChemistryParser(),
    "geometry": GeometryParser(),
    "algebra": AlgebraParser()
}

# --------------------------------------------------------------------------
# Request & Response Schemas
# --------------------------------------------------------------------------

class SolveRequest(BaseModel):
    domain: str = Field(..., description="Target domain, e.g., 'chemistry', 'geometry', 'algebra'")
    facts: List[str] = Field(..., description="List of raw assertions/facts.")
    goal: str = Field(..., description="Target goal to deduce.")
    strategy: str = Field("forward", description="Reasoning strategy: 'forward' or 'backward'")

class ExecutionStepResponse(BaseModel):
    rule_id: str = Field(..., description="ID of the fired rule.")
    fired_rule_repr: str = Field(..., description="String representation of the rule fired.")
    new_facts: List[str] = Field(..., description="New facts deduced at this step.")

class SolveResponse(BaseModel):
    goal_reached: bool = Field(..., description="Whether the goal fact was successfully deduced.")
    applied_rule_ids: List[str] = Field(..., description="Ordered list of applied rule IDs.")
    execution_trace: List[ExecutionStepResponse] = Field(..., description="Detailed proof steps.")
    known_facts: List[str] = Field(..., description="Final list of all deduced and initial facts.")

class SolveQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query, e.g., 'I have sodium and water, how do I make sodium hydroxide?'")
    domain: str = Field(..., description="Logical domain, e.g., 'chemistry', 'geometry'")

class SolveQueryResponse(BaseModel):
    query: str = Field(..., description="Original natural language query.")
    domain: str = Field(..., description="Target domain.")
    mapped_initial_facts: List[str] = Field(..., description="Semantic facts mapped from Qdrant.")
    mapped_goal: str = Field(..., description="Semantic goal mapped from Qdrant.")
    goal_reached: bool = Field(..., description="Whether the goal was reached.")
    applied_rule_ids: List[str] = Field(..., description="Sequence of fired rule IDs.")
    execution_trace: List[ExecutionStepResponse] = Field(..., description="Formal proof steps.")
    known_facts: List[str] = Field(..., description="All final known facts.")

class ExplainRequest(BaseModel):
    query: str = Field(..., description="Original natural language query.")
    domain: str = Field(..., description="Target domain.")
    execution_trace: List[ExecutionStepResponse] = Field(..., description="Solver execution trace.")
    goal_reached: bool = Field(True, description="Whether the goal was successfully reached.")

class ExplainResponse(BaseModel):
    explanation: str = Field(..., description="Rich, human-friendly, educational explanation.")
    structured: bool = Field(..., description="True if generated via LLM, False if generated via fallback template.")


# --------------------------------------------------------------------------
# API Endpoints (FastAPI & GraphRAG)
# --------------------------------------------------------------------------

@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health_check():
    """Verify system health and database connectivity."""
    db_conn = Neo4jConnection()
    db_connected = db_conn.verify_connectivity()
    db_conn.close()
    
    return {
        "status": "healthy" if db_connected else "degraded",
        "neo4j_connected": db_connected,
        "environment": {
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
            "qdrant_port": os.getenv("QDRANT_PORT", "6333")
        }
    }


@app.post("/solve", response_model=SolveResponse, tags=["Inference Engine"])
async def solve_problem(request: SolveRequest):
    """
    Solve a problem in a specific domain using structured initial facts and goal.
    """
    domain = request.domain.lower()
    if domain not in PARSERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported domain: '{domain}'. Choose from {list(PARSERS.keys())}"
        )
    
    parser = PARSERS[domain]
    strategy = request.strategy.lower()
    
    try:
        initial_facts = [parser.parse_fact(f, f"init_{i}") for i, f in enumerate(request.facts)]
        goal_fact = parser.parse_fact(request.goal, "goal_0")
        
        # Load rules from Neo4j
        rules = []
        try:
            db_conn = Neo4jConnection()
            if db_conn.verify_connectivity():
                with db_conn.get_session() as session:
                    result = session.run(
                        "MATCH (r:Rule) WHERE r.domain = $domain "
                        "OPTIONAL MATCH (f_in:Fact)-[:HAS_INPUT]->(r) "
                        "OPTIONAL MATCH (r)-[:HAS_OUTPUT]->(f_out:Fact) "
                        "WITH r, "
                        "     CASE WHEN r.inputs IS NOT NULL THEN r.inputs ELSE collect(DISTINCT f_in.value) END AS inputs, "
                        "     CASE WHEN r.outputs IS NOT NULL THEN r.outputs ELSE collect(DISTINCT f_out.value) END AS outputs "
                        "RETURN r.id AS id, r.name AS name, inputs, outputs, r.description AS description",
                        domain=domain
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
                    
                    rules = [parser.parse_rule(r) for r in raw_rules]
                    logger.info("Loaded %d rules from Neo4j for domain '%s'", len(rules), domain)
            db_conn.close()
        except Exception as db_err:
            logger.warning("Failed to fetch rules from Neo4j, falling back to mock: %s", db_err, exc_info=True)
            
        # Predefined fallback rules (representative subset of the full knowledge base)
        if not rules:
            logger.info("Using built-in fallback rules for domain '%s'", domain)
            if domain == "chemistry":
                raw_rules = [
                    {"id": "rxn_single_na_h2o", "name": "Sodium Reacting with Water", "inputs": ["Na", "H2O"], "outputs": ["NaOH", "H2"], "description": "2Na + 2H₂O → 2NaOH + H₂."},
                    {"id": "rxn_synth_water", "name": "Synthesis of Water", "inputs": ["H2", "O2"], "outputs": ["H2O"], "description": "2H₂ + O₂ → 2H₂O."},
                    {"id": "rxn_double_naoh_hcl", "name": "Neutralization: NaOH + HCl", "inputs": ["NaOH", "HCl"], "outputs": ["NaCl", "H2O"], "description": "NaOH + HCl → NaCl + H₂O."},
                    {"id": "rxn_double_na_hcl", "name": "Sodium Acid Reaction", "inputs": ["Na", "HCl"], "outputs": ["NaCl", "H2"], "description": "2Na + 2HCl → 2NaCl + H₂."},
                    {"id": "rxn_decomp_caco3", "name": "Decomposition of CaCO₃", "inputs": ["CaCO3"], "outputs": ["CaO", "CO2"], "description": "CaCO₃ → CaO + CO₂."},
                    {"id": "rxn_cao_h2o", "name": "Quicklime + Water", "inputs": ["CaO", "H2O"], "outputs": ["Ca(OH)2"], "description": "CaO + H₂O → Ca(OH)₂."},
                    {"id": "rxn_single_zn_hcl", "name": "Zinc + HCl", "inputs": ["Zn", "HCl"], "outputs": ["ZnCl2", "H2"], "description": "Zn + 2HCl → ZnCl₂ + H₂."},
                    {"id": "rxn_combust_ch4", "name": "Combustion of Methane", "inputs": ["CH4", "O2"], "outputs": ["CO2", "H2O"], "description": "CH₄ + 2O₂ → CO₂ + 2H₂O."},
                ]
            elif domain == "geometry":
                raw_rules = [
                    {"id": "geo_congruence_transitive", "name": "Congruence Transitivity", "inputs": ["Congruent(AB,CD)", "Congruent(CD,EF)"], "outputs": ["Congruent(AB,EF)"], "description": "Transitivity of congruence."},
                    {"id": "geo_perp_symmetry", "name": "Perpendicular Symmetry", "inputs": ["Perpendicular(AB,CD)"], "outputs": ["Perpendicular(CD,AB)"], "description": "Perpendicularity is symmetric."},
                    {"id": "geo_parallel_transitive", "name": "Parallel Transitivity", "inputs": ["Parallel(a,b)", "Parallel(b,c)"], "outputs": ["Parallel(a,c)"], "description": "Transitivity of parallel lines."},
                    {"id": "geo_triangle_angle_sum", "name": "Triangle Angle Sum", "inputs": ["Triangle(A,B,C)"], "outputs": ["Equal(Add(Angle(BAC),Angle(ABC),Angle(ACB)),180)"], "description": "Angles of a triangle sum to 180°."},
                    {"id": "geo_isosceles_base_angles", "name": "Isosceles Base Angles", "inputs": ["Triangle(A,B,C)", "Congruent(AB,AC)"], "outputs": ["Equal(Angle(ABC),Angle(ACB))"], "description": "Base angles of isosceles triangle are equal."},
                    {"id": "geo_sas_congruence", "name": "SAS Congruence", "inputs": ["Congruent(AB,DE)", "Equal(Angle(BAC),Angle(EDF))", "Congruent(AC,DF)"], "outputs": ["CongruentTriangles(ABC,DEF)"], "description": "Side-Angle-Side congruence."},
                    {"id": "geo_perp_to_parallel", "name": "Perp to Parallel", "inputs": ["Perpendicular(L,a)", "Parallel(a,b)"], "outputs": ["Perpendicular(L,b)"], "description": "Line perp to one parallel line is perp to the other."},
                ]
            elif domain == "algebra":
                raw_rules = [
                    {"id": "alg_sub_two", "name": "Subtraction Property of 2 (Demo)", "inputs": ["x+2=5", "Subtract(2,both_sides)"], "outputs": ["x=3"], "description": "Demo: x+2=5 → x=3."},
                    {"id": "alg_sub_both_sides", "name": "Subtraction Property of Equality", "inputs": ["Equation(LHS,RHS)", "Subtract(Val)"], "outputs": ["Equation(LHS-Val,RHS-Val)"], "description": "a=b → a-c=b-c."},
                    {"id": "alg_add_both_sides", "name": "Addition Property of Equality", "inputs": ["Equation(LHS,RHS)", "Add(Val)"], "outputs": ["Equation(LHS+Val,RHS+Val)"], "description": "a=b → a+c=b+c."},
                    {"id": "alg_distributive", "name": "Distributive Property", "inputs": ["Expression(a*(b+c))"], "outputs": ["Equal(a*(b+c),a*b+a*c)"], "description": "a(b+c) = ab+ac."},
                    {"id": "alg_square_of_sum", "name": "Square of a Sum", "inputs": ["Expression((a+b)^2)"], "outputs": ["Equal((a+b)^2,a^2+2*a*b+b^2)"], "description": "(a+b)² = a²+2ab+b²."},
                    {"id": "alg_diff_of_squares", "name": "Difference of Squares", "inputs": ["Expression(a^2-b^2)"], "outputs": ["Equal(a^2-b^2,(a+b)*(a-b))"], "description": "a²-b² = (a+b)(a-b)."},
                    {"id": "alg_quadratic_formula", "name": "Quadratic Formula", "inputs": ["QuadraticEquation(a*x^2+b*x+c,0)", "NotEqual(a,0)"], "outputs": ["Equal(x,(-b±sqrt(b^2-4*a*c))/(2*a))"], "description": "x = (-b±√(b²-4ac))/(2a)."},
                ]
            else:
                raw_rules = []
            rules = [parser.parse_rule(r) for r in raw_rules]

        if strategy == "forward":
            engine = ForwardChainingEngine(rules)
        elif strategy == "backward":
            engine = BackwardChainingEngine(rules)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported reasoning strategy: '{strategy}'")
            
        inference_result = engine.solve(initial_facts, goal_fact)
        
        steps = [
            ExecutionStepResponse(
                rule_id=step.rule_id,
                fired_rule_repr=step.fired_rule_repr,
                new_facts=[f.value for f in step.new_facts]
            ) for step in inference_result.execution_trace
        ]
        known_facts = sorted(list(set(f.value for f in inference_result.final_facts)))
        
        return SolveResponse(
            goal_reached=inference_result.goal_reached,
            applied_rule_ids=inference_result.applied_rule_ids,
            execution_trace=steps,
            known_facts=known_facts
        )
        
    except Exception as e:
        logger.error("Error during solve execution: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference execution error: {str(e)}")


@app.post("/api/solve", response_model=SolveQueryResponse, tags=["GraphRAG Inference"])
async def solve_query(request: SolveQueryRequest):
    """
    GraphRAG Endpoint: Resolves a natural language query by mapping it to Graph Facts,
    then executes the pure symbolic core engine.
    """
    domain = request.domain.lower()
    if domain not in PARSERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported domain: '{domain}'. Choose from {list(PARSERS.keys())}"
        )

    parser = PARSERS[domain]
    
    try:
        # Step 1: Run the Neuro-Symbolic Router (NLP -> Qdrant -> Neo4j Nodes)
        logger.info("Routing query: '%s' in domain '%s'", request.query, domain)
        initial_facts, goal_fact = route_query(request.query, domain)
        
        if not initial_facts:
            raise HTTPException(
                status_code=400, 
                detail="Could not map any initial facts from your query. Please be more specific."
            )

        # Step 2: Load rules from Neo4j
        rules = []
        try:
            db_conn = Neo4jConnection()
            if db_conn.verify_connectivity():
                with db_conn.get_session() as session:
                    result = session.run(
                        "MATCH (r:Rule) WHERE r.domain = $domain "
                        "OPTIONAL MATCH (f_in:Fact)-[:HAS_INPUT]->(r) "
                        "OPTIONAL MATCH (r)-[:HAS_OUTPUT]->(f_out:Fact) "
                        "WITH r, "
                        "     CASE WHEN r.inputs IS NOT NULL THEN r.inputs ELSE collect(DISTINCT f_in.value) END AS inputs, "
                        "     CASE WHEN r.outputs IS NOT NULL THEN r.outputs ELSE collect(DISTINCT f_out.value) END AS outputs "
                        "RETURN r.id AS id, r.name AS name, inputs, outputs, r.description AS description",
                        domain=domain
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
                    rules = [parser.parse_rule(r) for r in raw_rules]
                    logger.info("Loaded %d rules from Neo4j for query resolving in domain '%s'", len(rules), domain)
            db_conn.close()
        except Exception as db_err:
            logger.warning("Failed to fetch rules from Neo4j, using built-in fallback: %s", db_err, exc_info=True)

        # Default rules fallback (representative subset of the full knowledge base)
        if not rules:
            logger.info("Using built-in fallback rules for query resolving on domain '%s'", domain)
            if domain == "chemistry":
                raw_rules = [
                    {"id": "rxn_single_na_h2o", "name": "Sodium Reacting with Water", "inputs": ["Na", "H2O"], "outputs": ["NaOH", "H2"], "description": "2Na + 2H₂O → 2NaOH + H₂."},
                    {"id": "rxn_double_naoh_hcl", "name": "Neutralization: NaOH + HCl", "inputs": ["NaOH", "HCl"], "outputs": ["NaCl", "H2O"], "description": "NaOH + HCl → NaCl + H₂O."},
                    {"id": "rxn_double_na_hcl", "name": "Sodium Acid Reaction", "inputs": ["Na", "HCl"], "outputs": ["NaCl", "H2"], "description": "2Na + 2HCl → 2NaCl + H₂."},
                    {"id": "rxn_synth_water", "name": "Synthesis of Water", "inputs": ["H2", "O2"], "outputs": ["H2O"], "description": "2H₂ + O₂ → 2H₂O."},
                    {"id": "rxn_decomp_caco3", "name": "Decomposition of CaCO₃", "inputs": ["CaCO3"], "outputs": ["CaO", "CO2"], "description": "CaCO₃ → CaO + CO₂."},
                    {"id": "rxn_cao_h2o", "name": "Quicklime + Water", "inputs": ["CaO", "H2O"], "outputs": ["Ca(OH)2"], "description": "CaO + H₂O → Ca(OH)₂."},
                    {"id": "rxn_single_zn_hcl", "name": "Zinc + HCl", "inputs": ["Zn", "HCl"], "outputs": ["ZnCl2", "H2"], "description": "Zn + 2HCl → ZnCl₂ + H₂."},
                    {"id": "rxn_combust_ch4", "name": "Combustion of Methane", "inputs": ["CH4", "O2"], "outputs": ["CO2", "H2O"], "description": "CH₄ + 2O₂ → CO₂ + 2H₂O."},
                ]
            elif domain == "geometry":
                raw_rules = [
                    {"id": "geo_congruence_transitive", "name": "Congruence Transitivity", "inputs": ["Congruent(AB,CD)", "Congruent(CD,EF)"], "outputs": ["Congruent(AB,EF)"], "description": "Transitivity of congruence."},
                    {"id": "geo_perp_symmetry", "name": "Perpendicular Symmetry", "inputs": ["Perpendicular(AB,CD)"], "outputs": ["Perpendicular(CD,AB)"], "description": "Perpendicularity is symmetric."},
                    {"id": "geo_parallel_transitive", "name": "Parallel Transitivity", "inputs": ["Parallel(a,b)", "Parallel(b,c)"], "outputs": ["Parallel(a,c)"], "description": "Transitivity of parallel lines."},
                    {"id": "geo_triangle_angle_sum", "name": "Triangle Angle Sum", "inputs": ["Triangle(A,B,C)"], "outputs": ["Equal(Add(Angle(BAC),Angle(ABC),Angle(ACB)),180)"], "description": "Angles of a triangle sum to 180°."},
                    {"id": "geo_isosceles_base_angles", "name": "Isosceles Base Angles", "inputs": ["Triangle(A,B,C)", "Congruent(AB,AC)"], "outputs": ["Equal(Angle(ABC),Angle(ACB))"], "description": "Base angles of isosceles triangle are equal."},
                    {"id": "geo_sas_congruence", "name": "SAS Congruence", "inputs": ["Congruent(AB,DE)", "Equal(Angle(BAC),Angle(EDF))", "Congruent(AC,DF)"], "outputs": ["CongruentTriangles(ABC,DEF)"], "description": "Side-Angle-Side congruence."},
                    {"id": "geo_perp_to_parallel", "name": "Perp to Parallel", "inputs": ["Perpendicular(L,a)", "Parallel(a,b)"], "outputs": ["Perpendicular(L,b)"], "description": "Line perp to one parallel line is perp to the other."},
                ]
            elif domain == "algebra":
                raw_rules = [
                    {"id": "alg_sub_two", "name": "Subtraction Property of 2 (Demo)", "inputs": ["x+2=5", "Subtract(2,both_sides)"], "outputs": ["x=3"], "description": "Demo: x+2=5 → x=3."},
                    {"id": "alg_sub_both_sides", "name": "Subtraction Property of Equality", "inputs": ["Equation(LHS,RHS)", "Subtract(Val)"], "outputs": ["Equation(LHS-Val,RHS-Val)"], "description": "a=b → a-c=b-c."},
                    {"id": "alg_add_both_sides", "name": "Addition Property of Equality", "inputs": ["Equation(LHS,RHS)", "Add(Val)"], "outputs": ["Equation(LHS+Val,RHS+Val)"], "description": "a=b → a+c=b+c."},
                    {"id": "alg_distributive", "name": "Distributive Property", "inputs": ["Expression(a*(b+c))"], "outputs": ["Equal(a*(b+c),a*b+a*c)"], "description": "a(b+c) = ab+ac."},
                    {"id": "alg_square_of_sum", "name": "Square of a Sum", "inputs": ["Expression((a+b)^2)"], "outputs": ["Equal((a+b)^2,a^2+2*a*b+b^2)"], "description": "(a+b)² = a²+2ab+b²."},
                    {"id": "alg_diff_of_squares", "name": "Difference of Squares", "inputs": ["Expression(a^2-b^2)"], "outputs": ["Equal(a^2-b^2,(a+b)*(a-b))"], "description": "a²-b² = (a+b)(a-b)."},
                    {"id": "alg_quadratic_formula", "name": "Quadratic Formula", "inputs": ["QuadraticEquation(a*x^2+b*x+c,0)", "NotEqual(a,0)"], "outputs": ["Equal(x,(-b±sqrt(b^2-4*a*c))/(2*a))"], "description": "x = (-b±√(b²-4ac))/(2a)."},
                ]
            else:
                raw_rules = []
            rules = [parser.parse_rule(r) for r in raw_rules]

        # Step 3: Run the deterministic core engine (Forward Chaining)
        engine = ForwardChainingEngine(rules)
        inference_result = engine.solve(initial_facts, goal_fact)
        
        steps = [
            ExecutionStepResponse(
                rule_id=step.rule_id,
                fired_rule_repr=step.fired_rule_repr,
                new_facts=[f.value for f in step.new_facts]
            ) for step in inference_result.execution_trace
        ]
        known_facts = sorted(list(set(f.value for f in inference_result.final_facts)))

        return SolveQueryResponse(
            query=request.query,
            domain=domain,
            mapped_initial_facts=[f.value for f in initial_facts],
            mapped_goal=goal_fact.value,
            goal_reached=inference_result.goal_reached,
            applied_rule_ids=inference_result.applied_rule_ids,
            execution_trace=steps,
            known_facts=known_facts
        )

    except Exception as e:
        logger.error("Error during GraphRAG solve query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"GraphRAG solving error: {str(e)}")


@app.post("/api/explain", response_model=ExplainResponse, tags=["Explainability Agent"])
async def explain_proof(request: ExplainRequest):
    """
    Explainability Agent: Translates a dry logical execution trace into a human-readable,
    highly educational explanation using an LLM (LangChain) or a rich offline template.
    """
    domain = request.domain.lower()

    # Formulate a dry trace text summary
    trace_text = ""
    for idx, step in enumerate(request.execution_trace):
        trace_text += f"Step {idx+1}: Fired Rule [{step.rule_id}] -> {step.fired_rule_repr}. New facts deduced: {step.new_facts}\n"

    # LLM Pathway
    from rag_agent.llm_factory import get_llm
    llm = get_llm(temperature=0.3)
    if llm:
        try:
            from langchain_core.prompts import ChatPromptTemplate

            if request.goal_reached:
                system_prompt = (
                    "You are a Lead AI Architect and expert Neuro-Symbolic educational tutor. "
                    "Your goal is to explain a successful logical proof path found by our symbolic solver "
                    "in the domain '{domain}'. Make the explanation extremely engaging, "
                    "pedagogically rich, and easy to read. "
                    "Explain the chemical reactions, geometry axioms, or algebra theorems "
                    "so that a high-school student can understand them perfectly. "
                    "Use professional formatting, bullet points, and headers.\n"
                    "Strict Constraint: You MUST ground your explanation ONLY in the facts "
                    "and steps presented in the logical trace. Do not invent any new steps or hallucinate rules!"
                )
                human_prompt = "Explain this reasoning pathway for the question: '{query}'\n\nSolver Execution Trace:\n{trace}"
            else:
                system_prompt = (
                    "You are a Lead AI Architect and expert Neuro-Symbolic educational tutor. "
                    "Our symbolic solver attempted to prove the goal in the domain '{domain}' "
                    "but FAILED because there is no valid proof path using the available rules and facts.\n"
                    "Your goal is to explain this failure to a high-school student in an extremely engaging, "
                    "pedagogically rich, and constructive manner. "
                    "Identify what starting facts were mapped, what intermediate facts (if any) were deduced "
                    "in the trace, and explain clearly and mathematically why the goal could not be reached "
                    "(e.g., because the assumptions are not sufficient to prove the claim, or the registered rule database is missing the necessary theorems).\n"
                    "Use professional formatting, bullet points, and headers.\n"
                    "Strict Constraint: Do not hallucinate or claim that the goal was proven or reached! "
                    "Always start by clearly stating that the goal is UNPROVED based on the available knowledge base."
                )
                human_prompt = "Pedagogically analyze the logical failure for this query: '{query}'\n\nSolver Execution Trace:\n{trace}"

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])

            logger.info("Generating proof explanation via model-agnostic LLM instance (Goal Reached: %s)...", request.goal_reached)
            chain = prompt | llm

            response = chain.invoke({
                "domain": domain,
                "query": request.query,
                "trace": trace_text if trace_text else "No rules were triggered."
            })
            
            # Cleanly extract string content
            content_str = ""
            if isinstance(response.content, str):
                content_str = response.content
            elif isinstance(response.content, list):
                text_parts = []
                for part in response.content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif hasattr(part, "text"):
                        text_parts.append(getattr(part, "text"))
                content_str = "".join(text_parts)
            else:
                content_str = str(response.content)

            return ExplainResponse(
                explanation=content_str,
                structured=True
            )

        except Exception as e:
            logger.warning("LangChain explanation generation failed: %s. Using template fallback.", e)

    # Offline Fallback Template-based Explainer
    logger.info("Generating template-based offline proof explanation.")
    
    if request.goal_reached:
        explanation_parts = [
            f"# Educational Proof Explanation (Offline Mode)\n",
            f"**Original Question**: *\"{request.query}\"*\n",
            f"The symbolic core engine has successfully executed a formal deduction trace. Below is the step-by-step breakdown of how the goal was reached:\n",
            f"## 1. Initial State",
            f"We started with the concepts and assertions mapped from the query. These served as our starting points for logical chaining.\n",
            f"## 2. Deduction Steps"
        ]

        if not request.execution_trace:
            explanation_parts.append("No rules were triggered. The goal was already satisfied by the initial facts or could not be proved.")
        else:
            for idx, step in enumerate(request.execution_trace):
                explanation_parts.append(f"### Step {idx+1}: Activating {step.rule_id}")
                explanation_parts.append(f"* **Applied Rule**: `{step.fired_rule_repr}`")
                explanation_parts.append(f"* **New Deductions**: `{', '.join(step.new_facts)}`")
                
                # Domain specific educational descriptions
                if domain == "chemistry":
                    if "na_h2o" in step.rule_id or "r1" in step.rule_id:
                        explanation_parts.append("  * **Educational context**: Sodium (`Na`) is a highly reactive alkali metal. When it comes into contact with water (`H2O`), it undergoes a vigorous oxidation-reduction reaction to synthesize sodium hydroxide (`NaOH`) and release highly flammable hydrogen gas (`H2`).")
                    elif "naoh_hcl" in step.rule_id or "r3" in step.rule_id:
                        explanation_parts.append("  * **Educational context**: This is a classic neutralization reaction. The strong base sodium hydroxide (`NaOH`) reacts with the strong acid hydrochloric acid (`HCl`) to form water (`H2O`) and table salt, sodium chloride (`NaCl`), in a standard double-displacement reaction.")
                    else:
                        explanation_parts.append("  * **Educational context**: A chemical reaction took place combining the reactants to synthesize the products according to standard stoichiometric ratios.")
                elif domain == "geometry":
                    if "trans" in step.rule_id or "t_trans" in step.rule_id:
                        explanation_parts.append("  * **Educational context**: By Euclid's first common notion (Things which equal the same thing also equal one another), congruence is transitive. Since Segment 1 is congruent to Segment 2, and Segment 2 is congruent to Segment 3, Segment 1 is geometrically congruent to Segment 3.")
                    else:
                        explanation_parts.append("  * **Educational context**: A geometric theorem or axiom was applied to prove the congruence or relation between the geometric entities.")
                else:
                    explanation_parts.append("  * **Educational context**: Applied algebraic equivalence rules to transform the equations and isolate variables.")
                explanation_parts.append("")

        explanation_parts.append("## 3. Conclusion")
        explanation_parts.append("By executing the dry logical proof sequence above, the goal has been successfully established and verified to be 100% mathematically and scientifically correct!")
    else:
        explanation_parts = [
            f"# Educational Proof Analysis (Goal Unproved - Offline Mode)\n",
            f"**Original Question**: *\"{request.query}\"*\n",
            f"The symbolic core engine attempted to deduce the target goal using forward-chaining rules but was **unable to prove it** from the given assumptions.\n",
            f"## 1. Initial State & Mapping",
            f"We started with the concepts and assertions mapped from the query. These served as our initial set of facts.\n",
            f"## 2. Saturation & Attempted Deductions"
        ]

        if not request.execution_trace:
            explanation_parts.append("No rules could be triggered. The initial facts did not satisfy the prerequisites for any active theorems in the domain.")
        else:
            explanation_parts.append("The solver triggered the following rules but could not bridge the logical gap to reach the goal:")
            for idx, step in enumerate(request.execution_trace):
                explanation_parts.append(f"### Step {idx+1}: Activating {step.rule_id}")
                explanation_parts.append(f"* **Applied Rule**: `{step.fired_rule_repr}`")
                explanation_parts.append(f"* **New Deductions**: `{', '.join(step.new_facts)}`")
                explanation_parts.append("")

        explanation_parts.append("## 3. Conclusion")
        explanation_parts.append("⚠️ **Logical Gap Detected**: The target goal could not be proved. This is either because the starting facts are logically independent from the goal, or the registered rule database is missing the necessary connecting theorems.")

    explanation = "\n".join(explanation_parts)
    return ExplainResponse(
        explanation=explanation,
        structured=False
    )


@app.post("/api/explain/stream", tags=["Explainability Agent"])
async def explain_proof_stream(request: ExplainRequest):
    """
    Explainability Agent (Streaming): Streams a rich educational explanation chunk-by-chunk.
    """
    domain = request.domain.lower()

    # Formulate a dry trace text summary
    trace_text = ""
    for idx, step in enumerate(request.execution_trace):
        trace_text += f"Step {idx+1}: Fired Rule [{step.rule_id}] -> {step.fired_rule_repr}. New facts deduced: {step.new_facts}\n"

    # LLM Pathway
    from rag_agent.llm_factory import get_llm
    llm = get_llm(temperature=0.3)
    if llm:
        try:
            from langchain_core.prompts import ChatPromptTemplate

            if request.goal_reached:
                system_prompt = (
                    "You are a Lead AI Architect and expert Neuro-Symbolic educational tutor. "
                    "Your goal is to explain a successful logical proof path found by our symbolic solver "
                    "in the domain '{domain}'. Make the explanation extremely engaging, "
                    "pedagogically rich, and easy to read. "
                    "Explain the chemical reactions, geometry axioms, or algebra theorems "
                    "so that a high-school student can understand them perfectly. "
                    "Use professional formatting, bullet points, and headers.\n"
                    "Strict Constraint: You MUST ground your explanation ONLY in the facts "
                    "and steps presented in the logical trace. Do not invent any new steps or hallucinate rules!"
                )
                human_prompt = "Explain this reasoning pathway for the question: '{query}'\n\nSolver Execution Trace:\n{trace}"
            else:
                system_prompt = (
                    "You are a Lead AI Architect and expert Neuro-Symbolic educational tutor. "
                    "Our symbolic solver attempted to prove the goal in the domain '{domain}' "
                    "but FAILED because there is no valid proof path using the available rules and facts.\n"
                    "Your goal is to explain this failure to a high-school student in an extremely engaging, "
                    "pedagogically rich, and constructive manner. "
                    "Identify what starting facts were mapped, what intermediate facts (if any) were deduced "
                    "in the trace, and explain clearly and mathematically why the goal could not be reached "
                    "(e.g., because the assumptions are not sufficient to prove the claim, or the registered rule database is missing the necessary theorems).\n"
                    "Use professional formatting, bullet points, and headers.\n"
                    "Strict Constraint: Do not hallucinate or claim that the goal was proven or reached! "
                    "Always start by clearly stating that the goal is UNPROVED based on the available knowledge base."
                )
                human_prompt = "Pedagogically analyze the logical failure for this query: '{query}'\n\nSolver Execution Trace:\n{trace}"

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])

            logger.info("Streaming proof explanation via model-agnostic LLM instance (Goal Reached: %s)...", request.goal_reached)
            
            async def generate_chunks():
                chain = prompt | llm
                async for chunk in chain.astream({
                    "domain": domain,
                    "query": request.query,
                    "trace": trace_text if trace_text else "No rules were triggered."
                }):
                    content = chunk.content
                    if not content:
                        continue
                    
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, str):
                                text_parts.append(part)
                            elif isinstance(part, dict) and "text" in part:
                                text_parts.append(part["text"])
                            elif hasattr(part, "text"):
                                text_parts.append(getattr(part, "text"))
                        yield "".join(text_parts)
                    else:
                        yield str(content)

            return StreamingResponse(generate_chunks(), media_type="text/plain")

        except Exception as e:
            logger.warning("Streaming LLM explanation failed: %s. Using template fallback.", e)

    # Offline Fallback Template-based Explainer (Yields in chunks to simulate stream)
    logger.info("Generating template-based offline proof explanation in streaming mode.")
    
    explanation_parts = []
    if request.goal_reached:
        explanation_parts = [
            f"# Educational Proof Explanation (Offline Mode)\n\n",
            f"**Original Question**: *\"{request.query}\"*\n\n",
            f"The symbolic core engine has successfully executed a formal deduction trace. Below is the step-by-step breakdown of how the goal was reached:\n\n",
            f"## 1. Initial State\n",
            f"We started with the concepts and assertions mapped from the query. These served as our starting points for logical chaining.\n\n",
            f"## 2. Deduction Steps\n"
        ]

        if not request.execution_trace:
            explanation_parts.append("No rules were triggered. The goal was already satisfied by the initial facts or could not be proved.\n")
        else:
            for idx, step in enumerate(request.execution_trace):
                explanation_parts.append(f"### Step {idx+1}: Activating {step.rule_id}\n")
                explanation_parts.append(f"* **Applied Rule**: `{step.fired_rule_repr}`\n")
                explanation_parts.append(f"* **New Deductions**: `{', '.join(step.new_facts)}`\n")
                
                # Domain specific educational descriptions
                if domain == "chemistry":
                    if "na_h2o" in step.rule_id or "r1" in step.rule_id:
                        explanation_parts.append("  * **Educational context**: Sodium (`Na`) is a highly reactive alkali metal. When it comes into contact with water (`H2O`), it undergoes a vigorous oxidation-reduction reaction to synthesize sodium hydroxide (`NaOH`) and release highly flammable hydrogen gas (`H2`).\n")
                    elif "naoh_hcl" in step.rule_id or "r3" in step.rule_id:
                        explanation_parts.append("  * **Educational context**: This is a classic neutralization reaction. The strong base sodium hydroxide (`NaOH`) reacts with the strong acid hydrochloric acid (`HCl`) to form water (`H2O`) and table salt, sodium chloride (`NaCl`), in a standard double-displacement reaction.\n")
                    else:
                        explanation_parts.append("  * **Educational context**: A chemical reaction took place combining the reactants to synthesize the products according to standard stoichiometric ratios.\n")
                elif domain == "geometry":
                    if "trans" in step.rule_id or "t_trans" in step.rule_id:
                        explanation_parts.append("  * **Educational context**: By Euclid's first common notion (Things which equal the same thing also equal one another), congruence is transitive. Since Segment 1 is congruent to Segment 2, and Segment 2 is congruent to Segment 3, Segment 1 is geometrically congruent to Segment 3.\n")
                    else:
                        explanation_parts.append("  * **Educational context**: A geometric theorem or axiom was applied to prove the congruence or relation between the geometric entities.\n")
                else:
                    explanation_parts.append("  * **Educational context**: Applied algebraic equivalence rules to transform the equations and isolate variables.\n")
                explanation_parts.append("\n")

        explanation_parts.append("## 3. Conclusion\n")
        explanation_parts.append("By executing the dry logical proof sequence above, the goal has been successfully established and verified to be 100% mathematically and scientifically correct!\n")
    else:
        explanation_parts = [
            f"# Educational Proof Analysis (Goal Unproved - Offline Mode)\n\n",
            f"**Original Question**: *\"{request.query}\"*\n\n",
            f"The symbolic core engine attempted to deduce the target goal using forward-chaining rules but was **unable to prove it** from the given assumptions.\n\n",
            f"## 1. Initial State & Mapping\n",
            f"We started with the concepts and assertions mapped from the query. These served as our initial set of facts.\n\n",
            f"## 2. Saturation & Attempted Deductions\n"
        ]

        if not request.execution_trace:
            explanation_parts.append("No rules could be triggered. The initial facts did not satisfy the prerequisites for any active theorems in the domain.\n")
        else:
            explanation_parts.append("The solver triggered the following rules but could not bridge the logical gap to reach the goal:\n\n")
            for idx, step in enumerate(request.execution_trace):
                explanation_parts.append(f"### Step {idx+1}: Activating {step.rule_id}\n")
                explanation_parts.append(f"* **Applied Rule**: `{step.fired_rule_repr}`\n")
                explanation_parts.append(f"* **New Deductions**: `{', '.join(step.new_facts)}`\n")
                explanation_parts.append("\n")

        explanation_parts.append("## 3. Conclusion\n")
        explanation_parts.append("⚠️ **Logical Gap Detected**: The target goal could not be proved. This is either because the starting facts are logically independent from the goal, or the registered rule database is missing the necessary connecting theorems.\n")

    async def generate_template_chunks():
        for chunk in explanation_parts:
            yield chunk
            await asyncio.sleep(0.05)

    return StreamingResponse(generate_template_chunks(), media_type="text/plain")


@app.get("/rules", tags=["Knowledge Graph"])
@app.get("/api/rules", tags=["Knowledge Graph"])
async def get_rules(domain: Optional[str] = Query(None, description="Filter rules by domain")):
    """Get all rules currently registered in the Neo4j Knowledge Graph."""
    db_conn = Neo4jConnection()
    if not db_conn.verify_connectivity():
        db_conn.close()
        raise HTTPException(status_code=503, detail="Neo4j Knowledge Graph is unreachable.")
        
    try:
        with db_conn.get_session() as session:
            query = "MATCH (r:Rule) "
            params = {}
            if domain:
                query += "WHERE r.domain = $domain "
                params["domain"] = domain.lower()
            query += "RETURN r.id AS id, r.name AS name, r.inputs AS inputs, r.outputs AS outputs, r.domain AS domain, r.description AS description"
            
            result = session.run(query, **params)
            rules = []
            for record in result:
                rules.append({
                    "id": record["id"],
                    "name": record["name"],
                    "domain": record["domain"],
                    "inputs": record["inputs"],
                    "outputs": record["outputs"],
                    "description": record["description"]
                })
            return {"count": len(rules), "rules": rules}
    finally:
        db_conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
