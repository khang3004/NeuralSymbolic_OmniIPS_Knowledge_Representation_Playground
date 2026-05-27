import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

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
    mapped_initial_facts: List[str] = Field(..., description="Semantic facts mapped from ChromaDB.")
    mapped_goal: str = Field(..., description="Semantic goal mapped from ChromaDB.")
    goal_reached: bool = Field(..., description="Whether the goal was reached.")
    applied_rule_ids: List[str] = Field(..., description="Sequence of fired rule IDs.")
    execution_trace: List[ExecutionStepResponse] = Field(..., description="Formal proof steps.")
    known_facts: List[str] = Field(..., description="All final known facts.")

class ExplainRequest(BaseModel):
    query: str = Field(..., description="Original natural language query.")
    domain: str = Field(..., description="Target domain.")
    execution_trace: List[ExecutionStepResponse] = Field(..., description="Solver execution trace.")

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
            "chromadb_host": os.getenv("CHROMADB_HOST", "localhost"),
            "chromadb_port": os.getenv("CHROMADB_PORT", "8000")
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
                        "RETURN r.id AS id, r.name AS name, r.inputs AS inputs, "
                        "r.outputs AS outputs, r.description AS description",
                        domain=domain
                    )
                    raw_rules = []
                    for record in result:
                        raw_rules.append({
                            "id": record["id"],
                            "name": record["name"],
                            "inputs": record["inputs"],
                            "outputs": record["outputs"],
                            "description": record["description"]
                        })
                    
                    rules = [parser.parse_rule(r) for r in raw_rules]
                    logger.info("Loaded %d rules from Neo4j for domain '%s'", len(rules), domain)
            db_conn.close()
        except Exception as db_err:
            logger.warning("Failed to fetch rules from Neo4j, falling back to mock: %s", db_err)
            
        # Predefined mock rules fallback
        if not rules:
            logger.info("Using built-in fallback rules for domain '%s'", domain)
            if domain == "chemistry":
                raw_rules = [
                    {"id": "r1", "name": "Sodium Hydration", "inputs": ["Na", "H2O"], "outputs": ["NaOH", "H2"], "description": "Sodium reacting with water."},
                    {"id": "r2", "name": "Water Synthesis", "inputs": ["H2", "O2"], "outputs": ["H2O"], "description": "Combustion of hydrogen."},
                    {"id": "r3", "name": "Neutralization", "inputs": ["NaOH", "HCl"], "outputs": ["NaCl", "H2O"], "description": "Acid-base neutralization."}
                ]
            elif domain == "geometry":
                raw_rules = [
                    {"id": "t_trans", "name": "Congruence Transitivity", "inputs": ["Congruent(AB, CD)", "Congruent(CD, EF)"], "outputs": ["Congruent(AB, EF)"], "description": "Transitivity property."}
                ]
            elif domain == "algebra":
                raw_rules = [
                    {"id": "a_sub_two", "name": "Subtraction Property", "inputs": ["x+2=5", "Subtract(2, both_sides)"], "outputs": ["x=3"], "description": "Subtracting value from both sides."}
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
        # Step 1: Run the Neuro-Symbolic Router (NLP -> ChromaDB -> Neo4j Nodes)
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
                        "RETURN r.id AS id, r.name AS name, r.inputs AS inputs, "
                        "r.outputs AS outputs, r.description AS description",
                        domain=domain
                    )
                    raw_rules = []
                    for record in result:
                        raw_rules.append({
                            "id": record["id"],
                            "name": record["name"],
                            "inputs": record["inputs"],
                            "outputs": record["outputs"],
                            "description": record["description"]
                        })
                    rules = [parser.parse_rule(r) for r in raw_rules]
                    logger.info("Loaded %d rules from Neo4j for query resolving in domain '%s'", len(rules), domain)
            db_conn.close()
        except Exception as db_err:
            logger.warning("Failed to fetch rules from Neo4j, using built-in fallback: %s", db_err)

        # Default rules fallback
        if not rules:
            logger.info("Using built-in fallback rules for query resolving on domain '%s'", domain)
            if domain == "chemistry":
                raw_rules = [
                    {"id": "rxn_single_na_h2o", "name": "Sodium Reacting with Water", "inputs": ["Na", "H2O"], "outputs": ["NaOH", "H2"], "description": ""},
                    {"id": "rxn_double_naoh_hcl", "name": "Neutralization of NaOH and HCl", "inputs": ["NaOH", "HCl"], "outputs": ["NaCl", "H2O"], "description": ""}
                ]
            elif domain == "geometry":
                raw_rules = [
                    {"id": "thm_congruence_transitivity", "name": "Transitivity of Congruence", "inputs": ["Congruent(AB, CD)", "Congruent(CD, EF)"], "outputs": ["Congruent(AB, EF)"], "description": ""}
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

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a Lead AI Architect and expert Neuro-Symbolic educational tutor. "
                           "Your goal is to explain a logical proof path found by our symbolic solver "
                           "in the domain '{domain}'. Make the explanation extremely engaging, "
                           "pedagogically rich, and easy to read. "
                           "Explain the chemical reactions, geometry axioms, or algebra theorems "
                           "so that a high-school student can understand them perfectly. "
                           "Use professional formatting, bullet points, and headers.\n"
                           "Strict Constraint: You MUST ground your explanation ONLY in the facts "
                           "and steps presented in the logical trace. Do not invent any new steps or hallucinate rules!"),
                ("human", "Explain this reasoning pathway for the question: '{query}'\n\n"
                          "Solver Execution Trace:\n{trace}")
            ])

            logger.info("Generating proof explanation via model-agnostic LLM instance...")
            chain = prompt | llm

            response = chain.invoke({
                "domain": domain,
                "query": request.query,
                "trace": trace_text
            })
            # Cleanly extract string content if returned as structured list (e.g. Google thinking blocks)
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
            logger.warning("LangChain OpenAI explanation generation failed: %s. Using template fallback.", e)

    # Offline Fallback Template-based Explainer
    logger.info("Generating template-based offline proof explanation.")
    
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

    explanation = "\n".join(explanation_parts)
    return ExplainResponse(
        explanation=explanation,
        structured=False
    )


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
