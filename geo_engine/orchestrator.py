"""
GeoIPS — Multi-Agent Geo-Orchestrator.

Implements the complete AlphaGeometry-inspired multi-agent loop:
1. Geo-Parser Agent: NL Query → Structured Facts & Goal.
2. Symbolic & AR Engine: Forward Chaining + Algebraic Reasoning.
3. Diagnostic Reasoner Agent: Analyzes proof gap when solver hits saturation.
4. Auxiliary Construction Agent: Suggests point/line constructions when stuck.
5. Dynamic Rule Synthesizer Agent: Generates missing Datalog rules, validates syntax,
   hot-injects into engine, and persists to Neo4j.

STRICTLY DYNAMIC — NO HARDCODING.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

from core_engine.models import Fact, Rule, InferenceResult
from core_engine.solver import ForwardChainingEngine
from domains.geometry import GeometryParser

from geo_engine.auxiliary_agent import AuxiliaryConstructionAgent
from geo_engine.rule_synthesizer import DynamicRuleSynthesizerAgent
from graph_db.connection import Neo4jConnection
from rag_agent.router import route_query
from rag_agent.llm_factory import get_llm

logger = logging.getLogger("multi_agent_orchestrator")
PARSER = GeometryParser()


class MultiAgentGeoOrchestrator:
    """
    Master Multi-Agent Orchestrator for AlphaGeometry-grade Neuro-Symbolic Reasoning.
    """

    def __init__(self, db_conn: Optional[Neo4jConnection] = None):
        self.db_conn = db_conn

    async def solve_query(
        self,
        query: str,
        max_iterations: int = 4,
    ) -> Tuple[InferenceResult, List[Fact], Fact, List[str], List[str]]:
        """
        Run the complete multi-agent loop over a natural language query.

        Args:
            query: Natural language geometry problem statement.
            max_iterations: Maximum loop iterations (Aux + Rule Synthesis passes).

        Returns:
            (result, initial_facts, goal_fact, auxiliary_constructions, synthesized_rule_names)
        """
        logger.info("[Orchestrator] Starting multi-agent solve for query: '%s'", query)

        # 1. Geo-Parser Agent: Parse text → formal predicates
        initial_facts, goal_fact = route_query(query)

        if not initial_facts:
            raise ValueError("Could not extract any formal initial facts from query.")

        # 2. Fetch base rules from Neo4j DB
        base_rules = self._load_base_rules()
        active_rules = list(base_rules)

        current_facts = list(initial_facts)
        all_constructions: List[str] = []
        synthesized_rule_names: List[str] = []

        llm = get_llm(temperature=0.3)
        aux_agent = AuxiliaryConstructionAgent(llm) if llm else None
        rule_synth_agent = DynamicRuleSynthesizerAgent(llm) if llm else None

        final_result = None

        for iteration in range(max_iterations + 1):
            logger.info("[Orchestrator] Iteration %d/%d — Running Symbolic & AR Engine with %d rules and %d facts",
                        iteration, max_iterations, len(active_rules), len(current_facts))

            # Run Symbolic Engine + Algebraic Reasoning
            engine = ForwardChainingEngine(active_rules)
            final_result = engine.solve(current_facts, goal_fact)

            if final_result.goal_reached:
                logger.info("[Orchestrator] SUCCESS: Goal '%s' proved at iteration %d!", goal_fact.value, iteration)
                break

            if iteration == max_iterations:
                logger.info("[Orchestrator] Reached max iterations (%d). Goal not proved.", max_iterations)
                break

            # If no LLM available, cannot perform aux construction or rule synthesis
            if not llm:
                logger.warning("[Orchestrator] No LLM available for dynamic synthesis. Stopping.")
                break

            # 3. Diagnostic & Dynamic Synthesis Loop
            logger.info("[Orchestrator] Goal unproved. Triggering Diagnostic Reasoner & Synthesizer Agents...")

            # --- A. Dynamic Rule Synthesizer Pass ---
            new_rules = await rule_synth_agent.synthesize_missing_rules(
                current_facts=[f.value for f in current_facts],
                goal=goal_fact.value,
                failed_steps=[s.fired_rule_repr for s in final_result.execution_trace],
                max_rules=2,
            )

            rules_added = False
            for nr in new_rules:
                if nr not in active_rules and nr.id not in [r.id for r in active_rules]:
                    active_rules.append(nr)
                    synthesized_rule_names.append(nr.name)
                    rules_added = True
                    logger.info("[Orchestrator] Hot-injected synthesized rule: %s [%s]", nr.name, nr.id)
                    # Persist to Neo4j
                    self._persist_rule_to_neo4j(nr)

            # --- B. Auxiliary Construction Pass ---
            aux_added = False
            if aux_agent:
                suggestions = await aux_agent.suggest_constructions(
                    current_facts=[f.value for f in current_facts],
                    goal=goal_fact.value,
                    failed_steps=[s.fired_rule_repr for s in final_result.execution_trace],
                    max_suggestions=2,
                )

                for suggestion in suggestions:
                    for nf_str in suggestion.get("new_facts", []):
                        all_constructions.append(nf_str)
                        nf_obj = PARSER.parse_fact(nf_str, f"aux_{len(all_constructions)}")
                        if nf_obj not in current_facts:
                            current_facts.append(nf_obj)
                            aux_added = True
                            logger.info("[Orchestrator] Added auxiliary construction fact: %s", nf_str)

            # If neither new rules nor new facts were added, we've hit absolute saturation
            if not rules_added and not aux_added:
                logger.info("[Orchestrator] No new rules or facts generated by agents. Stopping loop.")
                break

        return final_result, initial_facts, goal_fact, all_constructions, synthesized_rule_names

    def _load_base_rules(self) -> List[Rule]:
        """Fetch base rule set from Neo4j Graph DB (or fallback)."""
        from api.main import _get_rules
        conn = self.db_conn or Neo4jConnection()
        try:
            rules = _get_rules(conn)
            return rules
        finally:
            if not self.db_conn:
                conn.close()

    def _persist_rule_to_neo4j(self, rule: Rule):
        """Persist a dynamically synthesized rule into Neo4j for future queries."""
        conn = self.db_conn or Neo4jConnection()
        try:
            query = """
            MERGE (r:Rule {id: $id})
            SET r.name = $name,
                r.domain = $domain,
                r.inputs = $inputs,
                r.outputs = $outputs,
                r.description = $description,
                r.synthesized = true
            """
            inputs_str = [f.value for f in rule.inputs]
            outputs_str = [f.value for f in rule.outputs]
            conn.query(query, {
                "id": rule.id,
                "name": rule.name,
                "domain": rule.domain,
                "inputs": inputs_str,
                "outputs": outputs_str,
                "description": rule.description,
            })
            logger.info("[Orchestrator] Persisted synthesized rule '%s' into Neo4j DB.", rule.id)
        except Exception as e:
            logger.warning("Could not persist rule to Neo4j: %s", e)
        finally:
            if not self.db_conn:
                conn.close()
