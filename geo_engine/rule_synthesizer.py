"""
GeoIPS — Dynamic Rule Synthesizer Agent.

When the symbolic solver hits saturation (cannot prove the goal from current facts/rules),
this agent asks an LLM to identify missing geometric theorems or domain rules,
synthesizes them into formal Datalog Rule objects with variable bindings (?A, ?B, ?C),
validates their syntax, and hot-injects them into Working Memory and Neo4j.

STRICTLY DYNAMIC — NO HARDCODING.
"""

import re
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional

from core_engine.models import Rule, Fact
from domains.geometry import GeometryParser

logger = logging.getLogger("dynamic_rule_synthesizer")
PARSER = GeometryParser()


class DynamicRuleSynthesizerAgent:
    """
    LLM-powered agent that synthesizes executable Datalog rules dynamically
    when the proof engine encounters an unproved goal.
    """

    def __init__(self, llm):
        """
        Args:
            llm: A LangChain chat model instance (from llm_factory.get_llm()).
        """
        self.llm = llm

    async def synthesize_missing_rules(
        self,
        current_facts: List[str],
        goal: str,
        failed_steps: List[str],
        max_rules: int = 3,
    ) -> List[Rule]:
        """
        Analyze the proof state, identify missing geometric theorems,
        and generate valid formal Datalog Rule objects.

        Args:
            current_facts: List of fact strings currently in Working Memory.
            goal: Target predicate string to prove.
            failed_steps: List of rule representations that were executed.
            max_rules: Maximum number of rules to synthesize in one step.

        Returns:
            List of validated Rule objects ready for hot-injection into the solver.
        """
        facts_str = "\n".join(f"  - {f}" for f in current_facts[:30])
        failed_str = "\n".join(f"  - {s}" for s in failed_steps[-10:]) if failed_steps else "  (none)"

        prompt_text = f"""You are a formal logic and plane geometry expert for a Neuro-Symbolic Reasoning Engine (GeoIPS).

The symbolic solver has reached SATURATION and cannot prove the GOAL using its current rule database.

KNOWN FACTS IN WORKING MEMORY:
{facts_str}

GOAL TO PROVE:
{goal}

RULES PREVIOUSLY EXECUTED:
{failed_str}

=== SYNTHESIS TASK ===
Identify missing geometric theorems, definitions, or properties needed to bridge the proof gap from KNOWN FACTS to GOAL.
Synthesize up to {max_rules} formal Datalog rules using variables (prefixed with ? like ?A, ?B, ?C, ?O, ?M, ?H).

=== PREDICATE & VARIABLE SYNTAX RULES ===
1. Variable notation: Use ? prefixed uppercase letters for points/entities, e.g. ?A, ?B, ?C, ?O, ?M, ?H, ?D, ?E, ?F.
2. Segment concatenation: Represent line segments as ?A?B or ?B?C.
3. Angles: Angle(?B?A?C) or Angle(?A).
4. Distance/Length: Length(?A?B) or ?A?B.
5. Common Predicates:
   - Triangle(?A,?B,?C)
   - RightTriangle(?A,?B,?C)
   - IsoscelesTriangle(?A,?B,?C)
   - CyclicQuadrilateral(?A,?B,?C,?D)
   - Square(?A,?B,?C,?D)
   - Circle(?O) or Circumcircle(?O,?A,?B,?C)
   - PointOnCircle(?P,Circle(?O))
   - Midpoint(?M,?B?C) or Midpoint(?M,?B,?C)
   - Foot(?H,?A,?B?C) or Foot(?H,?A,?B,?C)
   - Parallel(?A?B,?C?D)
   - Perpendicular(?A?B,?C?D)
   - Equal(Angle(?A),Angle(?B)) or Equal(?X,?Y)
   - Congruent(?A?B,?C?D)
   - SimilarTriangles(?A?B?C,?D?E?F)
   - CongruentTriangles(?A?B?C,?D?E?F)
   - Collinear(?A,?B,?C)
   - Concurrent(?A?D,?B?E,?C?F)

=== OUTPUT REQUIREMENTS ===
Respond ONLY with a JSON array of rule objects in this EXACT format:
[
  {{
    "name": "Triangle Midsegment Theorem",
    "inputs": ["Triangle(?A,?B,?C)", "Midpoint(?D,?A,?B)", "Midpoint(?E,?A,?C)"],
    "outputs": ["Parallel(?D?E,?B?C)", "Equal(Length(?D?E),Mul(Div(1,2),Length(?B?C)))"],
    "description": "The segment joining midpoints of two sides is parallel to third side and half its length."
  }}
]

Do NOT include any commentary outside the JSON array."""

        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm.ainvoke([HumanMessage(content=prompt_text)])

            content = response.content
            if isinstance(content, list):
                content = "".join(p if isinstance(p, str) else p.get("text", "") for p in content)

            rule_dicts = self._parse_json_rules(content)
            validated_rules: List[Rule] = []

            for rd in rule_dicts[:max_rules]:
                rule_obj = self._build_and_validate_rule(rd)
                if rule_obj:
                    validated_rules.append(rule_obj)
                    logger.info("Successfully synthesized & validated rule: %s [%s]", rule_obj.name, rule_obj.id)

            return validated_rules

        except Exception as e:
            logger.error("DynamicRuleSynthesizerAgent failed: %s", e)
            return []

    def _parse_json_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract and parse JSON array from raw LLM output text."""
        try:
            # Match JSON array pattern
            match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(text.strip())
        except Exception as e:
            logger.warning("Failed to parse JSON rules from LLM output: %s", e)
            return []

    def _build_and_validate_rule(self, rd: Dict[str, Any]) -> Optional[Rule]:
        """
        Validate structural integrity of a rule dictionary and construct a formal Rule model.
        """
        if not isinstance(rd, dict):
            return None

        inputs = rd.get("inputs")
        outputs = rd.get("outputs")
        name = str(rd.get("name", "Synthesized Rule"))
        description = str(rd.get("description", name))

        if not isinstance(inputs, list) or not isinstance(outputs, list):
            return None
        if not inputs or not outputs:
            return None

        # Clean string predicates
        clean_inputs = [str(i).strip() for i in inputs if str(i).strip()]
        clean_outputs = [str(o).strip() for o in outputs if str(o).strip()]

        if not clean_inputs or not clean_outputs:
            return None

        # Hash-based rule ID for deterministic uniqueness
        rule_hash = hashlib.md5(f"{name}:{','.join(clean_inputs)}->{','.join(clean_outputs)}".encode()).hexdigest()[:10]
        rule_id = f"geo_dyn_rule_{rule_hash}"

        # Construct raw rule dict for RuleParser
        raw_rule = {
            "id": rule_id,
            "name": name,
            "inputs": clean_inputs,
            "outputs": clean_outputs,
            "description": description,
        }

        try:
            parsed_rule = PARSER.parse_rule(raw_rule)
            return parsed_rule
        except Exception as e:
            logger.warning("Rule validation failed for '%s': %s", name, e)
            return None
