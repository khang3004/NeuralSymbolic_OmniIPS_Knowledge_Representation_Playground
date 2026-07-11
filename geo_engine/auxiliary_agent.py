"""
GeoIPS — Auxiliary Construction Agent (AlphaGeometry-inspired).

When the symbolic solver gets stuck (cannot prove the goal from current facts),
this agent calls an LLM to suggest useful geometric auxiliary constructions —
new points, lines, or segments that might bridge the logical gap.

The suggestions are added to Working Memory, and the solver retries.
This mirrors AlphaGeometry's neural beam search, but uses LLM prompting instead.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("aux_construction_agent")

# Common auxiliary construction patterns for the LLM to reference
CONSTRUCTION_EXAMPLES = """
Common geometric auxiliary constructions:
1. Draw altitude from vertex A to side BC → Foot(H,A,BC), Perpendicular(AH,BC), RightAngle(Angle(AHB))
2. Draw median from vertex A to midpoint M of BC → Midpoint(M,BC), Segment(AM)
3. Draw angle bisector from vertex A → AngleBisector(AD,Angle(BAC)), Equal(Angle(BAD),Angle(CAD))
4. Extend a side: extend BC beyond C to point D → Collinear(B,C,D), ExteriorAngle(ACD,C)
5. Draw a parallel line through a point: through P parallel to AB → Parallel(PQ,AB)
6. Draw a circumscribed circle → Circle(O), PointOnCircle(A,Circle(O)), PointOnCircle(B,Circle(O))
7. Connect midpoints of two sides → Midpoint(M,AB), Midpoint(N,AC), Segment(MN)
8. Drop perpendicular from external point to line → Perpendicular(PQ,Line(AB)), Foot(Q,P,AB)
"""


class AuxiliaryConstructionAgent:
    """
    AlphaGeometry-inspired auxiliary construction agent.

    Given the current proof state (facts + failed goal), suggests new
    geometric objects (points, lines, circles) to add to Working Memory
    so the symbolic solver can find a proof path.
    """

    def __init__(self, llm):
        """
        Args:
            llm: A LangChain chat model instance (from llm_factory.get_llm()).
        """
        self.llm = llm

    async def suggest_constructions(
        self,
        current_facts: List[str],
        goal: str,
        failed_steps: List[str],
        max_suggestions: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Ask the LLM to suggest auxiliary geometric constructions.

        Args:
            current_facts: List of fact strings currently in Working Memory.
            goal: The target fact string that could not be proved.
            failed_steps: List of rule representations that were tried.
            max_suggestions: Max number of constructions to return.

        Returns:
            List of dicts, each with:
            {
                "description": "Human-readable description of the construction",
                "new_facts": ["Fact1(...)", "Fact2(...)", ...]
            }
        """
        facts_str = "\n".join(f"  - {f}" for f in current_facts)
        failed_str = "\n".join(f"  - {s}" for s in failed_steps) if failed_steps else "  (none)"

        prompt_text = f"""You are a plane geometry expert helping a symbolic theorem prover (GeoIPS).

The solver has these KNOWN FACTS:
{facts_str}

The solver is trying to PROVE: {goal}

The following rules were tried but the goal remains unproved:
{failed_str}

{CONSTRUCTION_EXAMPLES}

Suggest up to {max_suggestions} auxiliary geometric constructions that might help prove the goal.
For each suggestion:
1. Briefly describe the construction in plain language.
2. List the new formal predicates it adds to the working memory (using the same predicate syntax).

Respond in this EXACT JSON format (array of objects):
[
  {{
    "description": "Draw the altitude from A to BC",
    "new_facts": ["Foot(H,A,BC)", "Perpendicular(AH,BC)", "RightAngle(Angle(AHB))", "RightAngle(Angle(AHC))"]
  }},
  ...
]

Only suggest constructions that are logically valid given the known facts.
Output ONLY the JSON array, no extra text."""

        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm.ainvoke([HumanMessage(content=prompt_text)])

            content = response.content
            if isinstance(content, list):
                content = "".join(
                    p if isinstance(p, str) else p.get("text", "") for p in content
                )

            # Extract JSON from response
            import json
            import re

            # Try to find JSON array in the response
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group())
            else:
                suggestions = json.loads(content.strip())

            # Validate structure
            valid_suggestions = []
            for s in suggestions:
                if isinstance(s, dict) and "description" in s and "new_facts" in s:
                    if isinstance(s["new_facts"], list):
                        valid_suggestions.append({
                            "description": str(s["description"]),
                            "new_facts": [str(f) for f in s["new_facts"]]
                        })

            logger.info(
                "Auxiliary agent suggested %d constructions for goal: %s",
                len(valid_suggestions), goal
            )
            return valid_suggestions[:max_suggestions]

        except Exception as e:
            logger.error("Auxiliary construction agent failed: %s", e)
            return []

    async def suggest_single(
        self,
        current_facts: List[str],
        goal: str,
    ) -> Optional[Dict[str, Any]]:
        """Convenience: return just the first suggestion."""
        results = await self.suggest_constructions(current_facts, goal, failed_steps=[], max_suggestions=1)
        return results[0] if results else None
