from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Fact(BaseModel):
    """
    Strictly-typed schema representing a single geometric assertion or predicate.
    """
    id: str = Field(..., description="Unique identifier for the fact")
    value: str = Field(..., description="Raw string value of the assertion, e.g., 'Congruent(AB, CD)'")
    domain: str = Field("geometry", description="Logical domain name (default: geometry)")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary for geometric attributes")

    def __hash__(self) -> int:
        return hash((self.value, self.domain))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Fact):
            return False
        return self.value == other.value and self.domain == other.domain


class Rule(BaseModel):
    """
    Strictly-typed schema representing a symbolic geometric production rule (antecedents -> consequents).
    """
    id: str = Field(..., description="Unique rule identifier, e.g., 'geo_sas_congruence'")
    name: str = Field(..., description="Human-readable rule name, e.g., 'SAS Triangle Congruence'")
    domain: str = Field("geometry", description="Logical domain name (default: geometry)")
    antecedents: List[Fact] = Field(..., description="Preconditions required for the rule to fire")
    consequents: List[Fact] = Field(..., description="Conclusions generated when the rule fires")
    description: Optional[str] = Field(None, description="Detailed explanation of the geometric theorem")

    def __repr__(self) -> str:
        ants = " + ".join([f.value for f in self.antecedents])
        cons = " + ".join([f.value for f in self.consequents])
        return f"{self.id} [{self.name}]: {ants} -> {cons}"


class ExecutionStep(BaseModel):
    """
    Records an individual execution step of the logic solver for post-hoc Explainability.
    """
    rule_id: str = Field(..., description="ID of the fired rule")
    fired_rule_repr: str = Field(..., description="String representation of the fired rule")
    new_facts: List[Fact] = Field(..., description="Facts added to Working Memory by this step")
    timestamp_ms: float = Field(..., description="Timestamp of when the rule was fired")


class InferenceResult(BaseModel):
    """
    Comprehensive execution summary of the solver execution.
    """
    goal_reached: Optional[bool] = Field(None, description="True if goal is satisfied, False if not, None if no goal set")
    final_facts: List[Fact] = Field(..., description="Final state of the working memory facts")
    execution_trace: List[ExecutionStep] = Field(..., description="Sequenced path of rule activations")
    applied_rule_ids: List[str] = Field(..., description="Sequenced list of triggered rule IDs")
