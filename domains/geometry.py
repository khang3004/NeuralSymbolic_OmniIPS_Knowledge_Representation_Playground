import re
from domains.base import DomainParser
from core_engine.models import Fact, Rule

class GeometryParser(DomainParser):
    """
    Concrete syntax parser for Plane Geometry.
    Parses geometric assertions such as Congruent(AB, CD) or Similar(ABC, DEF)
    and handles relation/arguments canonical forms.
    """
    @property
    def domain_name(self) -> str:
        return "geometry"

    def parse_fact(self, raw_input: str, fact_id: str) -> Fact:
        from core_engine.arithmetic_evaluator import canonicalize
        canonical_val = canonicalize(raw_input)
        
        first_paren = canonical_val.find("(")
        if first_paren != -1 and canonical_val.endswith(")"):
            relation = canonical_val[:first_paren]
            inner_content = canonical_val[first_paren+1:-1]
            
            args = []
            current_arg = []
            paren_level = 0
            for char in inner_content:
                if char == "," and paren_level == 0:
                    args.append("".join(current_arg).strip())
                    current_arg = []
                else:
                    if char == "(":
                        paren_level += 1
                    elif char == ")":
                        paren_level -= 1
                    current_arg.append(char)
            if current_arg:
                args.append("".join(current_arg).strip())
        else:
            relation = "Atom"
            args = [canonical_val]

        return Fact(
            id=fact_id,
            value=canonical_val,
            domain=self.domain_name,
            attributes={"relation": relation, "args": args}
        )


    def parse_rule(self, raw_rule: dict) -> Rule:
        rule_id = raw_rule["id"]
        name = raw_rule.get("name", f"Geometric Theorem {rule_id}")
        description = raw_rule.get("description", f"Theorem: If {' and '.join(raw_rule['inputs'])} then {' and '.join(raw_rule['outputs'])}")

        antecedents = [
            self.parse_fact(ant, f"{rule_id}_ant_{idx}")
            for idx, ant in enumerate(raw_rule["inputs"])
        ]
        consequents = [
            self.parse_fact(cons, f"{rule_id}_cons_{idx}")
            for idx, cons in enumerate(raw_rule["outputs"])
        ]

        return Rule(
            id=rule_id,
            name=name,
            domain=self.domain_name,
            antecedents=antecedents,
            consequents=consequents,
            description=description
        )

    def format_fact(self, fact: Fact) -> str:
        return fact.value
