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
        cleaned = raw_input.replace(" ", "")
        
        # Safe extraction of outer relation and balanced inner content
        first_paren = cleaned.find("(")
        if first_paren != -1 and cleaned.endswith(")"):
            relation = cleaned[:first_paren]
            inner_content = cleaned[first_paren+1:-1]
            
            # Split by comma only at the top-level (level 0) of parenthesis nesting
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
                
            # For commutative geometric relations, sort arguments
            commutative_relations = {"Congruent", "Similar", "Parallel", "Intersect"}
            if relation in commutative_relations:
                sorted_args = sorted(args)
                canonical_val = f"{relation}({','.join(sorted_args)})"
            else:
                canonical_val = f"{relation}({','.join(args)})"
        else:
            relation = "Atom"
            args = [cleaned]
            canonical_val = cleaned

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
