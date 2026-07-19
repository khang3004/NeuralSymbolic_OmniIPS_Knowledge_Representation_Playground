from abc import ABC, abstractmethod
from core_engine.models import Fact, Rule

class DomainParser(ABC):
    """
    Abstract Base Class defining syntax parsers and registrations for logical domains.
    Ensure all domain implementations adhere strictly to this interface.
    """
    @property
    @abstractmethod
    def domain_name(self) -> str:
        """
        Returns the domain identifier string (e.g., 'geometry').
        """
        pass

    @abstractmethod
    def parse_fact(self, raw_input: str, fact_id: str) -> Fact:
        """
        Parses a domain-specific string representation into a generic Fact.
        """
        pass

    @abstractmethod
    def parse_rule(self, raw_rule: dict) -> Rule:
        """
        Parses a domain-specific dictionary into a generic Rule.
        """
        pass

    @abstractmethod
    def format_fact(self, fact: Fact) -> str:
        """
        Formats a generic Fact back into the domain-specific string representation.
        """
        pass
