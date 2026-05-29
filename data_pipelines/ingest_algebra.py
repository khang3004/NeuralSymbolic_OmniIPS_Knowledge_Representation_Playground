
import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_db.connection import Neo4jConnection
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_algebra")

DOMAIN = "algebra"
COLLECTION_NAME = "algebra_facts"

# =============================================================================
# COMPREHENSIVE ELEMENTARY & INTERMEDIATE ALGEBRA KNOWLEDGE BASE
# Covers field axioms, equality properties, exponents, factoring identities,
# equation solving techniques, and fundamental inequalities.
# =============================================================================

ALGEBRA_RULES = [
    # =========================================================================
    # A. FIELD AXIOMS (8 rules)
    # =========================================================================
    {
        "id": "alg_commutative_add",
        "name": "Commutative Property of Addition",
        "inputs": ["Expression(a+b)"],
        "outputs": ["Equal(a+b,b+a)"],
        "description": "a + b = b + a. The order of addition does not change the sum."
    },
    {
        "id": "alg_commutative_mul",
        "name": "Commutative Property of Multiplication",
        "inputs": ["Expression(a*b)"],
        "outputs": ["Equal(a*b,b*a)"],
        "description": "a × b = b × a. The order of multiplication does not change the product."
    },
    {
        "id": "alg_associative_add",
        "name": "Associative Property of Addition",
        "inputs": ["Expression((a+b)+c)"],
        "outputs": ["Equal((a+b)+c,a+(b+c))"],
        "description": "(a + b) + c = a + (b + c). Grouping does not affect the sum."
    },
    {
        "id": "alg_associative_mul",
        "name": "Associative Property of Multiplication",
        "inputs": ["Expression((a*b)*c)"],
        "outputs": ["Equal((a*b)*c,a*(b*c))"],
        "description": "(a × b) × c = a × (b × c). Grouping does not affect the product."
    },
    {
        "id": "alg_distributive",
        "name": "Distributive Property",
        "inputs": ["Expression(a*(b+c))"],
        "outputs": ["Equal(a*(b+c),a*b+a*c)"],
        "description": "a(b + c) = ab + ac. Multiplication distributes over addition."
    },
    {
        "id": "alg_identity_add",
        "name": "Additive Identity",
        "inputs": ["Expression(a+0)"],
        "outputs": ["Equal(a+0,a)"],
        "description": "a + 0 = a. Zero is the additive identity element."
    },
    {
        "id": "alg_identity_mul",
        "name": "Multiplicative Identity",
        "inputs": ["Expression(a*1)"],
        "outputs": ["Equal(a*1,a)"],
        "description": "a × 1 = a. One is the multiplicative identity element."
    },
    {
        "id": "alg_inverse_add",
        "name": "Additive Inverse",
        "inputs": ["Expression(a+(-a))"],
        "outputs": ["Equal(a+(-a),0)"],
        "description": "a + (-a) = 0. Every number has an additive inverse."
    },

    # =========================================================================
    # B. EQUALITY PROPERTIES (5 rules)
    # =========================================================================
    {
        "id": "alg_add_both_sides",
        "name": "Addition Property of Equality",
        "inputs": ["Equation(LHS,RHS)", "Add(Val)"],
        "outputs": ["Equation(LHS+Val,RHS+Val)"],
        "description": "If a = b, then a + c = b + c. Adding the same value to both sides preserves equality."
    },
    {
        "id": "alg_sub_both_sides",
        "name": "Subtraction Property of Equality",
        "inputs": ["Equation(LHS,RHS)", "Subtract(Val)"],
        "outputs": ["Equation(LHS-Val,RHS-Val)"],
        "description": "If a = b, then a - c = b - c. Subtracting the same value from both sides preserves equality."
    },
    {
        "id": "alg_mul_both_sides",
        "name": "Multiplication Property of Equality",
        "inputs": ["Equation(LHS,RHS)", "Multiply(Val)"],
        "outputs": ["Equation(LHS*Val,RHS*Val)"],
        "description": "If a = b, then a × c = b × c. Multiplying both sides by the same value preserves equality."
    },
    {
        "id": "alg_div_both_sides",
        "name": "Division Property of Equality",
        "inputs": ["Equation(LHS,RHS)", "Divide(Val)", "NotEqual(Val,0)"],
        "outputs": ["Equation(LHS/Val,RHS/Val)"],
        "description": "If a = b and c ≠ 0, then a/c = b/c. Dividing both sides by a nonzero value preserves equality."
    },
    {
        "id": "alg_substitution",
        "name": "Substitution Property",
        "inputs": ["Equal(x,a)", "Expression(f(x))"],
        "outputs": ["Equal(f(x),f(a))"],
        "description": "If x = a, then x can be replaced by a in any expression: f(x) = f(a)."
    },

    # =========================================================================
    # C. EXPONENT & POWER RULES (7 rules)
    # =========================================================================
    {
        "id": "alg_power_product",
        "name": "Product of Powers Rule",
        "inputs": ["Expression(x^a)", "Expression(x^b)"],
        "outputs": ["Equal(x^a*x^b,x^(a+b))"],
        "description": "x^a × x^b = x^(a+b). When multiplying powers with the same base, add the exponents."
    },
    {
        "id": "alg_power_quotient",
        "name": "Quotient of Powers Rule",
        "inputs": ["Expression(x^a/x^b)"],
        "outputs": ["Equal(x^a/x^b,x^(a-b))"],
        "description": "x^a / x^b = x^(a-b). When dividing powers with the same base, subtract the exponents."
    },
    {
        "id": "alg_power_of_power",
        "name": "Power of a Power Rule",
        "inputs": ["Expression((x^a)^b)"],
        "outputs": ["Equal((x^a)^b,x^(a*b))"],
        "description": "(x^a)^b = x^(ab). When raising a power to another power, multiply the exponents."
    },
    {
        "id": "alg_power_zero",
        "name": "Zero Exponent Rule",
        "inputs": ["Expression(x^0)", "NotEqual(x,0)"],
        "outputs": ["Equal(x^0,1)"],
        "description": "x^0 = 1 for any nonzero x. Any nonzero number raised to the zero power equals one."
    },
    {
        "id": "alg_power_negative",
        "name": "Negative Exponent Rule",
        "inputs": ["Expression(x^(-n))", "NotEqual(x,0)"],
        "outputs": ["Equal(x^(-n),1/x^n)"],
        "description": "x^(-n) = 1/x^n. A negative exponent means the reciprocal of the positive power."
    },
    {
        "id": "alg_power_of_product",
        "name": "Power of a Product Rule",
        "inputs": ["Expression((a*b)^n)"],
        "outputs": ["Equal((a*b)^n,a^n*b^n)"],
        "description": "(ab)^n = a^n × b^n. The power of a product equals the product of the powers."
    },
    {
        "id": "alg_sqrt_definition",
        "name": "Square Root Definition",
        "inputs": ["Expression(sqrt(x))"],
        "outputs": ["Equal(sqrt(x),x^(1/2))"],
        "description": "√x = x^(1/2). The square root is the one-half power."
    },

    # =========================================================================
    # D. FACTORING & NOTABLE IDENTITIES (7 rules)
    # =========================================================================
    {
        "id": "alg_square_of_sum",
        "name": "Square of a Sum (Hằng đẳng thức 1)",
        "inputs": ["Expression((a+b)^2)"],
        "outputs": ["Equal((a+b)^2,a^2+2*a*b+b^2)"],
        "description": "(a + b)² = a² + 2ab + b². The first notable algebraic identity."
    },
    {
        "id": "alg_square_of_diff",
        "name": "Square of a Difference (Hằng đẳng thức 2)",
        "inputs": ["Expression((a-b)^2)"],
        "outputs": ["Equal((a-b)^2,a^2-2*a*b+b^2)"],
        "description": "(a - b)² = a² - 2ab + b². The second notable algebraic identity."
    },
    {
        "id": "alg_diff_of_squares",
        "name": "Difference of Squares (Hằng đẳng thức 3)",
        "inputs": ["Expression(a^2-b^2)"],
        "outputs": ["Equal(a^2-b^2,(a+b)*(a-b))"],
        "description": "a² - b² = (a + b)(a - b). The third notable algebraic identity."
    },
    {
        "id": "alg_cube_of_sum",
        "name": "Cube of a Sum (Hằng đẳng thức 4)",
        "inputs": ["Expression((a+b)^3)"],
        "outputs": ["Equal((a+b)^3,a^3+3*a^2*b+3*a*b^2+b^3)"],
        "description": "(a + b)³ = a³ + 3a²b + 3ab² + b³."
    },
    {
        "id": "alg_cube_of_diff",
        "name": "Cube of a Difference (Hằng đẳng thức 5)",
        "inputs": ["Expression((a-b)^3)"],
        "outputs": ["Equal((a-b)^3,a^3-3*a^2*b+3*a*b^2-b^3)"],
        "description": "(a - b)³ = a³ - 3a²b + 3ab² - b³."
    },
    {
        "id": "alg_sum_of_cubes",
        "name": "Sum of Cubes (Hằng đẳng thức 6)",
        "inputs": ["Expression(a^3+b^3)"],
        "outputs": ["Equal(a^3+b^3,(a+b)*(a^2-a*b+b^2))"],
        "description": "a³ + b³ = (a + b)(a² - ab + b²)."
    },
    {
        "id": "alg_diff_of_cubes",
        "name": "Difference of Cubes (Hằng đẳng thức 7)",
        "inputs": ["Expression(a^3-b^3)"],
        "outputs": ["Equal(a^3-b^3,(a-b)*(a^2+a*b+b^2))"],
        "description": "a³ - b³ = (a - b)(a² + ab + b²)."
    },

    # =========================================================================
    # E. EQUATION SOLVING TECHNIQUES (8 rules)
    # =========================================================================
    {
        "id": "alg_linear_solve",
        "name": "Linear Equation Solution",
        "inputs": ["LinearEquation(a*x+b,0)", "NotEqual(a,0)"],
        "outputs": ["Equal(x,Neg(b)/a)"],
        "description": "ax + b = 0 ⟹ x = -b/a. The general solution for a linear equation in one variable."
    },
    {
        "id": "alg_quadratic_formula",
        "name": "Quadratic Formula",
        "inputs": ["QuadraticEquation(a*x^2+b*x+c,0)", "NotEqual(a,0)"],
        "outputs": ["Equal(x,(-b±sqrt(b^2-4*a*c))/(2*a))"],
        "description": "For ax² + bx + c = 0: x = (-b ± √(b² - 4ac)) / (2a). The universal quadratic formula."
    },
    {
        "id": "alg_discriminant",
        "name": "Discriminant of Quadratic",
        "inputs": ["QuadraticEquation(a*x^2+b*x+c,0)"],
        "outputs": ["Equal(Delta,b^2-4*a*c)"],
        "description": "The discriminant Δ = b² - 4ac determines the nature of the roots: Δ > 0 (two distinct real roots), Δ = 0 (one repeated root), Δ < 0 (no real roots)."
    },
    {
        "id": "alg_vieta_sum",
        "name": "Vieta's Formula (Sum of Roots)",
        "inputs": ["QuadraticEquation(a*x^2+b*x+c,0)", "Roots(x1,x2)"],
        "outputs": ["Equal(x1+x2,-b/a)"],
        "description": "For a quadratic equation ax² + bx + c = 0 with roots x₁, x₂: x₁ + x₂ = -b/a."
    },
    {
        "id": "alg_vieta_product",
        "name": "Vieta's Formula (Product of Roots)",
        "inputs": ["QuadraticEquation(a*x^2+b*x+c,0)", "Roots(x1,x2)"],
        "outputs": ["Equal(x1*x2,c/a)"],
        "description": "For a quadratic equation ax² + bx + c = 0 with roots x₁, x₂: x₁ × x₂ = c/a."
    },
    {
        "id": "alg_cross_multiply",
        "name": "Cross Multiplication",
        "inputs": ["Proportion(a/b,c/d)", "NotEqual(b,0)", "NotEqual(d,0)"],
        "outputs": ["Equal(a*d,b*c)"],
        "description": "If a/b = c/d, then a×d = b×c. Cross multiplication for proportions."
    },
    {
        "id": "alg_factor_zero_product",
        "name": "Zero Product Property",
        "inputs": ["Equal(a*b,0)"],
        "outputs": ["Or(Equal(a,0),Equal(b,0))"],
        "description": "If a × b = 0, then a = 0 or b = 0. Foundation of solving factored equations."
    },
    {
        "id": "alg_completing_square",
        "name": "Completing the Square",
        "inputs": ["Expression(x^2+b*x)"],
        "outputs": ["Equal(x^2+b*x,(x+b/2)^2-(b/2)^2)"],
        "description": "x² + bx = (x + b/2)² - (b/2)². Rewriting a quadratic expression by completing the square."
    },

    # =========================================================================
    # F. INEQUALITIES (5 rules)
    # =========================================================================
    {
        "id": "alg_ineq_add",
        "name": "Addition Property of Inequality",
        "inputs": ["GreaterThan(a,b)", "Value(c)"],
        "outputs": ["GreaterThan(a+c,b+c)"],
        "description": "If a > b, then a + c > b + c. Adding the same value to both sides preserves the inequality."
    },
    {
        "id": "alg_ineq_mul_pos",
        "name": "Multiplication by Positive (Inequality)",
        "inputs": ["GreaterThan(a,b)", "GreaterThan(c,0)"],
        "outputs": ["GreaterThan(a*c,b*c)"],
        "description": "If a > b and c > 0, then ac > bc. Multiplying by a positive preserves direction."
    },
    {
        "id": "alg_ineq_mul_neg",
        "name": "Multiplication by Negative (Inequality Flip)",
        "inputs": ["GreaterThan(a,b)", "LessThan(c,0)"],
        "outputs": ["LessThan(a*c,b*c)"],
        "description": "If a > b and c < 0, then ac < bc. Multiplying by a negative reverses the inequality."
    },
    {
        "id": "alg_triangle_ineq",
        "name": "Triangle Inequality (Absolute Value)",
        "inputs": ["Value(a)", "Value(b)"],
        "outputs": ["LessOrEqual(Abs(a+b),Add(Abs(a),Abs(b)))"],
        "description": "|a + b| ≤ |a| + |b|. The absolute value of a sum is at most the sum of absolute values."
    },
    {
        "id": "alg_am_gm",
        "name": "AM-GM Inequality",
        "inputs": ["GreaterOrEqual(a,0)", "GreaterOrEqual(b,0)"],
        "outputs": ["GreaterOrEqual((a+b)/2,sqrt(a*b))"],
        "description": "For non-negative a, b: (a+b)/2 ≥ √(ab). The arithmetic mean is at least the geometric mean."
    },

    # =========================================================================
    # G. DEMO / SPECIFIC INSTANCES (kept for backward compatibility)
    # =========================================================================
    {
        "id": "alg_sub_two",
        "name": "Subtraction Property of 2 (Demo)",
        "inputs": ["x+2=5", "Subtract(2,both_sides)"],
        "outputs": ["x=3"],
        "description": "[DEMO] Subtract 2 from both sides of x+2=5 to yield x=3. Specific instance for testing."
    },
]


class AlgebraIngest:
    def __init__(self):
        self.neo4j_conn = Neo4jConnection()
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_to_neo4j(self):
        logger.info("Loading %d Algebra rules to Neo4j...", len(ALGEBRA_RULES))
        with self.neo4j_conn.get_session() as session:
            # Load Rules
            session.run("""
                UNWIND $batch AS row
                MERGE (r:Rule {id: row.id, domain: $domain})
                ON CREATE SET r.name = row.name, r.description = row.description,
                              r.inputs = row.inputs, r.outputs = row.outputs
                ON MATCH SET r.name = row.name, r.description = row.description,
                             r.inputs = row.inputs, r.outputs = row.outputs
                SET r:Algebra
            """, batch=ALGEBRA_RULES, domain=DOMAIN)
            
            # Extract and load facts (entities) from inputs/outputs
            all_facts = set()
            for r in ALGEBRA_RULES:
                all_facts.update(r["inputs"])
                all_facts.update(r["outputs"])
            
            logger.info("Extracted %d unique facts from algebra rules.", len(all_facts))
            fact_data = [{"value": f, "label": f, "domain": DOMAIN} for f in all_facts]
            
            session.run("""
                UNWIND $batch AS row
                MERGE (f:Fact {value: row.value, domain: row.domain})
                ON CREATE SET f.id = 'alg_fact_' + row.value, f.label = row.label
                SET f:Algebra
            """, batch=fact_data)
            
            # Create Relationships
            session.run("""
                UNWIND $batch AS row
                MATCH (r:Rule {id: row.id, domain: $domain})
                WITH r, row
                UNWIND row.inputs AS input_val
                MATCH (f_in:Fact {value: input_val, domain: $domain})
                MERGE (f_in)-[:HAS_INPUT]->(r)
                WITH r, row
                UNWIND row.outputs AS output_val
                MATCH (f_out:Fact {value: output_val, domain: $domain})
                MERGE (r)-[:HAS_OUTPUT]->(f_out)
            """, batch=ALGEBRA_RULES, domain=DOMAIN)

    def load_to_qdrant(self):
        logger.info("Loading Algebra facts to Qdrant...")
        all_facts = set()
        for r in ALGEBRA_RULES:
            all_facts.update(r["inputs"])
            all_facts.update(r["outputs"])
            
        try:
            from qdrant_client.models import Distance, VectorParams
            self.qdrant_client.delete_collection(COLLECTION_NAME)
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info("Recreated collection '%s' to clear stale vectors.", COLLECTION_NAME)
        except Exception as e:
            logger.warning("Failed to recreate collection '%s': %s", COLLECTION_NAME, e)

        points = []
        for f_val in all_facts:
            vector = self.embed_model.encode(f_val).tolist()
            points.append(PointStruct(
                id=abs(hash(f_val)) % (10**15),
                vector=vector,
                payload={"value": f_val, "label": f_val, "domain": DOMAIN}
            ))
            
        if points:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(COLLECTION_NAME, batch)
                logger.info("  Upserted batch %d/%d (%d points)", i // batch_size + 1, (len(points) + batch_size - 1) // batch_size, len(batch))
        
        logger.info("Total algebra facts loaded to Qdrant: %d", len(points))

    def run(self):
        self.load_to_neo4j()
        self.load_to_qdrant()
        logger.info("✅ Algebra ingestion complete. %d rules, %d unique facts.",
                     len(ALGEBRA_RULES),
                     len(set(f for r in ALGEBRA_RULES for f in r["inputs"] + r["outputs"])))

if __name__ == "__main__":
    AlgebraIngest().run()
