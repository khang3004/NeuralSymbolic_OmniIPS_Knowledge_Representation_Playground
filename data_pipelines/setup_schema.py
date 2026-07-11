"""
GeoIPS — Database Schema Setup.

Initializes:
1. Neo4j constraints and indices for geometry-only KB.
2. Qdrant geometry_facts collection (via factory — cloud or local).
"""

import os
import logging
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_db.connection import Neo4jConnection
from graph_db.qdrant_factory import get_qdrant_client
from qdrant_client.models import Distance, VectorParams

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup_schema")


def setup_neo4j_constraints():
    """Create Neo4j constraints and indices for GeoIPS."""
    logger.info("Setting up Neo4j constraints...")
    conn = Neo4jConnection()
    try:
        with conn.get_session() as session:
            session.run(
                "CREATE CONSTRAINT fact_id_domain_unique IF NOT EXISTS "
                "FOR (f:Fact) REQUIRE (f.id, f.domain) IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT rule_id_domain_unique IF NOT EXISTS "
                "FOR (r:Rule) REQUIRE (r.id, r.domain) IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT ontology_class_name_unique IF NOT EXISTS "
                "FOR (c:OntologyClass) REQUIRE c.name IS UNIQUE"
            )
            session.run(
                "CREATE INDEX fact_domain_idx IF NOT EXISTS FOR (f:Fact) ON (f.domain)"
            )
            session.run(
                "CREATE INDEX rule_domain_idx IF NOT EXISTS FOR (r:Rule) ON (r.domain)"
            )
            logger.info("Neo4j constraints and indices created successfully.")
    except Exception as e:
        logger.error("Failed to create Neo4j constraints: %s", e)
    finally:
        conn.close()


def setup_qdrant_collection():
    """Create geometry_facts collection in Qdrant (cloud or local)."""
    logger.info("Setting up Qdrant geometry_facts collection...")
    try:
        client = get_qdrant_client()
        collection_name = "geometry_facts"
        vector_size = 384  # all-MiniLM-L6-v2 embedding size

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", collection_name)
        else:
            logger.info("Qdrant collection already exists: %s", collection_name)
    except Exception as e:
        logger.error("Failed to setup Qdrant collection: %s", e)


if __name__ == "__main__":
    setup_neo4j_constraints()
    setup_qdrant_collection()
