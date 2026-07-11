"""
GeoIPS — Qdrant Payload Index Creator.

Creates a payload index on the 'value' field (keyword type) of the geometry_facts
collection. This fixes the Qdrant 400 Bad Request error when using exact filtering.
"""

import os
import sys
import logging
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_db.qdrant_factory import get_qdrant_client
from qdrant_client.models import PayloadSchemaType

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup_qdrant_indices")

def create_payload_indices():
    logger.info("Initializing Qdrant client...")
    client = get_qdrant_client()
    collection_name = "geometry_facts"
    
    logger.info("Creating payload index for field 'value' in collection '%s'...", collection_name)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="value",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logger.info("✅ Payload index created successfully for field 'value'.")
    except Exception as e:
        logger.error("Failed to create payload index: %s", e)

if __name__ == "__main__":
    create_payload_indices()
