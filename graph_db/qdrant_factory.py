"""
GeoIPS — Qdrant Client Factory.

Returns a QdrantClient configured for either:
- Cloud mode (QDRANT_MODE=cloud): connects to Qdrant Cloud via HTTPS + API key.
- Local mode (QDRANT_MODE=local, default): connects to a local Docker/process instance.

Usage:
    from graph_db.qdrant_factory import get_qdrant_client
    client = get_qdrant_client()
"""

import os
import logging
from qdrant_client import QdrantClient

logger = logging.getLogger("qdrant_factory")


def get_qdrant_client() -> QdrantClient:
    """
    Factory function that returns the correct QdrantClient based on QDRANT_MODE env var.

    Environment variables:
        QDRANT_MODE        : "cloud" or "local" (default: "local")
        QDRANT_CLOUD_URL   : Full HTTPS URL of Qdrant Cloud cluster (required for cloud mode)
                             e.g. https://xyz.us-east4-0.gcp.cloud.qdrant.io
        QDRANT_CLOUD_API_KEY: API key from Qdrant Cloud dashboard (required for cloud mode)
        QDRANT_HOST        : Hostname for local mode (default: "localhost")
        QDRANT_PORT        : Port for local mode (default: 6333)

    Returns:
        A ready-to-use QdrantClient instance.
    """
    mode = os.getenv("QDRANT_MODE", "local").strip().lower()

    if mode == "cloud":
        cloud_url = os.getenv("QDRANT_CLOUD_URL", "").strip()
        api_key = os.getenv("QDRANT_CLOUD_API_KEY", "").strip()

        if not cloud_url:
            raise ValueError(
                "QDRANT_MODE=cloud but QDRANT_CLOUD_URL is not set. "
                "Please add it to your .env file."
            )
        if not api_key or "your_" in api_key:
            raise ValueError(
                "QDRANT_MODE=cloud but QDRANT_CLOUD_API_KEY is not set or is a placeholder. "
                "Get your API key from https://cloud.qdrant.io"
            )

        logger.info("Connecting to Qdrant Cloud at: %s", cloud_url)
        return QdrantClient(
            url=cloud_url,
            api_key=api_key,
            timeout=30,
        )

    else:
        # Local mode — Docker container or local process
        host = os.getenv("QDRANT_HOST", "localhost").strip()
        port = int(os.getenv("QDRANT_PORT", "6333"))

        logger.info("Connecting to local Qdrant at %s:%d", host, port)
        return QdrantClient(host=host, port=port)
