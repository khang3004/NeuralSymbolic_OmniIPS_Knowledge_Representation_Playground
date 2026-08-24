import os
import logging
import threading
from typing import Optional, List
from dotenv import load_dotenv

# Ensure local configurations are loaded
load_dotenv()
logger = logging.getLogger("llm_factory")


class KeyRotator:
    """
    Thread-safe Round-Robin API Key Rotator.
    Dynamically scans environment variables matching a prefix followed by an index (e.g. GEMINI_API_KEY_1)
    and rotates through them on each request to prevent Rate Limit (429) errors.
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.lock = threading.Lock()
        self.counter = 0

    def get_next_key(self) -> Optional[str]:
        keys = []
        # Dynamically collect numbered keys (e.g. PREFIX_1 to PREFIX_10)
        for i in range(1, 11):
            key = os.getenv(f"{self.prefix}_{i}")
            if key and key.strip() and "your_" not in key:
                keys.append(key.strip())

        # Also include standard unnumbered key if defined
        standard_key = os.getenv(self.prefix)
        if standard_key and standard_key.strip() and "your_" not in standard_key:
            if standard_key.strip() not in keys:
                keys.append(standard_key.strip())

        if not keys:
            return None

        with self.lock:
            idx = self.counter % len(keys)
            key = keys[idx]
            self.counter += 1

            # Secure masking for logs
            masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "..."
            logger.info(
                "Rotating API key for '%s'. Selected Key Index: %d/%d (Key: %s)",
                self.prefix,
                idx + 1,
                len(keys),
                masked_key,
            )
            return key


# Initialize Key Rotators for Google, Groq, and OpenAI
gemini_rotator = KeyRotator("GEMINI_API_KEY")
groq_rotator = KeyRotator("GROQ_API_KEY")
openai_rotator = KeyRotator("OPENAI_API_KEY")


def get_llm(temperature: float = 0.3) -> Optional[object]:
    """
    Model-Agnostic LLM Factory using LangChain.
    Dynamically initializes the correct chat model wrapper based on .env variables.
    Supports Round-Robin key rotation to prevent 429 rate limit errors.

    Supported Providers:
    - openai (requires LLM_API_KEY or OPENAI_API_KEY)
    - ollama (requires local/host server, e.g. http://localhost:11434)
    - groq (requires LLM_API_KEY or GROQ_API_KEY)
    - google (requires LLM_API_KEY or GEMINI_API_KEY or GOOGLE_API_KEY)

    Returns:
        An initialized LangChain ChatModel instance, or None if missing config.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    api_key = os.getenv("LLM_API_KEY")

    # Pre-checks for placeholders
    if api_key == "your_llm_api_key_here" or api_key == "your_openai_api_key_here":
        api_key = None

    logger.info("Initializing LLM client. Provider: '%s', Model: '%s'", provider, model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        final_key = api_key or openai_rotator.get_next_key()
        if not final_key:
            logger.warning("OpenAI API Key is missing. Falling back to offline mode.")
            return None

        kwargs = {"model": model, "temperature": temperature, "api_key": final_key}

        # Pull optional base url if configured in environment
        base_url = os.getenv("LLM_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url

        try:
            return ChatOpenAI(**kwargs)
        except Exception as e:
            logger.error("Failed to initialize ChatOpenAI: %s", e)
            return None

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        final_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        )

        try:
            return ChatOllama(model=model, temperature=temperature, base_url=final_url)
        except Exception as e:
            logger.error("Failed to initialize ChatOllama: %s", e)
            return None

    elif provider == "groq":
        from langchain_groq import ChatGroq

        # Auto-correct safeguard or non-reasoning models
        if "safeguard" in model.lower() or not model or model == "gpt-4o-mini":
            logger.warning(
                "Model '%s' is not an optimal Groq model. Auto-correcting to 'openai/gpt-oss-120b'.",
                model,
            )
            model = "openai/gpt-oss-120b"

        # Collect all valid Groq keys
        groq_keys = []
        for i in range(1, 11):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k and k.strip() and "your_" not in k:
                groq_keys.append(k.strip())
        std_key = os.getenv("GROQ_API_KEY")
        if (
            std_key
            and std_key.strip()
            and "your_" not in std_key
            and std_key.strip() not in groq_keys
        ):
            groq_keys.append(std_key.strip())

        if api_key:
            groq_keys = [api_key] + groq_keys

        if not groq_keys:
            logger.warning("Groq API Key is missing. Falling back to offline mode.")
            return None

        # Build models with fallbacks
        models = []
        for idx, k in enumerate(groq_keys):
            try:
                models.append(ChatGroq(model=model, temperature=temperature, api_key=k, max_retries=0))
            except Exception as e:
                logger.error(
                    "Failed to initialize ChatGroq instance %d: %s", idx + 1, e
                )

        if not models:
            return None
        if len(models) == 1:
            return models[0]

        logger.info("Configured Groq provider with %d fallback API keys.", len(models))
        return models[0].with_fallbacks(models[1:])

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Collect all valid Gemini keys
        gemini_keys = []
        for i in range(1, 11):
            k = os.getenv(f"GEMINI_API_KEY_{i}")
            if k and k.strip() and "your_" not in k:
                gemini_keys.append(k.strip())
        std_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if (
            std_key
            and std_key.strip()
            and "your_" not in std_key
            and std_key.strip() not in gemini_keys
        ):
            gemini_keys.append(std_key.strip())

        if api_key:
            gemini_keys = [api_key] + gemini_keys

        if not gemini_keys:
            logger.warning(
                "Google/Gemini API Key is missing. Falling back to offline mode."
            )
            return None

        # Build models with fallbacks
        models = []
        for idx, k in enumerate(gemini_keys):
            try:
                models.append(
                    ChatGoogleGenerativeAI(
                        model=model, temperature=temperature, google_api_key=k
                    )
                )
            except Exception as e:
                logger.error(
                    "Failed to initialize ChatGoogleGenerativeAI instance %d: %s",
                    idx + 1,
                    e,
                )

        if not models:
            return None
        if len(models) == 1:
            return models[0]

        logger.info(
            "Configured Google provider with %d fallback API keys.", len(models)
        )
        return models[0].with_fallbacks(models[1:])

    else:
        logger.warning(
            "Unsupported LLM provider: '%s'. Supported: openai, ollama, groq, google.",
            provider,
        )
        return None
