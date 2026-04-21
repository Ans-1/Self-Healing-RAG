"""
LLM factory — swap providers by changing config.yaml (llm.provider).

Supported providers:
  grok       — xAI Grok API (OpenAI-compatible)
  openai     — OpenAI
  anthropic  — Anthropic Claude
  ollama     — Local Ollama instance
"""
import os
from langchain_core.language_models.chat_models import BaseChatModel
from src.config import load_config


def get_llm() -> BaseChatModel:
    config = load_config()
    llm_cfg = config["llm"]

    provider = llm_cfg["provider"]
    model = llm_cfg["model"]
    temperature = llm_cfg.get("temperature", 0.0)
    max_tokens = llm_cfg.get("max_tokens", 2048)

    if provider == "grok":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise EnvironmentError("GROK_API_KEY is not set in your .env file.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in your .env file.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set in your .env file.")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Supported: grok | openai | anthropic | ollama"
        )
