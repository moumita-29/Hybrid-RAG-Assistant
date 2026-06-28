"""LLM wrapper using Hugging Face Inference API.

Replaces local Ollama with a remote open-source model 
hosted on Hugging Face to eliminate local CPU slowness.
"""

import os
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from config import HF_LLM_MODEL

# Load environment variables from .env (for HF_TOKEN)
load_dotenv()

# Enable in-memory caching for all LLM calls globally
set_llm_cache(InMemoryCache())

_llm = None


def get_llm():
    """Return a singleton ChatHuggingFace instance.

    Connects to the Hugging Face Serverless Inference API.
    """
    global _llm
    if _llm is None:
        if not os.environ.get("HF_TOKEN"):
            raise ValueError("HF_TOKEN is missing! Please add it to your .env file.")

        endpoint = HuggingFaceEndpoint(
            repo_id=HF_LLM_MODEL,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            huggingfacehub_api_token=os.environ.get("HF_TOKEN")
        )
        _llm = ChatHuggingFace(llm=endpoint)
    return _llm
