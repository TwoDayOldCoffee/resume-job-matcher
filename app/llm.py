"""Groq LLM client for the app. No disk cache: resume text is never written to disk."""
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()  # locally reads .env; on Hugging Face the key comes from the Space's secrets

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError("GROQ_API_KEY is not set")

_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=_api_key,
                 max_retries=0, timeout=60)


class LLMLimitError(Exception):
    """Raised when Groq's rate limit can't be waited out within a web request."""


@lru_cache(maxsize=256)
def llm(prompt, model="openai/gpt-oss-20b", system=None, reasoning_effort="low", max_retries=5):
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    for _ in range(max_retries):
        try:
            resp = _client.chat.completions.create(model=model, messages=messages,
                                                   reasoning_effort=reasoning_effort)
            return resp.choices[0].message.content
        except RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 10))
            if wait > 60:
                raise LLMLimitError("Daily LLM limit reached - please try again later.") from e
            time.sleep(wait + 1)
    raise LLMLimitError("Rate limited repeatedly - please try again in a minute.")