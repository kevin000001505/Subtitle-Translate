import os
import time
import asyncio
import logging
import re
import requests

from subtitle_files import SOURCE_LANG, language_name

logger = logging.getLogger("SubtitleAPI")

TRANSLATE_GEMMA_MODEL = os.getenv("TRANSLATE_GEMMA_MODEL", "translatemgemma")
TRANSLATE_GEMMA_URL = os.getenv("TRANSLATE_GEMMA_URL", "http://localhost:8080")
TRANSLATE_GEMMA_API_KEY = os.getenv("TRANSLATE_GEMMA_API_KEY", "random_key")


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def uses_translategemma_prompt(model: str) -> bool:
    """TranslateGemma needs its structured request (language codes beside the
    text); every other model gets a plain chat prompt. The configured default
    model is TranslateGemma by definition of TRANSLATE_GEMMA_MODEL."""
    return "translategemma" in model.lower() or model == TRANSLATE_GEMMA_MODEL


def list_models(http=requests) -> list[dict]:
    """The models llama-swap serves, from its OpenAI-compatible
    GET /v1/models (the list Open WebUI reads). Alias entries are left out so
    each model appears once. Raises requests.RequestException when llama-swap
    can't be reached."""
    response = http.get(f"{TRANSLATE_GEMMA_URL.rstrip('/')}/v1/models",
                        headers={"Authorization": f"Bearer {TRANSLATE_GEMMA_API_KEY}"}, timeout=5)
    response.raise_for_status()
    models = []
    for record in response.json().get("data", []):
        llamaswap = (record.get("meta") or {}).get("llamaswap") or {}
        if llamaswap.get("type") == "alias":
            continue
        models.append({
            "id": record["id"],
            "name": record.get("name") or record["id"],
            "loaded": (record.get("status") or {}).get("value") == "loaded",
        })
    return sorted(models, key=lambda m: m["name"].lower())


class TranslateGemmaClient:
    def __init__(self):
        self.base_url = TRANSLATE_GEMMA_URL.rstrip("/")
        self.api_key = TRANSLATE_GEMMA_API_KEY
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self._last_request_time = 0.0
        self._min_interval = 0.1  # 100ms between requests to avoid overwhelming local server
        self._rate_lock = asyncio.Lock()

    async def translate(self, batch_text: str, context_text: str = "", model: str = TRANSLATE_GEMMA_MODEL,
                        target_lang: str = "zh-TW") -> str | None:
        url = f"{self.base_url}/v1/chat/completions"
        instructions = (
            "CRITICAL RULES:\n"
            "1. Format MUST be: [ID] Translated Text (e.g., [0] Hello world)\n"
            "2. If the original has multiple lines, your output MUST have multiple lines too.\n"
            "3. Do not merge separate dialogue lines into one.\n"
            "4. Keep the [ID] numbers in the same order as the input.\n"
            "5. Do NOT add any extra text, explanations, or markdown formatting.\n"
            "6. Output ONLY the translated lines in [ID] format.\n\n"
        )
        if context_text:
            instructions += f"CONTEXT: The movie is about: {context_text[:2000]}...\n\n"
        prompt = f"""{instructions}INPUT:
{batch_text}"""

        if uses_translategemma_prompt(model):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": SOURCE_LANG,
                            "target_lang_code": target_lang,
                            "text": prompt,
                        }
                    ],
                }
            ]
        else:
            messages = [
                {"role": "system", "content": (
                    f"You translate English subtitles into {language_name(target_lang)} "
                    f"({target_lang}).\n\n{instructions}")},
                {"role": "user", "content": f"INPUT:\n{batch_text}"},
            ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            # Qwen-style models think aloud unless told not to.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        for attempt in range(3):
            try:
                # Rate limiting — serialize the pre-call wait across coroutines
                async with self._rate_lock:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self._min_interval:
                        await asyncio.sleep(self._min_interval - elapsed)
                    self._last_request_time = time.time()

                response = await asyncio.to_thread(
                    requests.post, url,
                    headers=self.headers, json=payload, timeout=120,
                )

                if response.status_code != 200:
                    logger.warning(f"TranslateGemma error: {response.status_code} - {response.text[:200]}")
                    if attempt == 2:
                        return None
                    await asyncio.sleep((attempt + 1) * 2)
                    continue

                data = response.json()
                return _THINK_RE.sub("", data["choices"][0]["message"]["content"]).strip()

            except Exception as e:
                logger.warning(f"TranslateGemma request failed (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return None
                await asyncio.sleep((attempt + 1) * 2)
