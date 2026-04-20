import os
import time
import asyncio
import logging
import requests

logger = logging.getLogger("SubtitleAPI")

TRANSLATE_GEMMA_MODEL = os.getenv("TRANSLATE_GEMMA_MODEL", "translatemgemma")
TRANSLATE_GEMMA_URL = os.getenv("TRANSLATE_GEMMA_URL", "http://localhost:8080")
TRANSLATE_GEMMA_API_KEY = os.getenv("TRANSLATE_GEMMA_API_KEY", "random_key")


class TranslateGemmaClient:
    def __init__(self):
        self.base_url = TRANSLATE_GEMMA_URL.rstrip("/")
        self.model = TRANSLATE_GEMMA_MODEL
        self.api_key = TRANSLATE_GEMMA_API_KEY
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self._last_request_time = 0
        self._min_interval = 0.1  # 100ms between requests to avoid overwhelming local server

    async def translate(self, batch_text: str, context_text: str = "") -> str | None:
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

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": "en",
                            "target_lang_code": "zh-TW",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        for attempt in range(3):
            try:
                # Rate limiting
                elapsed = time.time() - self._last_request_time
                if elapsed < self._min_interval:
                    await asyncio.sleep(self._min_interval - elapsed)

                response = requests.post(url, headers=self.headers, json=payload, timeout=120)
                self._last_request_time = time.time()

                if response.status_code != 200:
                    logger.warning(f"TranslateGemma error: {response.status_code} - {response.text[:200]}")
                    if attempt == 2:
                        return None
                    await asyncio.sleep((attempt + 1) * 2)
                    continue

                data = response.json()
                return data["choices"][0]["message"]["content"].strip()

            except Exception as e:
                logger.warning(f"TranslateGemma request failed (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return None
                await asyncio.sleep((attempt + 1) * 2)
