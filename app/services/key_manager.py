import asyncio
import logging
import os
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, TooManyRequests, PermissionDenied, GoogleAPICallError
from app.config import settings

logger = logging.getLogger("key_manager")

# 1. Load keys dynamically from environment / config (GEMINI_API_KEY & GEMINI_BACKUP_KEYS)
ENV_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or getattr(settings, "GEMINI_API_KEY", "")

backup_env_str = os.getenv("GEMINI_BACKUP_KEYS", "")
BACKUP_KEYS = [k.strip() for k in backup_env_str.split(",") if k.strip()]

RAW_KEYS = [ENV_KEY] + BACKUP_KEYS
AVAILABLE_KEYS = []
for k in RAW_KEYS:
    if k and isinstance(k, str) and k.strip() and k.strip() not in AVAILABLE_KEYS:
        AVAILABLE_KEYS.append(k.strip())

if not AVAILABLE_KEYS:
    raise ValueError("System requires at least 1 valid API Key.")

current_key_index = 0

async def call_llm_api_with_fallback(func, *args, **kwargs):
    """
    Wrapper executing Gemini API calls. Automatically rotates keys upon Rate Limit/Quota/ResourceExhausted errors.
    """
    global current_key_index
    max_attempts = len(AVAILABLE_KEYS)
    attempts = 0

    while attempts < max_attempts:
        active_key = AVAILABLE_KEYS[current_key_index]
        
        try:
            # Configure SDK with active key
            genai.configure(api_key=active_key)
            
            # Execute target function
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return await asyncio.to_thread(func, *args, **kwargs)
            
        except (ResourceExhausted, TooManyRequests, PermissionDenied, GoogleAPICallError) as e:
            err_msg = str(e).lower()
            is_quota_error = isinstance(e, (ResourceExhausted, TooManyRequests, PermissionDenied)) or any(
                kw in err_msg for kw in ["quota", "rate limit", "429", "403", "resourceexhausted", "resource_exhausted"]
            )
            
            if is_quota_error and len(AVAILABLE_KEYS) > 1:
                masked_key = active_key[-4:] if len(active_key) >= 4 else active_key
                logger.warning(
                    "[API Key Manager] Key ending in '%s' hit rate limit/quota (%s). Rotating to backup key...",
                    masked_key, type(e).__name__
                )
                current_key_index = (current_key_index + 1) % len(AVAILABLE_KEYS)
                attempts += 1
                await asyncio.sleep(0.5)
            else:
                raise e
        except Exception as e:
            err_msg = str(e).lower()
            if any(kw in err_msg for kw in ["quota", "rate limit", "429", "403", "resourceexhausted"]) and len(AVAILABLE_KEYS) > 1:
                masked_key = active_key[-4:] if len(active_key) >= 4 else active_key
                logger.warning(
                    "[API Key Manager] Key ending in '%s' hit rate limit/quota (%s). Rotating to backup key...",
                    masked_key, type(e).__name__
                )
                current_key_index = (current_key_index + 1) % len(AVAILABLE_KEYS)
                attempts += 1
                await asyncio.sleep(0.5)
            else:
                raise e

    raise RuntimeError("Tất cả API Keys trong pool đều đã cạn kiệt hoặc bị chặn.")
