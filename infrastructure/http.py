"""
api_utils.py — Shared HTTP utilities with retry/backoff
=========================================================
Provides a retry wrapper for aiohttp requests. All API modules
(forecast_scanner, ensemble_blender, polymarket_parser) use this
instead of raw session.get() to handle transient failures gracefully.

NWS in particular is notorious for intermittent 5xx errors and timeouts.
A simple 3-retry with exponential backoff catches most transient issues.
"""

import asyncio
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0  # seconds; doubles each retry: 1s, 2s, 4s


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: Optional[dict] = None,
    timeout_sec: float = 15.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    label: str = "",
) -> Optional[dict]:
    """
    GET a JSON endpoint with exponential backoff on transient failures.

    Returns parsed JSON dict on success, None on exhausted retries.
    Retries on: 5xx status, timeouts, connection errors.
    Does NOT retry on: 4xx (client errors — our fault, retrying won't help).
    """
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    for attempt in range(1, max_retries + 1):
        try:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                if resp.status == 200:
                    # Guard against HTML maintenance pages returned as 200 OK.
                    # NWS frequently returns 200 + HTML instead of JSON during
                    # maintenance. We must catch the parse error and retry.
                    try:
                        return await resp.json()
                    except (aiohttp.ContentTypeError, ValueError) as e:
                        logger.warning(
                            f"[{label}] JSON parse error on 200 OK "
                            f"(attempt {attempt}/{max_retries}): {e}"
                        )
                        # Fall through to retry logic below
                        continue

                # 4xx — don't retry, it's a client error
                if 400 <= resp.status < 500:
                    logger.warning(f"[{label}] HTTP {resp.status} (client error, no retry): {url}")
                    return None

                # 5xx — transient server error, retry
                logger.warning(
                    f"[{label}] HTTP {resp.status} (attempt {attempt}/{max_retries}): {url}"
                )

        except asyncio.TimeoutError:
            logger.warning(f"[{label}] Timeout (attempt {attempt}/{max_retries}): {url}")
        except aiohttp.ClientError as e:
            logger.warning(f"[{label}] Connection error (attempt {attempt}/{max_retries}): {e}")
        except Exception as e:
            logger.error(f"[{label}] Unexpected error (attempt {attempt}/{max_retries}): {e}")
            return None  # unknown error — don't retry

        if attempt < max_retries:
            delay = backoff_base * (2 ** (attempt - 1))
            logger.debug(f"[{label}] Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

    logger.error(f"[{label}] All {max_retries} retries exhausted: {url}")
    return None
