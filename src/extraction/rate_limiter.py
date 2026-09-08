"""Thread-safe rate limiter with sliding window and inter-request pacing."""

import time
import threading
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Guarantees strict compliance with API rate limits (e.g. 50 RPM, 1000 RPD).
    Enforces minimum inter-request interval (e.g. 1.5 seconds) across concurrent threads.
    """

    def __init__(
        self,
        max_rpm: float = 40.0,
        max_rpd: int = 1000,
        min_interval_seconds: float = 1.5,
    ):
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd
        self.min_interval = min_interval_seconds
        self.lock = threading.Lock()
        self.last_request_time: float = 0.0
        self.cooldown_until: float = 0.0
        self.minute_window: deque = deque()
        self.day_window: deque = deque()

    def acquire(self) -> None:
        """Block until it is safe to execute the next request within rate limits."""
        with self.lock:
            now = time.time()

            # 0. Enforce global cooldown if penalize was triggered by a 429
            if now < self.cooldown_until:
                wait_cool = self.cooldown_until - now
                logger.warning(f"Groq cooldown active: holding for {wait_cool:.2f}s...")
                time.sleep(wait_cool)
                now = time.time()

            # 1. Purge timestamps older than 60s and 86400s (24h)
            while self.minute_window and now - self.minute_window[0] > 60.0:
                self.minute_window.popleft()
            while self.day_window and now - self.day_window[0] > 86400.0:
                self.day_window.popleft()

            # 2. Check Daily Limit (1000 RPD)
            if len(self.day_window) >= self.max_rpd:
                wait_day = 86400.0 - (now - self.day_window[0]) + 0.1
                logger.warning(f"Groq RPD limit reached ({self.max_rpd}/day). Waiting {wait_day:.1f}s.")
                time.sleep(max(0.1, wait_day))
                now = time.time()

            # 3. Check Minute Window Limit (<= max_rpm, default 40 RPM < 50 RPM limit)
            if len(self.minute_window) >= self.max_rpm:
                wait_min = 60.0 - (now - self.minute_window[0]) + 0.05
                logger.info(f"Groq RPM throttle active ({len(self.minute_window)} reqs in window). Pacing for {wait_min:.2f}s.")
                time.sleep(max(0.05, wait_min))
                now = time.time()

            # 4. Enforce minimum inter-request pacing (1.5s interval)
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_needed = self.min_interval - elapsed
                logger.debug(f"Pacing Groq request by {sleep_needed:.2f}s (min interval: {self.min_interval}s).")
                time.sleep(sleep_needed)
                now = time.time()

            # Record timestamp
            self.last_request_time = now
            self.minute_window.append(now)
            self.day_window.append(now)

    def get_stats(self) -> dict:
        """Return current rate limiter telemetry."""
        with self.lock:
            now = time.time()
            current_rpm = len([t for t in self.minute_window if now - t <= 60.0])
            current_rpd = len([t for t in self.day_window if now - t <= 86400.0])
            return {
                "rpm_used": current_rpm,
                "rpm_limit": self.max_rpm,
                "rpd_used": current_rpd,
                "rpd_limit": self.max_rpd,
                "min_interval_seconds": self.min_interval,
            }

    def configure(
        self,
        max_rpm: Optional[float] = None,
        min_interval_seconds: Optional[float] = None,
        max_rpd: Optional[int] = None,
    ) -> None:
        """Update rate limiting parameters dynamically."""
        with self.lock:
            if max_rpm is not None:
                self.max_rpm = max_rpm
            if min_interval_seconds is not None:
                self.min_interval = min_interval_seconds
            if max_rpd is not None:
                self.max_rpd = max_rpd
            logger.info(
                f"RateLimiter updated: max_rpm={self.max_rpm}, min_interval={self.min_interval}s, max_rpd={self.max_rpd}"
            )

    def penalize(self, cooldown_seconds: float) -> None:
        """Enforce a global cooldown across all threads after a 429 response."""
        with self.lock:
            now = time.time()
            target = now + cooldown_seconds
            if target > self.cooldown_until:
                self.cooldown_until = target
                logger.warning(
                    f"RateLimiter penalize: global 429 cooldown active for {cooldown_seconds:.2f}s."
                )


# Process-wide singleton instance configured from settings
from src.config import settings

groq_rate_limiter = RateLimiter(
    max_rpm=settings.groq_rpm_limit,
    max_rpd=settings.groq_rpd_limit,
    min_interval_seconds=settings.groq_min_interval_seconds,
)

