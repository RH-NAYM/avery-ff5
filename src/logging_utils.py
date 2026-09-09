"""Structured JSON logging shared by agent.py and api.py.

Both processes normally log with Python's default plain-text formatter,
which is fine to read in a terminal but can't be correlated across the two
processes (or ingested cleanly by a log aggregator) for a given call. This
module gives them one shared, dependency-free JSON formatter instead: no
new package, just a ~30-line stdlib `logging.Formatter`.

Usage: call `configure_logging()` once near the top of each process's
entrypoint, then pass `extra={"call_id": call_id}` (or any other field) to
individual `logger.info(...)`/`logger.warning(...)` calls to have it ride
along in the JSON output.

Defaults to plain text (`logging.basicConfig`) so it doesn't change local
dev behavior -- including LiveKit's own colored `console`/`dev` CLI output
-- unless a production deployment opts in with `LOG_FORMAT=json`.
"""

from __future__ import annotations

import json
import logging
import os

# Every attribute a stock LogRecord carries, plus the couple of pseudo-attrs
# record.getMessage()/formatTime() derive on the fly. Anything NOT in this
# set on a given record is something the caller passed via `extra=`, and
# gets folded into the JSON payload as its own field (e.g. `call_id`).
_STANDARD_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Renders each log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configure the root logger once at process startup.

    `LOG_FORMAT=json` (any deployment env, e.g. LiveKit Cloud agent secrets
    or the outbound-call API's hosting) switches to structured JSON output.
    Anything else (including unset, the default) leaves Python's normal
    plain-text logging alone.
    """
    if os.environ.get("LOG_FORMAT", "text").lower() != "json":
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
