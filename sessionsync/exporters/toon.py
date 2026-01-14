from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from toon_format import encode

if TYPE_CHECKING:
    from pydantic import JsonValue

    from sessionsync.schema import Message, Session


def _get_filename(session: Session, messages: list[Message], ext: str) -> str:
    """Generate filename for the exported session.

    Args:
        session: Session metadata.
        messages: List of messages (used for start datetime).
        ext: File extension without dot.

    Returns:
        Filename in format: {start_datetime}_{session_id}.{ext}
        or {start_datetime}_{session_id}_{sub_id}.{ext} for subagents.
    """
    start_dt = messages[0].created_at.astimezone().strftime("%Y-%m-%dT%H-%M-%S") if messages else "unknown"

    if session.parent_session_id is not None:
        sub_id = session.id.split(":")[-1] if ":" in session.id else session.id
        return f"{start_dt}_{session.parent_session_id}_{sub_id}.{ext}"
    return f"{start_dt}_{session.id}.{ext}"


def _normalize_value(obj: object) -> JsonValue:
    """Normalize values for TOON serialization.

    Args:
        obj: Object to normalize.

    Returns:
        TOON-serializable representation.
    """
    match obj:
        case datetime():
            return obj.astimezone().isoformat()
        case Path():
            return str(obj)
        case str() | int() | float() | bool() | None:
            return obj
        case dict():
            return {str(k): _normalize_value(v) for k, v in obj.items()}  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        case list():
            return [_normalize_value(item) for item in obj]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        case _:
            return str(obj)


def export(session: Session, messages: list[Message], output: Path) -> Path:
    """Export session and messages to a TOON file.

    Args:
        session: Session metadata.
        messages: List of filtered messages.
        output: Output directory path.

    Returns:
        Path to the exported file.
    """
    filename = _get_filename(session, messages, "toon")
    filepath = output / filename

    data = {
        "session": _normalize_value(asdict(session)),
        "messages": [_normalize_value(asdict(msg)) for msg in messages],
    }

    filepath.write_text(encode(data))
    return filepath
