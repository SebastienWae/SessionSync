from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def _json_serializer(obj: object) -> str | dict[str, object]:
    """Custom JSON serializer for non-serializable types.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.

    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, datetime):
        return obj.astimezone().isoformat()
    if isinstance(obj, Path):
        return str(obj)
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def export(session: Session, messages: list[Message], output: Path) -> Path:
    """Export session and messages to a JSON file.

    Args:
        session: Session metadata.
        messages: List of filtered messages.
        output: Output directory path.

    Returns:
        Path to the exported file.
    """
    filename = _get_filename(session, messages, "json")
    filepath = output / filename

    data = {
        "session": asdict(session),
        "messages": [asdict(msg) for msg in messages],
    }

    filepath.write_text(json.dumps(data, indent=2, default=_json_serializer))
    return filepath
