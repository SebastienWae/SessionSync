from __future__ import annotations

import base64
import logging
import operator
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, JsonValue, ValidationError, field_validator

from sessionsync.schema import (
    AgentType,
    AssistantMessage,
    AttachmentMessage,
    Message,
    Session,
    ToolResultMessage,
    ToolUseMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

AGENT_TYPE = AgentType.OPENCODE

_MS_THRESHOLD = 1e12


class _RawTime(BaseModel):
    created: datetime | None = None

    @field_validator("created", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | None) -> datetime | None:
        """Parse a Unix timestamp to a datetime object.

        Args:
            v: Unix timestamp in seconds or milliseconds, or None.

        Returns:
            Timezone-aware datetime object in UTC, or None.
        """
        if v is None:
            return None
        ts = v / 1000 if v > _MS_THRESHOLD else v
        return datetime.fromtimestamp(ts, tz=UTC)


class _RawSession(BaseModel):
    id: str
    parent_id: str | None = Field(default=None, alias="parentID")
    directory: str | None = None


class _RawMessage(BaseModel):
    id: str
    session_id: str | None = Field(default=None, alias="sessionID")
    role: str | None = None
    model_id: str | None = Field(default=None, alias="modelID")
    time: _RawTime | None = None


class _RawPartState(BaseModel):
    input: dict[str, JsonValue] | None = None
    output: str | None = None


class _RawPart(BaseModel):
    type: Literal[
        "text",
        "reasoning",
        "reasoning.text",
        "reasoning.encrypted",
        "tool",
        "file",
        "step-start",
        "step-finish",
        "snapshot",
        "patch",
        "agent",
        "retry",
        "compaction",
        "subtask",
    ]
    text: str | None = None
    tool: str | None = None
    call_id: str | None = Field(default=None, alias="callID")
    state: _RawPartState | None = None
    mime: str | None = None
    filename: str | None = None
    url: str | None = None


class _SessionIdExtract(BaseModel):
    id: str | None = None
    session_id: str | None = Field(default=None, alias="sessionID")


def get_root() -> Path | None:
    """Get the OpenCode storage root directory.

    Returns:
        Path to the storage directory if found, None otherwise.
    """
    if env_path := os.environ.get("OPENCODE_STORAGE_ROOT"):
        p = Path(env_path)
        if p.exists():
            return p

    if xdg_data := os.environ.get("XDG_DATA_HOME"):
        storage_dir = Path(xdg_data) / "opencode" / "storage"
        if storage_dir.exists():
            return storage_dir

    storage_dir = Path.home() / ".local" / "share" / "opencode" / "storage"
    return storage_dir if storage_dir.exists() else None


def _extract_data_from_url(url: str) -> str:
    """Extract base64 data from a data URL or read file content.

    Args:
        url: A data URL (data:mime;base64,...) or file path.

    Returns:
        Base64 encoded data string.
    """
    if url.startswith("data:"):
        if "," in url:
            return url.split(",", 1)[1]
        return ""
    try:
        return base64.b64encode(Path(url).read_bytes()).decode("ascii")
    except OSError:
        return ""


def _process_tool_part(raw: _RawPart, created_at: datetime, model: str) -> list[Message]:
    """Convert a raw tool part into normalized messages.

    Args:
        raw: Raw part data from OpenCode.
        created_at: Timestamp for the messages.
        model: Model identifier string.

    Returns:
        List of ToolUseMessage and optionally ToolResultMessage.
    """
    if not raw.tool or not raw.call_id:
        return []

    messages: list[Message] = [
        ToolUseMessage(
            created_at=created_at,
            model=model,
            tool_id=raw.call_id,
            tool_name=raw.tool,
            parameters=raw.state.input if raw.state and raw.state.input else {},
        )
    ]
    if raw.state and raw.state.output:
        messages.append(
            ToolResultMessage(
                created_at=created_at,
                tool_id=raw.call_id,
                result={"content": raw.state.output, "is_error": False},
            )
        )
    return messages


def _parts_to_messages(
    part_dir: Path,
    message_id: str,
    created_at: datetime,
    model: str,
    *,
    is_user: bool,
) -> list[Message]:
    """Convert message parts from a directory into normalized messages.

    Args:
        part_dir: Root directory containing part subdirectories.
        message_id: ID of the message to load parts for.
        created_at: Timestamp for the messages.
        model: Model identifier string.
        is_user: Whether this is a user message.

    Returns:
        List of normalized Message objects.
    """
    msg_part_dir = part_dir / message_id
    if not msg_part_dir.exists():
        return []

    messages: list[Message] = []
    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for part_file in sorted(msg_part_dir.glob("*.json")):
        try:
            raw = _RawPart.model_validate_json(part_file.read_bytes())
        except (OSError, ValidationError) as e:
            logger.debug("Skipped part file %s: %s", part_file, e)
            continue

        match raw.type:
            case "text":
                if raw.text and raw.text.strip():
                    text_parts.append(raw.text.strip())
            case "reasoning" | "reasoning.text":
                if raw.text and raw.text.strip():
                    thinking_parts.append(raw.text.strip())
            case "reasoning.encrypted":
                thinking_parts.append("[encrypted reasoning]")
            case "tool":
                messages.extend(_process_tool_part(raw, created_at, model))
            case "file":
                if raw.mime and raw.url:
                    data = _extract_data_from_url(raw.url)
                    if data:
                        messages.append(
                            AttachmentMessage(
                                created_at=created_at,
                                media_type=raw.mime,
                                data=data,
                                filename=raw.filename,
                            )
                        )
            case "step-start" | "step-finish" | "snapshot" | "patch" | "agent" | "retry" | "compaction" | "subtask":
                pass

    if text_parts:
        content = "\n".join(text_parts)
        if is_user:
            messages.insert(0, UserMessage(created_at=created_at, content=content))
        else:
            messages.insert(0, AssistantMessage(created_at=created_at, model=model, is_thinking=False, content=content))

    if thinking_parts:
        messages.append(
            AssistantMessage(created_at=created_at, model=model, is_thinking=True, content="\n".join(thinking_parts))
        )

    return messages


def _load_messages(storage_root: Path, session_id: str) -> list[Message]:
    """Load all messages for a session from the storage directory.

    Args:
        storage_root: Root path of the OpenCode storage.
        session_id: ID of the session to load messages for.

    Returns:
        List of normalized Message objects sorted by timestamp.
    """
    message_dir = storage_root / "message" / session_id
    part_dir = storage_root / "part"

    if not message_dir.exists():
        return []

    raw_data: list[tuple[_RawMessage, datetime]] = []
    min_datetime = datetime.min.replace(tzinfo=UTC)
    for msg_file in message_dir.glob("*.json"):
        try:
            raw = _RawMessage.model_validate_json(msg_file.read_bytes())
        except (OSError, ValidationError) as e:
            logger.debug("Skipped message file %s: %s", msg_file, e)
            continue
        if raw.role not in {"user", "assistant"}:
            continue
        ts = raw.time.created if raw.time and raw.time.created else min_datetime
        raw_data.append((raw, ts))

    raw_data.sort(key=operator.itemgetter(1))

    messages: list[Message] = []
    for raw, created_at in raw_data:
        if raw.time is None or raw.time.created is None:
            continue
        model = raw.model_id or "unknown"
        messages.extend(_parts_to_messages(part_dir, raw.id, created_at, model, is_user=raw.role == "user"))

    return messages


def _parse_session_file(session_file: Path) -> Session | None:
    """Parse a session file into a normalized Session object.

    Args:
        session_file: Path to the session JSON file.

    Returns:
        Normalized Session object, or None if parsing fails.
    """
    try:
        raw = _RawSession.model_validate_json(session_file.read_bytes())
    except (OSError, ValidationError) as e:
        logger.debug("Failed to parse session file %s: %s", session_file, e)
        return None

    if not raw.directory:
        logger.debug("No directory in session %s", session_file)
        return None

    return Session(
        id=raw.id,
        agent=AGENT_TYPE,
        workspace=Path(raw.directory),
        git_branch=None,
        parent_session_id=raw.parent_id,
    )


def _find_session_file(session_id: str) -> Path | None:
    """Find the session file for a given session ID.

    Args:
        session_id: The session ID to find.

    Returns:
        Path to the session JSON file, or None if not found.
    """
    storage_root = get_root()
    if storage_root is None:
        return None

    session_dir = storage_root / "session"
    if not session_dir.exists():
        return None

    for project_dir in session_dir.iterdir():
        if not project_dir.is_dir():
            continue
        session_file = project_dir / f"{session_id}.json"
        if session_file.exists():
            return session_file

    return None


def _extract_session_id_from_json(file_path: Path) -> str | None:
    """Extract session ID from a JSON file's content.

    Args:
        file_path: Path to a JSON file.

    Returns:
        Session ID if found, None otherwise.
    """
    try:
        raw = _SessionIdExtract.model_validate_json(file_path.read_bytes())
    except (OSError, ValidationError) as e:
        logger.debug("Failed to extract session ID from %s: %s", file_path, e)
        return None
    return raw.session_id or raw.id


def get_sessions() -> Iterator[Session]:
    """Get all OpenCode sessions.

    Yields:
        Session objects for all sessions.
    """
    storage_root = get_root()
    if storage_root is None:
        return

    session_dir = storage_root / "session"
    if not session_dir.exists():
        return

    seen_ids: set[str] = set()
    for project_dir in session_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for session_file in project_dir.glob("*.json"):
            session_id = session_file.stem
            if session_id in seen_ids:
                continue
            seen_ids.add(session_id)
            if session := _parse_session_file(session_file):
                yield session


def get_session_from_id(session_id: str) -> Session | None:
    """Get a session by its ID.

    Args:
        session_id: The session ID.

    Returns:
        The Session if found, None otherwise.
    """
    if session_file := _find_session_file(session_id):
        return _parse_session_file(session_file)
    return None


def get_session_from_path(session_path: Path) -> Session | None:
    """Get a session from a file path.

    Args:
        session_path: Path to a session-related file.

    Returns:
        The Session if found, None otherwise.
    """
    if not session_path.exists() or session_path.suffix != ".json":
        return None
    if session_id := _extract_session_id_from_json(session_path):
        return get_session_from_id(session_id)
    return None


def get_messages(session: Session) -> Iterator[Message]:
    """Get all messages for a session in chronological order.

    Args:
        session: The Session to get messages for.

    Yields:
        Message objects in chronological order.
    """
    if storage_root := get_root():
        yield from _load_messages(storage_root, session.id)
