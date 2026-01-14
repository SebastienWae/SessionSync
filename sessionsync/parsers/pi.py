import logging
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, JsonValue, ValidationError

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

AGENT_TYPE = AgentType.PI


def _parse_unix_ms(value: int | datetime) -> datetime:
    """Parse a Unix millisecond timestamp to a UTC datetime.

    Args:
        value: Unix timestamp in milliseconds or an existing datetime.

    Returns:
        Timezone-aware datetime object in UTC.
    """
    if isinstance(value, datetime):
        return value
    return datetime.fromtimestamp(value / 1000, tz=UTC)


_UnixMsTimestamp = Annotated[datetime, BeforeValidator(_parse_unix_ms)]


class _TextContent(BaseModel):
    type: Literal["text"]
    text: str


class _ThinkingContent(BaseModel):
    type: Literal["thinking"]
    thinking: str


class _ImageContent(BaseModel):
    type: Literal["image"]
    data: str
    mime_type: str = Field(alias="mimeType")


class _ToolCall(BaseModel):
    type: Literal["toolCall"]
    id: str
    name: str
    arguments: dict[str, JsonValue]


_AssistantContent = Annotated[_TextContent | _ThinkingContent | _ToolCall, Field(discriminator="type")]
_UserContent = Annotated[_TextContent | _ImageContent, Field(discriminator="type")]


class _RawUserMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    role: Literal["user"]
    content: str | list[_UserContent]
    timestamp: _UnixMsTimestamp


class _RawAssistantMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    role: Literal["assistant"]
    content: list[_AssistantContent]
    model: str = "unknown"
    timestamp: _UnixMsTimestamp


class _RawToolResultMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", populate_by_name=True)
    role: Literal["toolResult"]
    tool_call_id: str = Field(alias="toolCallId")
    content: list[dict[str, str]]
    is_error: bool = Field(default=False, alias="isError")
    timestamp: _UnixMsTimestamp


class _RawBashExecutionMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    role: Literal["bashExecution"]
    command: str
    output: str
    timestamp: _UnixMsTimestamp


_RawMessage = Annotated[
    _RawUserMessage | _RawAssistantMessage | _RawToolResultMessage | _RawBashExecutionMessage,
    Field(discriminator="role"),
]


class _RawMessageEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", populate_by_name=True)
    type: Literal["message"]
    id: str
    parent_id: str | None = Field(default=None, alias="parentId")
    message: _RawMessage


class _RawCompactionEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", populate_by_name=True)
    type: Literal["compaction"]
    id: str
    first_kept_entry_id: str = Field(alias="firstKeptEntryId")


class _RawSessionHeader(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", populate_by_name=True)
    type: Literal["session"]
    id: str
    cwd: str
    parent_session: str | None = Field(default=None, alias="parentSession")


class _RawEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", populate_by_name=True)
    type: str
    id: str | None = None
    parent_id: str | None = Field(default=None, alias="parentId")


def get_root() -> Path | None:
    """Get the pi sessions root directory.

    Returns:
        Path to the sessions directory if it exists, None otherwise.
    """
    root = Path.home() / ".pi" / "agent" / "sessions"
    return root if root.exists() else None


def _process_user_content(content: list[_UserContent], ts: datetime) -> list[Message]:
    """Process user message content blocks into normalized messages.

    Args:
        content: List of user content blocks (text or image).
        ts: Timestamp for the messages.

    Returns:
        List of UserMessage and ImageAttachmentMessage objects.
    """
    messages: list[Message] = []
    text_parts: list[str] = []
    for block in content:
        match block:
            case _TextContent(text=text) if text.strip():
                text_parts.append(text.strip())
            case _ImageContent(data=data, mime_type=mt):
                messages.append(AttachmentMessage(created_at=ts, media_type=mt, data=data))
            case _:
                pass
    if text_parts:
        messages.append(UserMessage(created_at=ts, content="\n".join(text_parts)))
    return messages


def _process_assistant_content(content: list[_AssistantContent], ts: datetime, model: str) -> list[Message]:
    """Process assistant message content blocks into normalized messages.

    Args:
        content: List of assistant content blocks (text, thinking, or tool call).
        ts: Timestamp for the messages.
        model: Model identifier string.

    Returns:
        List of AssistantMessage and ToolUseMessage objects.
    """
    messages: list[Message] = []
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in content:
        match block:
            case _TextContent(text=text) if text.strip():
                text_parts.append(text.strip())
            case _ThinkingContent(thinking=thinking) if thinking.strip():
                thinking_parts.append(thinking.strip())
            case _ToolCall(id=tid, name=name, arguments=args):
                messages.append(
                    ToolUseMessage(created_at=ts, model=model, tool_id=tid, tool_name=name, parameters=args)
                )
            case _:
                pass
    if text_parts:
        messages.append(AssistantMessage(created_at=ts, model=model, is_thinking=False, content="\n".join(text_parts)))
    if thinking_parts:
        messages.append(
            AssistantMessage(created_at=ts, model=model, is_thinking=True, content="\n".join(thinking_parts))
        )
    return messages


def _process_message(msg: _RawMessage) -> list[Message]:
    """Process a raw message into normalized messages.

    Args:
        msg: Raw message from pi.

    Returns:
        List of normalized Message objects.
    """
    match msg:
        case _RawUserMessage(content=str() as text, timestamp=ts):
            return [UserMessage(created_at=ts, content=text.strip())] if text.strip() else []
        case _RawUserMessage(content=list() as content, timestamp=ts):
            return _process_user_content(content, ts)
        case _RawAssistantMessage(content=content, timestamp=ts, model=model):
            return _process_assistant_content(content, ts, model)
        case _RawToolResultMessage(tool_call_id=tid, content=content, is_error=err, timestamp=ts):
            result = "\n".join(b.get("text", "") for b in content)
            return [ToolResultMessage(created_at=ts, tool_id=tid, result={"content": result, "is_error": err})]
        case _RawBashExecutionMessage(command=cmd, output=out, timestamp=ts):
            return [UserMessage(created_at=ts, content=f"Ran `{cmd}`\n```\n{out}\n```")]
        case _:
            return []


def _read_session_header(jsonl_file: Path) -> _RawSessionHeader | None:
    """Read and parse the session header from a JSONL file.

    Args:
        jsonl_file: Path to the JSONL file.

    Returns:
        Parsed session header, or None if parsing fails.
    """
    try:
        with jsonl_file.open(encoding="utf-8") as f:
            if first_line := f.readline().strip():
                return _RawSessionHeader.model_validate_json(first_line)
    except (OSError, ValidationError) as e:
        logger.debug("Failed to read session header from %s: %s", jsonl_file, e)
    return None


def _find_session_file(session_id: str) -> Path | None:
    """Find the JSONL file for a given session ID.

    Args:
        session_id: The session ID (UUID).

    Returns:
        Path to the JSONL file, or None if not found.
    """
    root = get_root()
    if root is None:
        return None

    for workspace_dir in root.iterdir():
        if not workspace_dir.is_dir():
            continue
        for jsonl_file in workspace_dir.glob("*.jsonl"):
            if (header := _read_session_header(jsonl_file)) and header.id == session_id:
                return jsonl_file
    return None


def _parse_session_metadata(jsonl_path: Path) -> Session | None:
    """Parse a JSONL session file and extract only metadata.

    Args:
        jsonl_path: Path to the JSONL session file.

    Returns:
        Session object with metadata only, or None if parsing fails.
    """
    if header := _read_session_header(jsonl_path):
        return Session(
            id=header.id,
            agent=AGENT_TYPE,
            workspace=Path(header.cwd),
            git_branch=None,
            parent_session_id=header.parent_session,
        )
    return None


def _process_line(
    stripped: str,
    entries: list[_RawEntry],
    message_entries: dict[str, _RawMessageEntry],
) -> str | None:
    """Process a single JSONL line and update entry collections.

    Args:
        stripped: Stripped JSON line from the session file.
        entries: List to append parsed entries to.
        message_entries: Dict to populate with message entries by ID.

    Returns:
        The compaction first_kept_entry_id if this line is a compaction entry, None otherwise.
    """
    try:
        base = _RawEntry.model_validate_json(stripped)
    except ValidationError:
        return None

    if base.type == "session":
        return None
    entries.append(base)

    if base.type == "message" and base.id:
        with suppress(ValidationError):
            message_entries[base.id] = _RawMessageEntry.model_validate_json(stripped)
    elif base.type == "compaction":
        with suppress(ValidationError):
            return _RawCompactionEntry.model_validate_json(stripped).first_kept_entry_id
    return None


def _parse_session_entries(
    jsonl_path: Path,
) -> tuple[list[_RawEntry], dict[str, _RawMessageEntry], str | None] | None:
    """Parse all entries from a JSONL session file.

    Args:
        jsonl_path: Path to the JSONL file.

    Returns:
        Tuple of (entries, message_entries, compaction_first_kept) or None if parsing fails.
    """
    logger.debug("Parsing session entries from %s", jsonl_path)
    entries: list[_RawEntry] = []
    message_entries: dict[str, _RawMessageEntry] = {}
    compaction_first_kept: str | None = None

    try:
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if (stripped := line.strip()) and (r := _process_line(stripped, entries, message_entries)):
                    compaction_first_kept = r
    except OSError as e:
        logger.warning("Failed to read %s: %s", jsonl_path, e)
        return None

    return (entries, message_entries, compaction_first_kept) if entries else None


def _iter_messages_from_file(jsonl_path: Path) -> Iterator[Message]:
    """Iterate over messages in a JSONL session file.

    Pi stores sessions as a tree structure. This function walks from the
    leaf to root and yields messages in chronological order.

    Args:
        jsonl_path: Path to the JSONL file.

    Yields:
        Message objects in chronological order.
    """
    parsed = _parse_session_entries(jsonl_path)
    if parsed is None:
        return
    entries, message_entries, compaction_first_kept = parsed

    by_id = {e.id: e for e in entries if e.id}
    path: list[_RawEntry] = []
    current: _RawEntry | None = entries[-1]
    while current:
        path.insert(0, current)
        current = by_id.get(current.parent_id) if current.parent_id else None

    found_first_kept = compaction_first_kept is None
    for entry in path:
        if not found_first_kept:
            if entry.id == compaction_first_kept:
                found_first_kept = True
            continue
        if entry.type == "message" and entry.id and entry.id in message_entries:
            yield from _process_message(message_entries[entry.id].message)


def get_sessions() -> Iterator[Session]:
    """Get all pi sessions.

    Yields:
        Session objects for all sessions.
    """
    root = get_root()
    if root is None:
        return

    for workspace_dir in root.iterdir():
        if not workspace_dir.is_dir():
            continue
        for jsonl_file in workspace_dir.glob("*.jsonl"):
            if session := _parse_session_metadata(jsonl_file):
                yield session


def get_session_from_id(session_id: str) -> Session | None:
    """Get a session by its ID.

    Args:
        session_id: The session ID.

    Returns:
        The Session if found, None otherwise.
    """
    if session_file := _find_session_file(session_id):
        return _parse_session_metadata(session_file)
    return None


def get_session_from_path(session_path: Path) -> Session | None:
    """Get a session from a file path.

    Args:
        session_path: Path to a session JSONL file.

    Returns:
        The Session if found, None otherwise.
    """
    if session_path.exists() and session_path.suffix == ".jsonl":
        return _parse_session_metadata(session_path)
    return None


def get_messages(session: Session) -> Iterator[Message]:
    """Get all messages for a session in chronological order.

    Args:
        session: The Session to get messages for.

    Yields:
        Message objects in chronological order.
    """
    if session_file := _find_session_file(session.id):
        yield from _iter_messages_from_file(session_file)
