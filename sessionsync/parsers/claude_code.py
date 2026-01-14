import logging
import re
from datetime import datetime  # noqa: TC003
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

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
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class _TextBlock(BaseModel):
    type: Literal["text"]
    text: str


class _ThinkingBlock(BaseModel):
    type: Literal["thinking"]
    thinking: str


class _ImageSource(BaseModel):
    type: str
    media_type: str
    data: str


class _ImageBlock(BaseModel):
    type: Literal["image"]
    source: _ImageSource


class _ToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, JsonValue]


class _ToolResultTextBlock(BaseModel):
    type: Literal["text"]
    text: str


class _ToolResultBlock(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[_ToolResultTextBlock]
    is_error: bool = False


_RawContentBlock = Annotated[
    _TextBlock | _ThinkingBlock | _ImageBlock | _ToolUseBlock | _ToolResultBlock,
    Field(discriminator="type"),
]


class _RawMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    content: str | list[_RawContentBlock] | None = None
    model: str | None = None


class _RawEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    type: str | None = None
    timestamp: datetime | None = None
    cwd: str | None = None
    git_branch: str | None = Field(default=None, alias="gitBranch")
    message: _RawMessage | None = None
    is_api_error_message: bool = Field(default=False, alias="isApiErrorMessage")


def _is_main_session(stem: str) -> bool:
    """Check if a filename stem represents a main session (UUID format).

    Args:
        stem: Filename stem without extension.

    Returns:
        True if the stem is a valid UUID format.
    """
    return bool(_UUID_PATTERN.match(stem))


def get_root() -> Path | None:
    """Get the Claude Code projects root directory.

    Returns:
        Path to the ~/.claude/projects directory, or None if not found.
    """
    root = Path.home() / ".claude" / "projects"
    return root if root.exists() else None


def _find_session_file(session_id: str) -> Path | None:
    """Find the JSONL file for a given session ID.

    Args:
        session_id: The session ID (UUID for main, UUID:agent-X for subagent).

    Returns:
        Path to the JSONL file, or None if not found.
    """
    root = get_root()
    if root is None:
        return None

    if ":" in session_id:
        parent_id, agent_id = session_id.split(":", 1)
        for workspace_dir in root.iterdir():
            if not workspace_dir.is_dir():
                continue
            subagent_file = workspace_dir / parent_id / "subagents" / f"{agent_id}.jsonl"
            if subagent_file.exists():
                return subagent_file
    else:
        for workspace_dir in root.iterdir():
            if not workspace_dir.is_dir():
                continue
            session_file = workspace_dir / f"{session_id}.jsonl"
            if session_file.exists():
                return session_file

    return None


def _make_text_message(created_at: datetime, model: str, content: str, *, is_user: bool) -> Message:
    """Create a text message (user or assistant).

    Args:
        created_at: Timestamp for the message.
        model: Model identifier string.
        content: Text content of the message.
        is_user: Whether this is a user message.

    Returns:
        UserMessage if is_user is True, otherwise AssistantMessage.
    """
    if is_user:
        return UserMessage(created_at=created_at, content=content)
    return AssistantMessage(created_at=created_at, model=model, is_thinking=False, content=content)


def _process_blocks(
    content: list[_RawContentBlock], created_at: datetime, model: str
) -> tuple[list[Message], list[str], list[str]]:
    """Process raw content blocks into messages and text parts.

    Args:
        content: List of raw content blocks from Claude Code.
        created_at: Timestamp for the messages.
        model: Model identifier string.

    Returns:
        Tuple of (messages, text_parts, thinking_parts).
    """
    messages: list[Message] = []
    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for block in content:
        match block:
            case _TextBlock(text=text):
                if stripped := text.strip():
                    text_parts.append(stripped)
            case _ThinkingBlock(thinking=thinking):
                if stripped := thinking.strip():
                    thinking_parts.append(stripped)
            case _ToolUseBlock(id=tool_id, name=name, input=params):
                messages.append(
                    ToolUseMessage(
                        created_at=created_at, model=model, tool_id=tool_id, tool_name=name, parameters=params
                    )
                )
            case _ToolResultBlock(tool_use_id=tool_id, content=raw_content, is_error=is_error):
                result = raw_content if isinstance(raw_content, str) else "\n".join(b.text for b in raw_content)
                messages.append(
                    ToolResultMessage(
                        created_at=created_at, tool_id=tool_id, result={"content": result, "is_error": is_error}
                    )
                )
            case _ImageBlock(source=source):
                messages.append(
                    AttachmentMessage(created_at=created_at, media_type=source.media_type, data=source.data)
                )

    return (messages, text_parts, thinking_parts)


def _parse_entry_to_messages(entry: _RawEntry, created_at: datetime) -> list[Message]:
    """Parse a raw JSONL entry into normalized messages.

    Args:
        entry: Raw entry from the JSONL file.
        created_at: Timestamp for the messages.

    Returns:
        List of normalized Message objects.
    """
    if entry.message is None or entry.message.content is None:
        return []

    content = entry.message.content
    model = entry.message.model or "unknown"
    is_user = entry.type == "user"

    if isinstance(content, str):
        if not (stripped := content.strip()):
            return []
        return [_make_text_message(created_at, model, stripped, is_user=is_user)]

    messages, text_parts, thinking_parts = _process_blocks(content, created_at, model)

    if text_parts:
        messages.append(_make_text_message(created_at, model, "\n".join(text_parts), is_user=is_user))

    if thinking_parts:
        messages.append(
            AssistantMessage(created_at=created_at, model=model, is_thinking=True, content="\n".join(thinking_parts))
        )

    return messages


def _iter_messages(jsonl_path: Path) -> Iterator[Message]:
    """Iterate over messages in a JSONL session file.

    Args:
        jsonl_path: Path to the JSONL file.

    Yields:
        Message objects in chronological order.
    """
    logger.debug("Loading messages from %s", jsonl_path)
    try:
        with jsonl_path.open(encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, 1):
                if not (stripped := raw_line.strip()):
                    continue
                try:
                    entry = _RawEntry.model_validate_json(stripped)
                except ValidationError as e:
                    logger.debug("Skipped line %d in %s: %s", line_num, jsonl_path.name, e)
                    continue

                if entry.type not in {"user", "assistant"} or entry.is_api_error_message or entry.timestamp is None:
                    continue

                yield from _parse_entry_to_messages(entry, entry.timestamp)
    except OSError as e:
        logger.warning("Failed to read %s: %s", jsonl_path, e)
        return


def _parse_session_metadata(
    jsonl_path: Path,
    session_id: str,
    parent_session_id: str | None = None,
) -> Session | None:
    """Parse a JSONL session file and extract only metadata.

    Args:
        jsonl_path: Path to the JSONL file.
        session_id: ID to assign to the session.
        parent_session_id: Optional parent session ID for subagents.

    Returns:
        Session object with metadata only, or None if parsing fails.
    """
    workspace: Path | None = None
    git_branch: str | None = None

    try:
        with jsonl_path.open(encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, 1):
                if not (stripped := raw_line.strip()):
                    continue
                try:
                    entry = _RawEntry.model_validate_json(stripped)
                except ValidationError as e:
                    logger.debug("Skipped line %d in %s: %s", line_num, jsonl_path.name, e)
                    continue

                if workspace is None and entry.cwd:
                    workspace = Path(entry.cwd)
                if git_branch is None:
                    git_branch = entry.git_branch

                if workspace is not None and git_branch is not None:
                    break
    except OSError as e:
        logger.warning("Failed to read %s: %s", jsonl_path, e)
        return None

    if workspace is None:
        logger.debug("No workspace found in %s", jsonl_path)
        return None

    return Session(
        id=session_id,
        agent=AgentType.CLAUDE_CODE,
        workspace=workspace,
        git_branch=git_branch,
        parent_session_id=parent_session_id,
    )


def _iter_subagent_sessions(workspace_dir: Path, parent_id: str) -> Iterator[Session]:
    """Iterate over subagent sessions for a parent session.

    Args:
        workspace_dir: Path to the workspace directory.
        parent_id: ID of the parent session.

    Yields:
        Session objects for each subagent.
    """
    subagents_dir = workspace_dir / parent_id / "subagents"
    if not subagents_dir.is_dir():
        return

    for subagent_file in subagents_dir.glob("agent-*.jsonl"):
        session = _parse_session_metadata(
            subagent_file, session_id=f"{parent_id}:{subagent_file.stem}", parent_session_id=parent_id
        )
        if session is not None:
            yield session


def get_sessions() -> Iterator[Session]:
    """Get all Claude Code sessions.

    Yields:
        Session objects for all main sessions and subagents.
    """
    root = get_root()
    if root is None:
        return

    for workspace_dir in root.iterdir():
        if not workspace_dir.is_dir():
            continue

        for jsonl_file in workspace_dir.glob("*.jsonl"):
            stem = jsonl_file.stem
            session = _parse_session_metadata(jsonl_file, session_id=stem)

            if session is not None:
                yield session

            if _is_main_session(stem):
                yield from _iter_subagent_sessions(workspace_dir, stem)


def get_session_from_id(session_id: str) -> Session | None:
    """Get a session by its ID.

    Args:
        session_id: The session ID (UUID for main, UUID:agent-X for subagent).

    Returns:
        The Session if found, None otherwise.
    """
    session_file = _find_session_file(session_id)
    if session_file is None:
        return None

    parent_session_id = session_id.split(":", 1)[0] if ":" in session_id else None
    return _parse_session_metadata(session_file, session_id, parent_session_id)


def get_session_from_path(session_path: Path) -> Session | None:
    """Get a session from a file path.

    Args:
        session_path: Path to a session JSONL file.

    Returns:
        The Session if found, None otherwise.
    """
    if not session_path.exists() or session_path.suffix != ".jsonl":
        return None

    if session_path.parent.name == "subagents":
        parent_id = session_path.parent.parent.name
        session_id = f"{parent_id}:{session_path.stem}"
        return _parse_session_metadata(session_path, session_id, parent_session_id=parent_id)

    return _parse_session_metadata(session_path, session_path.stem)


def get_messages(session: Session) -> Iterator[Message]:
    """Get all messages for a session in chronological order.

    Args:
        session: The Session to get messages for.

    Yields:
        Message objects in chronological order.
    """
    session_file = _find_session_file(session.id)
    if session_file is None:
        return

    yield from _iter_messages(session_file)
