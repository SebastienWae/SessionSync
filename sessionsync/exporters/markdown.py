from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sessionsync.schema import (
    AssistantMessage,
    AttachmentMessage,
    ToolResultMessage,
    ToolUseMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from pathlib import Path

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


def _format_frontmatter(session: Session) -> str:
    """Format session metadata as YAML frontmatter.

    Args:
        session: Session metadata.

    Returns:
        YAML frontmatter string.
    """
    lines = [
        "---",
        f"id: {session.id}",
        f"agent: {session.agent.value}",
        f"workspace: {session.workspace}",
    ]
    if session.git_branch is not None:
        lines.append(f"git_branch: {session.git_branch}")
    if session.parent_session_id is not None:
        lines.append(f"parent_session_id: {session.parent_session_id}")
    lines.append("---")
    return "\n".join(lines)


def _format_message(msg: Message) -> str:
    """Format a single message as markdown.

    Args:
        msg: Message to format.

    Returns:
        Markdown formatted message.
    """
    timestamp = msg.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")

    match msg:
        case UserMessage(content=content):
            return f"## User\n_({timestamp})_\n\n{content}"

        case AssistantMessage(model=model, is_thinking=is_thinking, content=content):
            header = "Thinking" if is_thinking else "Assistant"
            return f"## {header}\n_({timestamp}, {model})_\n\n{content}"

        case ToolUseMessage(model=model, tool_name=tool_name, parameters=parameters):
            params_json = json.dumps(parameters, indent=2)
            return f"## Tool Call: {tool_name}\n_({timestamp}, {model})_\n\n```json\n{params_json}\n```"

        case ToolResultMessage(tool_id=tool_id, result=result):
            result_json = json.dumps(result, indent=2)
            return f"## Tool Result\n_({timestamp}, tool_id: {tool_id})_\n\n```json\n{result_json}\n```"

        case AttachmentMessage(media_type=media_type, data=data, filename=filename):
            if media_type.startswith("image/"):
                return f"## Image\n_({timestamp})_\n\n![Image](data:{media_type};base64,{data})"
            display_name = filename or f"({media_type})"
            return f"## Attachment\n_({timestamp})_\n\n[Attachment: {display_name}]"


def export(session: Session, messages: list[Message], output: Path) -> Path:
    """Export session and messages to a markdown file.

    Args:
        session: Session metadata.
        messages: List of filtered messages.
        output: Output directory path.

    Returns:
        Path to the exported file.
    """
    filename = _get_filename(session, messages, "md")
    filepath = output / filename

    parts = [_format_frontmatter(session), ""]
    for msg in messages:
        parts.extend((_format_message(msg), ""))

    filepath.write_text("\n".join(parts))
    return filepath
