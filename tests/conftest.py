from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sessionsync.schema import (
    AgentType,
    AssistantMessage,
    AttachmentMessage,
    Session,
    ToolResultMessage,
    ToolUseMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue


@pytest.fixture
def sample_datetime() -> datetime:
    """Provide a fixed datetime for consistent testing.

    Returns:
        A fixed UTC datetime for test reproducibility.
    """
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)


@pytest.fixture
def make_session() -> Callable[..., Session]:
    """Factory fixture for creating Session objects.

    Returns:
        A factory function that creates Session objects with defaults.
    """

    def _make(
        session_id: str = "550e8400-e29b-41d4-a716-446655440000",
        agent: AgentType = AgentType.CLAUDE_CODE,
        workspace: Path | None = None,
        git_branch: str | None = "main",
        parent_session_id: str | None = None,
    ) -> Session:
        return Session(
            id=session_id,
            agent=agent,
            workspace=workspace or Path("/test/workspace"),
            git_branch=git_branch,
            parent_session_id=parent_session_id,
        )

    return _make


@pytest.fixture
def make_user_message() -> Callable[..., UserMessage]:
    """Factory fixture for creating UserMessage objects.

    Returns:
        A factory function that creates UserMessage objects with defaults.
    """

    def _make(
        content: str = "Hello, world!",
        created_at: datetime | None = None,
    ) -> UserMessage:
        return UserMessage(
            created_at=created_at or datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            content=content,
        )

    return _make


@pytest.fixture
def make_assistant_message() -> Callable[..., AssistantMessage]:
    """Factory fixture for creating AssistantMessage objects.

    Returns:
        A factory function that creates AssistantMessage objects with defaults.
    """

    def _make(
        content: str = "I can help with that.",
        model: str = "claude-3-opus",
        created_at: datetime | None = None,
        *,
        is_thinking: bool = False,
    ) -> AssistantMessage:
        return AssistantMessage(
            created_at=created_at or datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            model=model,
            is_thinking=is_thinking,
            content=content,
        )

    return _make


@pytest.fixture
def make_tool_use_message() -> Callable[..., ToolUseMessage]:
    """Factory fixture for creating ToolUseMessage objects.

    Returns:
        A factory function that creates ToolUseMessage objects with defaults.
    """

    def _make(
        tool_name: str = "Read",
        tool_id: str = "tool_123",
        parameters: dict[str, JsonValue] | None = None,
        model: str = "claude-3-opus",
        created_at: datetime | None = None,
    ) -> ToolUseMessage:
        return ToolUseMessage(
            created_at=created_at or datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            model=model,
            tool_id=tool_id,
            tool_name=tool_name,
            parameters=parameters or {"file_path": "/test/file.txt"},
        )

    return _make


@pytest.fixture
def make_tool_result_message() -> Callable[..., ToolResultMessage]:
    """Factory fixture for creating ToolResultMessage objects.

    Returns:
        A factory function that creates ToolResultMessage objects with defaults.
    """

    def _make(
        tool_id: str = "tool_123",
        result: dict[str, JsonValue] | None = None,
        created_at: datetime | None = None,
    ) -> ToolResultMessage:
        return ToolResultMessage(
            created_at=created_at or datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            tool_id=tool_id,
            result=result or {"content": "File contents here", "is_error": False},
        )

    return _make


@pytest.fixture
def make_attachment_message() -> Callable[..., AttachmentMessage]:
    """Factory fixture for creating AttachmentMessage objects.

    Returns:
        A factory function that creates AttachmentMessage objects with defaults.
    """

    def _make(
        media_type: str = "image/png",
        data: str = "iVBORw0KGgo=",
        filename: str | None = None,
        created_at: datetime | None = None,
    ) -> AttachmentMessage:
        return AttachmentMessage(
            created_at=created_at or datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            media_type=media_type,
            data=data,
            filename=filename,
        )

    return _make
