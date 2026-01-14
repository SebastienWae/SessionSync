from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from sessionsync.cli import (
    ENABLED_PARSERS,
    ParserName,
    _filter_messages,
    _filter_sessions,
    _get_parsers,
    _sync_session,
)
from sessionsync.schema import (
    AssistantMessage,
    AttachmentMessage,
    Session,
    ToolResultMessage,
    ToolUseMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestGetParsers:
    """Tests for _get_parsers function."""

    def test_all_returns_all_parsers(self) -> None:
        """Test that 'all' returns all enabled parsers."""
        result = list(_get_parsers("all"))

        assert len(result) == len(ENABLED_PARSERS)

    def test_specific_agent_returns_single_parser(self) -> None:
        """Test that a specific agent name returns only that parser."""
        result = list(_get_parsers("claude"))

        assert len(result) == 1
        assert result[0] is ENABLED_PARSERS["claude"]

    @pytest.mark.parametrize("agent", ["claude", "opencode", "pi"])
    def test_each_agent_returns_correct_parser(self, agent: str) -> None:
        """Test that each valid agent name returns its parser."""
        agent_name = cast("ParserName", agent)
        result = list(_get_parsers(agent_name))

        assert len(result) == 1
        assert result[0] is ENABLED_PARSERS[agent_name]


class TestFilterSessions:
    """Tests for _filter_sessions function."""

    def test_filter_by_workspace(self, make_session: Callable[..., Session]) -> None:
        """Test filtering sessions by workspace."""
        target_workspace = Path("/project/a")
        sessions = iter([
            make_session(session_id="1", workspace=target_workspace),
            make_session(session_id="2", workspace=Path("/project/b")),
            make_session(session_id="3", workspace=target_workspace),
        ])

        result = list(
            _filter_sessions(
                sessions,
                workspace=target_workspace,
                branch=None,
                include_subagents=True,
            )
        )

        assert len(result) == 2
        assert all(s.workspace == target_workspace for s in result)

    def test_filter_by_branch(self, make_session: Callable[..., Session]) -> None:
        """Test filtering sessions by git branch."""
        sessions = iter([
            make_session(session_id="1", git_branch="main"),
            make_session(session_id="2", git_branch="develop"),
            make_session(session_id="3", git_branch="main"),
        ])

        result = list(
            _filter_sessions(
                sessions,
                workspace=None,
                branch="main",
                include_subagents=True,
            )
        )

        assert len(result) == 2
        assert all(s.git_branch == "main" for s in result)

    def test_none_branch_included_when_filtering(self, make_session: Callable[..., Session]) -> None:
        """Test that sessions with None branch are included when branch filter is set."""
        sessions = iter([
            make_session(session_id="1", git_branch="main"),
            make_session(session_id="2", git_branch=None),
        ])

        result = list(
            _filter_sessions(
                sessions,
                workspace=None,
                branch="main",
                include_subagents=True,
            )
        )

        assert len(result) == 2

    def test_none_branch_included_when_no_filter(self, make_session: Callable[..., Session]) -> None:
        """Test that sessions with None branch are included when branch filter is None."""
        sessions = iter([
            make_session(session_id="1", git_branch="main"),
            make_session(session_id="2", git_branch=None),
        ])

        result = list(
            _filter_sessions(
                sessions,
                workspace=None,
                branch=None,
                include_subagents=True,
            )
        )

        assert len(result) == 2

    def test_exclude_subagents(self, make_session: Callable[..., Session]) -> None:
        """Test excluding subagent sessions."""
        sessions = iter([
            make_session(session_id="1", parent_session_id=None),
            make_session(session_id="2", parent_session_id="1"),
            make_session(session_id="3", parent_session_id=None),
        ])

        result = list(
            _filter_sessions(
                sessions,
                workspace=None,
                branch=None,
                include_subagents=False,
            )
        )

        assert len(result) == 2
        assert all(s.parent_session_id is None for s in result)

    def test_include_subagents(self, make_session: Callable[..., Session]) -> None:
        """Test including subagent sessions."""
        sessions = iter([
            make_session(session_id="1", parent_session_id=None),
            make_session(session_id="2", parent_session_id="1"),
        ])

        result = list(
            _filter_sessions(
                sessions,
                workspace=None,
                branch=None,
                include_subagents=True,
            )
        )

        assert len(result) == 2

    def test_combined_filters(self, make_session: Callable[..., Session]) -> None:
        """Test combining workspace, branch, and subagent filters."""
        target = Path("/project")
        sessions = iter([
            make_session(session_id="1", workspace=target, git_branch="main", parent_session_id=None),
            make_session(session_id="2", workspace=target, git_branch="develop", parent_session_id=None),
            make_session(session_id="3", workspace=Path("/other"), git_branch="main", parent_session_id=None),
            make_session(session_id="4", workspace=target, git_branch="main", parent_session_id="1"),
        ])

        result = list(
            _filter_sessions(
                sessions,
                workspace=target,
                branch="main",
                include_subagents=False,
            )
        )

        assert len(result) == 1
        assert result[0].id == "1"


class TestFilterMessages:
    """Tests for _filter_messages function."""

    def test_exclude_tool_messages(
        self,
        make_user_message: Callable[..., UserMessage],
        make_assistant_message: Callable[..., AssistantMessage],
        make_tool_use_message: Callable[..., ToolUseMessage],
        make_tool_result_message: Callable[..., ToolResultMessage],
    ) -> None:
        """Test excluding tool use and tool result messages."""
        messages = iter([
            make_user_message(),
            make_assistant_message(),
            make_tool_use_message(),
            make_tool_result_message(),
        ])

        result = list(
            _filter_messages(
                messages,
                include_tools=False,
                include_thinking=True,
                include_attachments=True,
            )
        )

        assert len(result) == 2
        assert isinstance(result[0], UserMessage)
        assert isinstance(result[1], AssistantMessage)

    def test_include_tool_messages(
        self,
        make_tool_use_message: Callable[..., ToolUseMessage],
        make_tool_result_message: Callable[..., ToolResultMessage],
    ) -> None:
        """Test including tool messages."""
        messages = iter([
            make_tool_use_message(),
            make_tool_result_message(),
        ])

        result = list(
            _filter_messages(
                messages,
                include_tools=True,
                include_thinking=True,
                include_attachments=True,
            )
        )

        assert len(result) == 2

    def test_exclude_thinking_messages(
        self,
        make_assistant_message: Callable[..., AssistantMessage],
    ) -> None:
        """Test excluding thinking messages."""
        messages = iter([
            make_assistant_message(is_thinking=False),
            make_assistant_message(is_thinking=True),
        ])

        result = list(
            _filter_messages(
                messages,
                include_tools=True,
                include_thinking=False,
                include_attachments=True,
            )
        )

        assert len(result) == 1
        assert isinstance(result[0], AssistantMessage)
        assert result[0].is_thinking is False

    def test_include_thinking_messages(
        self,
        make_assistant_message: Callable[..., AssistantMessage],
    ) -> None:
        """Test including thinking messages."""
        messages = iter([
            make_assistant_message(is_thinking=True),
        ])

        result = list(
            _filter_messages(
                messages,
                include_tools=True,
                include_thinking=True,
                include_attachments=True,
            )
        )

        assert len(result) == 1

    def test_exclude_attachment_messages(
        self,
        make_user_message: Callable[..., UserMessage],
        make_attachment_message: Callable[..., AttachmentMessage],
    ) -> None:
        """Test excluding attachment messages."""
        messages = iter([
            make_user_message(),
            make_attachment_message(),
        ])

        result = list(
            _filter_messages(
                messages,
                include_tools=True,
                include_thinking=True,
                include_attachments=False,
            )
        )

        assert len(result) == 1
        assert isinstance(result[0], UserMessage)

    def test_include_attachment_messages(
        self,
        make_attachment_message: Callable[..., AttachmentMessage],
    ) -> None:
        """Test including attachment messages."""
        messages = iter([
            make_attachment_message(),
        ])

        result = list(
            _filter_messages(
                messages,
                include_tools=True,
                include_thinking=True,
                include_attachments=True,
            )
        )

        assert len(result) == 1

    def test_include_all(
        self,
        make_user_message: Callable[..., UserMessage],
        make_assistant_message: Callable[..., AssistantMessage],
        make_tool_use_message: Callable[..., ToolUseMessage],
        make_tool_result_message: Callable[..., ToolResultMessage],
        make_attachment_message: Callable[..., AttachmentMessage],
    ) -> None:
        """Test including all message types."""
        messages = iter([
            make_user_message(),
            make_assistant_message(is_thinking=False),
            make_assistant_message(is_thinking=True),
            make_tool_use_message(),
            make_tool_result_message(),
            make_attachment_message(),
        ])

        result = list(
            _filter_messages(
                messages,
                include_tools=True,
                include_thinking=True,
                include_attachments=True,
            )
        )

        assert len(result) == 6

    def test_exclude_all_optional(
        self,
        make_user_message: Callable[..., UserMessage],
        make_assistant_message: Callable[..., AssistantMessage],
        make_tool_use_message: Callable[..., ToolUseMessage],
        make_tool_result_message: Callable[..., ToolResultMessage],
        make_attachment_message: Callable[..., AttachmentMessage],
    ) -> None:
        """Test excluding all optional message types."""
        messages = iter([
            make_user_message(),
            make_assistant_message(is_thinking=False),
            make_assistant_message(is_thinking=True),
            make_tool_use_message(),
            make_tool_result_message(),
            make_attachment_message(),
        ])

        result = list(
            _filter_messages(
                messages,
                include_tools=False,
                include_thinking=False,
                include_attachments=False,
            )
        )

        assert len(result) == 2
        assert isinstance(result[0], UserMessage)
        assert isinstance(result[1], AssistantMessage)
        assert result[1].is_thinking is False


class TestSyncSession:
    """Tests for _sync_session function."""

    def test_skips_empty_session(
        self,
        tmp_path: Path,
        make_session: Callable[..., Session],
    ) -> None:
        """Test that sessions with no messages are skipped."""
        from unittest.mock import MagicMock

        from sessionsync.exporters.base import Exporter
        from sessionsync.parsers.base import Parser

        session = make_session()
        output = tmp_path / "output"
        output.mkdir()

        mock_parser = MagicMock(spec=Parser)
        mock_parser.get_messages.return_value = iter([])  # pyright: ignore[reportAny]

        mock_exporter: Exporter = MagicMock(spec=Exporter)

        _sync_session(
            mock_parser,
            session,
            output,
            mock_exporter,
            include_tools=True,
            include_thinking=True,
            include_attachments=True,
        )

        mock_exporter.export.assert_not_called()  # pyright: ignore[reportAny]
