from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sessionsync.exporters.markdown import _format_frontmatter, _format_message, _get_filename, export
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
    from collections.abc import Callable
    from pathlib import Path


class TestGetFilename:
    """Tests for _get_filename function."""

    def test_main_session_filename(
        self,
        make_session: Callable[..., Session],
        make_user_message: Callable[..., UserMessage],
    ) -> None:
        """Test filename generation for a main session."""
        session = make_session(session_id="550e8400-e29b-41d4-a716-446655440000")
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        messages: list[Message] = [make_user_message(created_at=dt)]
        expected_dt = dt.astimezone().strftime("%Y-%m-%dT%H-%M-%S")

        filename = _get_filename(session, messages, "md")

        assert filename == f"{expected_dt}_550e8400-e29b-41d4-a716-446655440000.md"

    def test_subagent_session_filename(
        self,
        make_session: Callable[..., Session],
        make_user_message: Callable[..., UserMessage],
    ) -> None:
        """Test filename generation for a subagent session."""
        session = make_session(
            session_id="parent-uuid:agent-0",
            parent_session_id="parent-uuid",
        )
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        messages: list[Message] = [make_user_message(created_at=dt)]
        expected_dt = dt.astimezone().strftime("%Y-%m-%dT%H-%M-%S")

        filename = _get_filename(session, messages, "md")

        assert filename == f"{expected_dt}_parent-uuid_agent-0.md"

    def test_empty_messages_uses_unknown(
        self,
        make_session: Callable[..., Session],
    ) -> None:
        """Test filename generation with no messages."""
        session = make_session(session_id="test-session")
        messages: list[Message] = []

        filename = _get_filename(session, messages, "md")

        assert filename == "unknown_test-session.md"

    def test_different_extension(
        self,
        make_session: Callable[..., Session],
        make_user_message: Callable[..., UserMessage],
    ) -> None:
        """Test filename with different extension."""
        session = make_session(session_id="test")
        messages: list[Message] = [make_user_message(created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC))]

        filename = _get_filename(session, messages, "json")

        assert filename.endswith(".json")


class TestFormatFrontmatter:
    """Tests for _format_frontmatter function."""

    def test_basic_frontmatter(self, make_session: Callable[..., Session]) -> None:
        """Test basic frontmatter generation."""
        session = make_session(
            session_id="test-id",
            agent=AgentType.CLAUDE_CODE,
            git_branch="main",
        )

        result = _format_frontmatter(session)

        assert "---" in result
        assert "id: test-id" in result
        assert "agent: claude_code" in result
        assert "workspace:" in result
        assert "git_branch: main" in result

    def test_frontmatter_without_branch(self, make_session: Callable[..., Session]) -> None:
        """Test frontmatter without git branch."""
        session = make_session(git_branch=None)

        result = _format_frontmatter(session)

        assert "git_branch:" not in result

    def test_frontmatter_with_parent_session(self, make_session: Callable[..., Session]) -> None:
        """Test frontmatter with parent session ID."""
        session = make_session(parent_session_id="parent-123")

        result = _format_frontmatter(session)

        assert "parent_session_id: parent-123" in result

    def test_frontmatter_structure(self, make_session: Callable[..., Session]) -> None:
        """Test that frontmatter has correct YAML structure."""
        session = make_session()

        result = _format_frontmatter(session)
        lines = result.split("\n")

        assert lines[0] == "---"
        assert lines[-1] == "---"


class TestFormatMessage:
    """Tests for _format_message function."""

    def test_user_message(self, make_user_message: Callable[..., UserMessage]) -> None:
        """Test formatting a user message."""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        msg = make_user_message(
            content="Hello, world!",
            created_at=dt,
        )
        expected_ts = dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")

        result = _format_message(msg)

        assert "## User" in result
        assert f"({expected_ts})" in result
        assert "Hello, world!" in result

    def test_assistant_message(self, make_assistant_message: Callable[..., AssistantMessage]) -> None:
        """Test formatting an assistant message."""
        msg = make_assistant_message(
            content="I can help!",
            model="claude-3-opus",
            is_thinking=False,
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )

        result = _format_message(msg)

        assert "## Assistant" in result
        assert "claude-3-opus" in result
        assert "I can help!" in result

    def test_thinking_message(self, make_assistant_message: Callable[..., AssistantMessage]) -> None:
        """Test formatting a thinking message."""
        msg = make_assistant_message(
            content="Let me think...",
            is_thinking=True,
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )

        result = _format_message(msg)

        assert "## Thinking" in result
        assert "Let me think..." in result

    def test_tool_use_message(self, make_tool_use_message: Callable[..., ToolUseMessage]) -> None:
        """Test formatting a tool use message."""
        msg = make_tool_use_message(
            tool_name="Read",
            parameters={"file_path": "/test.txt"},
            model="claude-3",
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )

        result = _format_message(msg)

        assert "## Tool Call: Read" in result
        assert "claude-3" in result
        assert "```json" in result
        assert "file_path" in result

    def test_tool_result_message(self, make_tool_result_message: Callable[..., ToolResultMessage]) -> None:
        """Test formatting a tool result message."""
        msg = make_tool_result_message(
            tool_id="tool_123",
            result={"content": "File contents", "is_error": False},
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )

        result = _format_message(msg)

        assert "## Tool Result" in result
        assert "tool_id: tool_123" in result
        assert "```json" in result
        assert "File contents" in result

    def test_image_attachment_message(self, make_attachment_message: Callable[..., AttachmentMessage]) -> None:
        """Test formatting an image attachment message."""
        msg = make_attachment_message(
            media_type="image/png",
            data="iVBORw0KGgo=",
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )

        result = _format_message(msg)

        assert "## Image" in result
        assert "![Image]" in result
        assert "data:image/png;base64,iVBORw0KGgo=" in result

    def test_file_attachment_message(self, make_attachment_message: Callable[..., AttachmentMessage]) -> None:
        """Test formatting a non-image file attachment."""
        msg = make_attachment_message(
            media_type="application/pdf",
            data="JVBERi0=",
            filename="document.pdf",
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )

        result = _format_message(msg)

        assert "## Attachment" in result
        assert "document.pdf" in result

    def test_file_attachment_without_filename(self, make_attachment_message: Callable[..., AttachmentMessage]) -> None:
        """Test formatting a file attachment without filename."""
        msg = make_attachment_message(
            media_type="application/pdf",
            data="JVBERi0=",
            filename=None,
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )

        result = _format_message(msg)

        assert "## Attachment" in result
        assert "(application/pdf)" in result


class TestExportIntegration:
    """Integration tests for the export function."""

    def test_export_creates_file(
        self,
        tmp_path: Path,
        make_session: Callable[..., Session],
        make_user_message: Callable[..., UserMessage],
        make_assistant_message: Callable[..., AssistantMessage],
    ) -> None:
        """Test that export creates a file in the output directory."""
        session = make_session(session_id="test-session")
        messages: list[Message] = [
            make_user_message(content="Hello"),
            make_assistant_message(content="Hi there!"),
        ]

        filepath = export(session, messages, tmp_path)

        assert filepath.exists()
        assert filepath.parent == tmp_path
        assert filepath.suffix == ".md"

    def test_export_file_contains_frontmatter(
        self,
        tmp_path: Path,
        make_session: Callable[..., Session],
        make_user_message: Callable[..., UserMessage],
    ) -> None:
        """Test that exported file contains YAML frontmatter."""
        session = make_session(session_id="test-session", git_branch="main")
        messages: list[Message] = [make_user_message()]

        filepath = export(session, messages, tmp_path)
        content = filepath.read_text()

        assert content.startswith("---")
        assert "id: test-session" in content
        assert "git_branch: main" in content

    def test_export_file_contains_messages(
        self,
        tmp_path: Path,
        make_session: Callable[..., Session],
        make_user_message: Callable[..., UserMessage],
        make_assistant_message: Callable[..., AssistantMessage],
    ) -> None:
        """Test that exported file contains formatted messages."""
        session = make_session()
        messages: list[Message] = [
            make_user_message(content="Test user message"),
            make_assistant_message(content="Test assistant message"),
        ]

        filepath = export(session, messages, tmp_path)
        content = filepath.read_text()

        assert "## User" in content
        assert "Test user message" in content
        assert "## Assistant" in content
        assert "Test assistant message" in content

    def test_export_empty_messages(
        self,
        tmp_path: Path,
        make_session: Callable[..., Session],
    ) -> None:
        """Test exporting a session with no messages."""
        session = make_session()
        messages: list[Message] = []

        filepath = export(session, messages, tmp_path)
        content = filepath.read_text()

        assert "---" in content
        assert "## User" not in content
