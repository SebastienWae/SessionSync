from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sessionsync.parsers.claude_code import (
    _ImageBlock,
    _ImageSource,
    _is_main_session,
    _iter_messages,
    _make_text_message,
    _parse_entry_to_messages,
    _process_blocks,
    _RawContentBlock,
    _RawEntry,
    _RawMessage,
    _TextBlock,
    _ThinkingBlock,
    _ToolResultBlock,
    _ToolUseBlock,
)
from sessionsync.schema import (
    AssistantMessage,
    AttachmentMessage,
    ToolResultMessage,
    ToolUseMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from datetime import datetime


class TestIsMainSession:
    """Tests for _is_main_session function."""

    @pytest.mark.parametrize(
        "stem",
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "550E8400-E29B-41D4-A716-446655440000",  # uppercase
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "00000000-0000-0000-0000-000000000000",  # all zeros
            "ffffffff-ffff-ffff-ffff-ffffffffffff",  # all f's
        ],
    )
    def test_valid_uuid(self, stem: str) -> None:
        """Test that valid UUIDs are recognized as main sessions."""
        assert _is_main_session(stem) is True

    @pytest.mark.parametrize(
        "stem",
        [
            "not-a-uuid",
            "agent-0",
            "agent-1",
            "550e8400-e29b-41d4-a716",  # incomplete
            "550e8400-e29b-41d4-a716-4466554400001",  # too long
            "550e8400-e29b-41d4-a716-44665544000",  # too short
            "550e8400-e29b-41d4-a716-44665544000g",  # invalid char
            "",
            "subagent",
        ],
    )
    def test_invalid_uuid(self, stem: str) -> None:
        """Test that invalid UUIDs are not recognized as main sessions."""
        assert _is_main_session(stem) is False


class TestMakeTextMessage:
    """Tests for _make_text_message function."""

    def test_user_message(self, sample_datetime: datetime) -> None:
        """Test creating a user message."""
        msg = _make_text_message(sample_datetime, "claude-3-opus", "Hello!", is_user=True)

        assert isinstance(msg, UserMessage)
        assert msg.content == "Hello!"
        assert msg.created_at == sample_datetime

    def test_assistant_message(self, sample_datetime: datetime) -> None:
        """Test creating an assistant message."""
        msg = _make_text_message(sample_datetime, "claude-3-opus", "Hi there!", is_user=False)

        assert isinstance(msg, AssistantMessage)
        assert msg.content == "Hi there!"
        assert msg.model == "claude-3-opus"
        assert msg.is_thinking is False
        assert msg.created_at == sample_datetime


class TestProcessBlocks:
    """Tests for _process_blocks function."""

    def test_text_block(self, sample_datetime: datetime) -> None:
        """Test processing a single text block."""
        blocks: list[_RawContentBlock] = [_TextBlock(type="text", text="Hello world")]
        messages, text_parts, thinking_parts = _process_blocks(blocks, sample_datetime, "model")

        assert messages == []
        assert text_parts == ["Hello world"]
        assert thinking_parts == []

    def test_text_block_strips_whitespace(self, sample_datetime: datetime) -> None:
        """Test that text blocks have whitespace stripped."""
        blocks: list[_RawContentBlock] = [_TextBlock(type="text", text="  Hello  ")]
        _messages, text_parts, _thinking_parts = _process_blocks(blocks, sample_datetime, "model")

        assert text_parts == ["Hello"]

    def test_empty_text_block_ignored(self, sample_datetime: datetime) -> None:
        """Test that empty text blocks are ignored."""
        blocks: list[_RawContentBlock] = [_TextBlock(type="text", text="   ")]
        _messages, text_parts, _thinking_parts = _process_blocks(blocks, sample_datetime, "model")

        assert text_parts == []

    def test_thinking_block(self, sample_datetime: datetime) -> None:
        """Test processing a thinking block."""
        blocks: list[_RawContentBlock] = [_ThinkingBlock(type="thinking", thinking="Let me think...")]
        messages, text_parts, thinking_parts = _process_blocks(blocks, sample_datetime, "model")

        assert messages == []
        assert text_parts == []
        assert thinking_parts == ["Let me think..."]

    def test_tool_use_block(self, sample_datetime: datetime) -> None:
        """Test processing a tool use block."""
        blocks: list[_RawContentBlock] = [
            _ToolUseBlock(
                type="tool_use",
                id="tool_123",
                name="Read",
                input={"file_path": "/test.txt"},
            )
        ]
        messages, _text_parts, _thinking_parts = _process_blocks(blocks, sample_datetime, "claude-3")

        assert len(messages) == 1
        assert isinstance(messages[0], ToolUseMessage)
        assert messages[0].tool_id == "tool_123"
        assert messages[0].tool_name == "Read"
        assert messages[0].parameters == {"file_path": "/test.txt"}
        assert messages[0].model == "claude-3"

    def test_tool_result_block_string_content(self, sample_datetime: datetime) -> None:
        """Test processing a tool result block with string content."""
        blocks: list[_RawContentBlock] = [
            _ToolResultBlock(
                type="tool_result",
                tool_use_id="tool_123",
                content="File contents here",
                is_error=False,
            )
        ]
        messages, _text_parts, _thinking_parts = _process_blocks(blocks, sample_datetime, "model")

        assert len(messages) == 1
        assert isinstance(messages[0], ToolResultMessage)
        assert messages[0].tool_id == "tool_123"
        assert messages[0].result == {"content": "File contents here", "is_error": False}

    def test_tool_result_block_with_error(self, sample_datetime: datetime) -> None:
        """Test processing a tool result block with is_error=True."""
        blocks: list[_RawContentBlock] = [
            _ToolResultBlock(
                type="tool_result",
                tool_use_id="tool_123",
                content="Error occurred",
                is_error=True,
            )
        ]
        messages, _, _ = _process_blocks(blocks, sample_datetime, "model")

        assert isinstance(messages[0], ToolResultMessage)
        assert messages[0].result["is_error"] is True

    def test_image_block(self, sample_datetime: datetime) -> None:
        """Test processing an image block."""
        blocks: list[_RawContentBlock] = [
            _ImageBlock(
                type="image",
                source=_ImageSource(type="base64", media_type="image/png", data="iVBORw0KGgo="),
            )
        ]
        messages, _text_parts, _thinking_parts = _process_blocks(blocks, sample_datetime, "model")

        assert len(messages) == 1
        assert isinstance(messages[0], AttachmentMessage)
        assert messages[0].media_type == "image/png"
        assert messages[0].data == "iVBORw0KGgo="

    def test_mixed_blocks(self, sample_datetime: datetime) -> None:
        """Test processing multiple different block types."""
        blocks: list[_RawContentBlock] = [
            _TextBlock(type="text", text="Here's the file:"),
            _ToolUseBlock(type="tool_use", id="t1", name="Read", input={"path": "/a.txt"}),
            _ThinkingBlock(type="thinking", thinking="I should read this file"),
            _TextBlock(type="text", text="Let me explain..."),
        ]
        messages, text_parts, thinking_parts = _process_blocks(blocks, sample_datetime, "model")

        assert len(messages) == 1
        assert isinstance(messages[0], ToolUseMessage)
        assert text_parts == ["Here's the file:", "Let me explain..."]
        assert thinking_parts == ["I should read this file"]


class TestParseEntryToMessages:
    """Tests for _parse_entry_to_messages function."""

    def test_none_message(self, sample_datetime: datetime) -> None:
        """Test entry with no message returns empty list."""
        entry = _RawEntry(type="user", timestamp=sample_datetime, message=None)
        messages = _parse_entry_to_messages(entry, sample_datetime)

        assert messages == []

    def test_none_content(self, sample_datetime: datetime) -> None:
        """Test entry with message but no content returns empty list."""
        entry = _RawEntry(
            type="user",
            timestamp=sample_datetime,
            message=_RawMessage(content=None, model="claude-3"),
        )
        messages = _parse_entry_to_messages(entry, sample_datetime)

        assert messages == []

    def test_string_content_user(self, sample_datetime: datetime) -> None:
        """Test user entry with string content."""
        entry = _RawEntry(
            type="user",
            timestamp=sample_datetime,
            message=_RawMessage(content="Hello!", model="claude-3"),
        )
        messages = _parse_entry_to_messages(entry, sample_datetime)

        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "Hello!"

    def test_string_content_assistant(self, sample_datetime: datetime) -> None:
        """Test assistant entry with string content."""
        entry = _RawEntry(
            type="assistant",
            timestamp=sample_datetime,
            message=_RawMessage(content="Hello!", model="claude-3"),
        )
        messages = _parse_entry_to_messages(entry, sample_datetime)

        assert len(messages) == 1
        assert isinstance(messages[0], AssistantMessage)
        assert messages[0].content == "Hello!"
        assert messages[0].model == "claude-3"

    def test_empty_string_content(self, sample_datetime: datetime) -> None:
        """Test entry with empty string content returns empty list."""
        entry = _RawEntry(
            type="user",
            timestamp=sample_datetime,
            message=_RawMessage(content="   ", model="claude-3"),
        )
        messages = _parse_entry_to_messages(entry, sample_datetime)

        assert messages == []

    def test_block_content_with_text_and_thinking(self, sample_datetime: datetime) -> None:
        """Test entry with text and thinking blocks."""
        entry = _RawEntry(
            type="assistant",
            timestamp=sample_datetime,
            message=_RawMessage(
                content=[
                    _TextBlock(type="text", text="Response here"),
                    _ThinkingBlock(type="thinking", thinking="Internal reasoning"),
                ],
                model="claude-3",
            ),
        )
        messages = _parse_entry_to_messages(entry, sample_datetime)

        assert len(messages) == 2
        assert isinstance(messages[0], AssistantMessage)
        assert messages[0].content == "Response here"
        assert messages[0].is_thinking is False
        assert isinstance(messages[1], AssistantMessage)
        assert messages[1].content == "Internal reasoning"
        assert messages[1].is_thinking is True

    def test_block_content_with_tool_use(self, sample_datetime: datetime) -> None:
        """Test entry with tool use block."""
        entry = _RawEntry(
            type="assistant",
            timestamp=sample_datetime,
            message=_RawMessage(
                content=[
                    _ToolUseBlock(type="tool_use", id="t1", name="Write", input={"path": "/x"}),
                ],
                model="claude-3",
            ),
        )
        messages = _parse_entry_to_messages(entry, sample_datetime)

        assert len(messages) == 1
        assert isinstance(messages[0], ToolUseMessage)

    def test_unknown_model_defaults_to_unknown(self, sample_datetime: datetime) -> None:
        """Test that missing model defaults to 'unknown'."""
        entry = _RawEntry(
            type="assistant",
            timestamp=sample_datetime,
            message=_RawMessage(content="Hello", model=None),
        )
        messages = _parse_entry_to_messages(entry, sample_datetime)

        assert isinstance(messages[0], AssistantMessage)
        assert messages[0].model == "unknown"


class TestIterMessagesIntegration:
    """Integration tests for _iter_messages with real JSONL files."""

    @pytest.fixture
    def sample_session_file(self) -> Path:
        """Get path to the sample session fixture.

        Returns:
            Path to the sample JSONL session file.
        """
        return Path(__file__).parent.parent / "fixtures" / "claude_code" / "sample_session.jsonl"

    def test_parses_sample_session(self, sample_session_file: Path) -> None:
        """Test parsing a complete sample session file."""
        messages = list(_iter_messages(sample_session_file))

        assert len(messages) == 6

    def test_message_types_in_order(self, sample_session_file: Path) -> None:
        """Test that message types are in correct order."""
        messages = list(_iter_messages(sample_session_file))

        assert isinstance(messages[0], UserMessage)
        assert isinstance(messages[1], AssistantMessage)
        assert isinstance(messages[2], UserMessage)
        assert isinstance(messages[3], ToolUseMessage)
        assert isinstance(messages[4], ToolResultMessage)
        assert isinstance(messages[5], AssistantMessage)

    def test_user_message_content(self, sample_session_file: Path) -> None:
        """Test user message content is parsed correctly."""
        messages = list(_iter_messages(sample_session_file))

        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "Hello, can you help me?"

    def test_assistant_message_model(self, sample_session_file: Path) -> None:
        """Test assistant message has correct model."""
        messages = list(_iter_messages(sample_session_file))

        assert isinstance(messages[1], AssistantMessage)
        assert messages[1].model == "claude-3-opus"

    def test_tool_use_details(self, sample_session_file: Path) -> None:
        """Test tool use message has correct details."""
        messages = list(_iter_messages(sample_session_file))

        assert isinstance(messages[3], ToolUseMessage)
        assert messages[3].tool_name == "Read"
        assert messages[3].tool_id == "tool_1"

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        """Test that nonexistent file returns empty iterator."""
        messages = list(_iter_messages(tmp_path / "nonexistent.jsonl"))

        assert messages == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        """Test that empty file returns empty iterator."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        messages = list(_iter_messages(empty_file))

        assert messages == []
