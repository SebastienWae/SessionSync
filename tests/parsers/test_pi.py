import json
from datetime import UTC, datetime
from typing import ClassVar

from sessionsync.parsers.pi import (
    _AssistantContent,
    _ImageContent,
    _parse_unix_ms,
    _process_assistant_content,
    _process_line,
    _process_message,
    _process_user_content,
    _RawAssistantMessage,
    _RawBashExecutionMessage,
    _RawEntry,
    _RawMessageEntry,
    _RawToolResultMessage,
    _RawUserMessage,
    _TextContent,
    _ThinkingContent,
    _ToolCall,
    _UserContent,
)
from sessionsync.schema import (
    AssistantMessage,
    AttachmentMessage,
    ToolResultMessage,
    ToolUseMessage,
    UserMessage,
)


class TestParseUnixMs:
    """Tests for _parse_unix_ms function."""

    def test_converts_milliseconds_to_datetime(self) -> None:
        """Test converting Unix milliseconds to datetime."""
        # 2024-01-15 10:30:00 UTC in milliseconds
        ms = 1705314600000
        result = _parse_unix_ms(ms)

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo == UTC

    def test_zero_timestamp(self) -> None:
        """Test converting zero timestamp (Unix epoch)."""
        result = _parse_unix_ms(0)

        assert result.year == 1970
        assert result.month == 1
        assert result.day == 1


class TestProcessUserContent:
    """Tests for _process_user_content function."""

    def test_text_only(self, sample_datetime: datetime) -> None:
        """Test processing user content with only text."""
        content: list[_UserContent] = [_TextContent(type="text", text="Hello, world!")]
        messages = _process_user_content(content, sample_datetime)

        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "Hello, world!"

    def test_multiple_text_blocks_joined(self, sample_datetime: datetime) -> None:
        """Test that multiple text blocks are joined with newlines."""
        content: list[_UserContent] = [
            _TextContent(type="text", text="First line"),
            _TextContent(type="text", text="Second line"),
        ]
        messages = _process_user_content(content, sample_datetime)

        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "First line\nSecond line"

    def test_text_whitespace_stripped(self, sample_datetime: datetime) -> None:
        """Test that text content is stripped of whitespace."""
        content: list[_UserContent] = [_TextContent(type="text", text="  Hello  ")]
        messages = _process_user_content(content, sample_datetime)

        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "Hello"

    def test_empty_text_ignored(self, sample_datetime: datetime) -> None:
        """Test that empty text blocks are ignored."""
        content: list[_UserContent] = [_TextContent(type="text", text="   ")]
        messages = _process_user_content(content, sample_datetime)

        assert messages == []

    def test_image_content(self, sample_datetime: datetime) -> None:
        """Test processing user content with an image."""
        content: list[_UserContent] = [_ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png")]
        messages = _process_user_content(content, sample_datetime)

        assert len(messages) == 1
        assert isinstance(messages[0], AttachmentMessage)
        assert messages[0].media_type == "image/png"
        assert messages[0].data == "iVBORw0KGgo="

    def test_mixed_text_and_images(self, sample_datetime: datetime) -> None:
        """Test processing user content with both text and images."""
        content: list[_UserContent] = [
            _TextContent(type="text", text="Here's an image:"),
            _ImageContent(type="image", data="abc123", mimeType="image/jpeg"),
        ]
        messages = _process_user_content(content, sample_datetime)

        assert len(messages) == 2
        assert isinstance(messages[0], AttachmentMessage)
        assert isinstance(messages[1], UserMessage)


class TestProcessAssistantContent:
    """Tests for _process_assistant_content function."""

    def test_text_only(self, sample_datetime: datetime) -> None:
        """Test processing assistant content with only text."""
        content: list[_AssistantContent] = [_TextContent(type="text", text="I can help with that.")]
        messages = _process_assistant_content(content, sample_datetime, "claude-3")

        assert len(messages) == 1
        assert isinstance(messages[0], AssistantMessage)
        assert messages[0].content == "I can help with that."
        assert messages[0].model == "claude-3"
        assert messages[0].is_thinking is False

    def test_thinking_content(self, sample_datetime: datetime) -> None:
        """Test processing assistant content with thinking."""
        content: list[_AssistantContent] = [_ThinkingContent(type="thinking", thinking="Let me analyze this...")]
        messages = _process_assistant_content(content, sample_datetime, "claude-3")

        assert len(messages) == 1
        assert isinstance(messages[0], AssistantMessage)
        assert messages[0].content == "Let me analyze this..."
        assert messages[0].is_thinking is True

    def test_tool_call(self, sample_datetime: datetime) -> None:
        """Test processing assistant content with a tool call."""
        content: list[_AssistantContent] = [
            _ToolCall(
                type="toolCall",
                id="tool_123",
                name="Read",
                arguments={"file_path": "/test.txt"},
            )
        ]
        messages = _process_assistant_content(content, sample_datetime, "claude-3")

        assert len(messages) == 1
        assert isinstance(messages[0], ToolUseMessage)
        assert messages[0].tool_id == "tool_123"
        assert messages[0].tool_name == "Read"
        assert messages[0].parameters == {"file_path": "/test.txt"}
        assert messages[0].model == "claude-3"

    def test_mixed_content(self, sample_datetime: datetime) -> None:
        """Test processing mixed assistant content."""
        content: list[_AssistantContent] = [
            _ThinkingContent(type="thinking", thinking="I should read the file"),
            _TextContent(type="text", text="Let me read that file for you."),
            _ToolCall(type="toolCall", id="t1", name="Read", arguments={"path": "/x"}),
        ]
        messages = _process_assistant_content(content, sample_datetime, "claude-3")

        assert len(messages) == 3
        assert isinstance(messages[0], ToolUseMessage)
        assert isinstance(messages[1], AssistantMessage)
        assert messages[1].is_thinking is False
        assert isinstance(messages[2], AssistantMessage)
        assert messages[2].is_thinking is True

    def test_empty_text_ignored(self, sample_datetime: datetime) -> None:
        """Test that empty text blocks are ignored."""
        content: list[_AssistantContent] = [_TextContent(type="text", text="   ")]
        messages = _process_assistant_content(content, sample_datetime, "claude-3")

        assert messages == []


class TestProcessMessage:
    """Tests for _process_message function."""

    TS: ClassVar[datetime] = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

    def test_user_message_string(self) -> None:
        """Test processing a user message with string content."""
        msg = _RawUserMessage(role="user", content="Hello!", timestamp=self.TS)
        messages = _process_message(msg)

        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "Hello!"

    def test_user_message_empty_string(self) -> None:
        """Test processing a user message with empty string returns empty list."""
        msg = _RawUserMessage(role="user", content="   ", timestamp=self.TS)
        messages = _process_message(msg)

        assert messages == []

    def test_user_message_list_content(self) -> None:
        """Test processing a user message with list content."""
        msg = _RawUserMessage(
            role="user",
            content=[_TextContent(type="text", text="Hello!")],
            timestamp=self.TS,
        )
        messages = _process_message(msg)

        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)

    def test_assistant_message(self) -> None:
        """Test processing an assistant message."""
        msg = _RawAssistantMessage(
            role="assistant",
            content=[_TextContent(type="text", text="I can help!")],
            model="claude-3",
            timestamp=self.TS,
        )
        messages = _process_message(msg)

        assert len(messages) == 1
        assert isinstance(messages[0], AssistantMessage)
        assert messages[0].model == "claude-3"

    def test_tool_result_message(self) -> None:
        """Test processing a tool result message."""
        msg = _RawToolResultMessage(
            role="toolResult",
            toolCallId="tool_123",
            content=[{"text": "File contents"}],
            isError=False,
            timestamp=self.TS,
        )
        messages = _process_message(msg)

        assert len(messages) == 1
        assert isinstance(messages[0], ToolResultMessage)
        assert messages[0].tool_id == "tool_123"
        assert messages[0].result["content"] == "File contents"
        assert messages[0].result["is_error"] is False

    def test_tool_result_with_error(self) -> None:
        """Test processing a tool result with error flag."""
        msg = _RawToolResultMessage(
            role="toolResult",
            toolCallId="tool_123",
            content=[{"text": "Error occurred"}],
            isError=True,
            timestamp=self.TS,
        )
        messages = _process_message(msg)

        assert isinstance(messages[0], ToolResultMessage)
        assert messages[0].result["is_error"] is True

    def test_bash_execution_message(self) -> None:
        """Test processing a bash execution message."""
        msg = _RawBashExecutionMessage(
            role="bashExecution",
            command="ls -la",
            output="file1.txt\nfile2.txt",
            timestamp=self.TS,
        )
        messages = _process_message(msg)

        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert "ls -la" in messages[0].content
        assert "file1.txt" in messages[0].content


class TestProcessLine:
    """Tests for _process_line function."""

    def test_session_entry_ignored(self) -> None:
        """Test that session entries are ignored (return None)."""
        line = json.dumps({"type": "session", "id": "sess_1", "cwd": "/test"})
        entries: list[_RawEntry] = []
        message_entries: dict[str, _RawMessageEntry] = {}

        result = _process_line(line, entries, message_entries)

        assert result is None
        assert entries == []

    def test_message_entry_collected(self) -> None:
        """Test that message entries are collected."""
        ts = 1705314600000  # Unix ms timestamp
        line = json.dumps({
            "type": "message",
            "id": "msg_1",
            "parentId": None,
            "message": {
                "role": "user",
                "content": "Hello",
                "timestamp": ts,
            },
        })
        entries: list[_RawEntry] = []
        message_entries: dict[str, _RawMessageEntry] = {}

        result = _process_line(line, entries, message_entries)

        assert result is None
        assert len(entries) == 1
        assert entries[0].type == "message"
        assert entries[0].id == "msg_1"
        assert "msg_1" in message_entries

    def test_compaction_entry_returns_first_kept_id(self) -> None:
        """Test that compaction entries return the first kept entry ID."""
        line = json.dumps({
            "type": "compaction",
            "id": "compact_1",
            "firstKeptEntryId": "msg_5",
        })
        entries: list[_RawEntry] = []
        message_entries: dict[str, _RawMessageEntry] = {}

        result = _process_line(line, entries, message_entries)

        assert result == "msg_5"
        assert len(entries) == 1

    def test_invalid_json_returns_none(self) -> None:
        """Test that invalid JSON returns None."""
        entries: list[_RawEntry] = []
        message_entries: dict[str, _RawMessageEntry] = {}

        result = _process_line("not valid json", entries, message_entries)

        assert result is None
        assert entries == []

    def test_entry_with_parent_id(self) -> None:
        """Test collecting an entry with a parent ID."""
        ts = 1705314600000
        line = json.dumps({
            "type": "message",
            "id": "msg_2",
            "parentId": "msg_1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi"}],
                "model": "claude-3",
                "timestamp": ts,
            },
        })
        entries: list[_RawEntry] = []
        message_entries: dict[str, _RawMessageEntry] = {}

        _process_line(line, entries, message_entries)

        assert entries[0].parent_id == "msg_1"
