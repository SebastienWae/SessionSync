from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sessionsync.parsers.opencode import (
    _extract_data_from_url,
    _process_tool_part,
    _RawPart,
    _RawPartState,
    _RawTime,
)
from sessionsync.schema import ToolResultMessage, ToolUseMessage

if TYPE_CHECKING:
    from pathlib import Path


class TestRawTimeParseTimestamp:
    """Tests for _RawTime.parse_timestamp validator."""

    def test_none_returns_none(self) -> None:
        """Test that None input returns None."""
        raw = _RawTime(created=None)
        assert raw.created is None

    def test_seconds_timestamp(self) -> None:
        """Test parsing a Unix seconds timestamp."""
        # 2024-01-15 10:30:00 UTC in seconds
        raw = _RawTime.model_validate({"created": 1705314600})

        assert raw.created is not None
        assert raw.created.year == 2024
        assert raw.created.month == 1
        assert raw.created.day == 15
        assert raw.created.tzinfo == UTC

    def test_milliseconds_timestamp(self) -> None:
        """Test parsing a Unix milliseconds timestamp (auto-detected)."""
        # 2024-01-15 10:30:00 UTC in milliseconds
        raw = _RawTime.model_validate({"created": 1705314600000})

        assert raw.created is not None
        assert raw.created.year == 2024
        assert raw.created.month == 1
        assert raw.created.day == 15

    def test_threshold_detection(self) -> None:
        """Test that timestamps above threshold are treated as milliseconds."""
        raw_ms = _RawTime.model_validate({"created": 1_000_000_000_001})
        raw_s = _RawTime.model_validate({"created": 1705314600})

        assert raw_ms.created is not None
        assert raw_s.created is not None
        assert raw_ms.created.year == 2001
        assert raw_s.created.year == 2024


class TestExtractDataFromUrl:
    """Tests for _extract_data_from_url function."""

    def test_data_url_base64(self) -> None:
        """Test extracting data from a base64 data URL."""
        url = "data:image/png;base64,iVBORw0KGgo="
        result = _extract_data_from_url(url)

        assert result == "iVBORw0KGgo="

    def test_data_url_no_comma(self) -> None:
        """Test data URL without comma returns empty string."""
        url = "data:image/png;base64"
        result = _extract_data_from_url(url)

        assert result == ""

    def test_data_url_with_charset(self) -> None:
        """Test data URL with charset parameter."""
        url = "data:text/plain;charset=utf-8;base64,SGVsbG8="
        result = _extract_data_from_url(url)

        assert result == "SGVsbG8="

    def test_file_path_reads_and_encodes(self, tmp_path: Path) -> None:
        """Test that file paths are read and base64 encoded."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Hello, World!")

        result = _extract_data_from_url(str(test_file))

        expected = base64.b64encode(b"Hello, World!").decode("ascii")
        assert result == expected

    def test_nonexistent_file_returns_empty(self) -> None:
        """Test that nonexistent file path returns empty string."""
        result = _extract_data_from_url("/nonexistent/path/file.txt")

        assert result == ""


class TestProcessToolPart:
    """Tests for _process_tool_part function."""

    def test_missing_tool_name_returns_empty(self, sample_datetime: datetime) -> None:
        """Test that missing tool name returns empty list."""
        raw = _RawPart(type="tool", tool=None, callID="t1", state=None)
        messages = _process_tool_part(raw, sample_datetime, "claude-3")

        assert messages == []

    def test_missing_call_id_returns_empty(self, sample_datetime: datetime) -> None:
        """Test that missing call_id returns empty list."""
        raw = _RawPart(type="tool", tool="Read", callID=None, state=None)
        messages = _process_tool_part(raw, sample_datetime, "claude-3")

        assert messages == []

    def test_tool_use_without_state(self, sample_datetime: datetime) -> None:
        """Test creating a tool use message without state."""
        raw = _RawPart(type="tool", tool="Read", callID="t1", state=None)
        messages = _process_tool_part(raw, sample_datetime, "claude-3")

        assert len(messages) == 1
        assert isinstance(messages[0], ToolUseMessage)
        assert messages[0].tool_name == "Read"
        assert messages[0].tool_id == "t1"
        assert messages[0].parameters == {}
        assert messages[0].model == "claude-3"

    def test_tool_use_with_input(self, sample_datetime: datetime) -> None:
        """Test creating a tool use message with input parameters."""
        raw = _RawPart(
            type="tool",
            tool="Read",
            callID="t1",
            state=_RawPartState(input={"file_path": "/test.txt"}, output=None),
        )
        messages = _process_tool_part(raw, sample_datetime, "claude-3")

        assert len(messages) == 1
        assert isinstance(messages[0], ToolUseMessage)
        assert messages[0].parameters == {"file_path": "/test.txt"}

    def test_tool_use_with_output_creates_result(self, sample_datetime: datetime) -> None:
        """Test that output creates both tool use and tool result messages."""
        raw = _RawPart(
            type="tool",
            tool="Read",
            callID="t1",
            state=_RawPartState(
                input={"file_path": "/test.txt"},
                output="File contents here",
            ),
        )
        messages = _process_tool_part(raw, sample_datetime, "claude-3")

        assert len(messages) == 2
        assert isinstance(messages[0], ToolUseMessage)
        assert isinstance(messages[1], ToolResultMessage)
        assert messages[1].tool_id == "t1"
        assert messages[1].result == {"content": "File contents here", "is_error": False}

    def test_tool_use_with_empty_output_no_result(self, sample_datetime: datetime) -> None:
        """Test that empty output doesn't create a result message."""
        raw = _RawPart(
            type="tool",
            tool="Read",
            callID="t1",
            state=_RawPartState(input={"path": "/x"}, output=""),
        )
        messages = _process_tool_part(raw, sample_datetime, "claude-3")

        assert len(messages) == 1
        assert isinstance(messages[0], ToolUseMessage)
