from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from sessionsync.schema import Message, Session


class Exporter(Protocol):
    """Protocol for session exporters."""

    @staticmethod
    def export(session: Session, messages: list[Message], output: Path) -> Path:
        """Export session and messages to a file.

        Args:
            session: Session metadata.
            messages: List of filtered messages.
            output: Output directory path.

        Returns:
            Path to the exported file.
        """
        ...
