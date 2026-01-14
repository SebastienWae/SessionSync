from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sessionsync.schema import AgentType, Message, Session


class Parser(Protocol):
    """Protocol for agent trace parsers."""

    AGENT_TYPE: Final[AgentType]

    @staticmethod
    def get_root() -> Path | None:
        """Get the root directory where sessions are stored.

        Returns:
            Path to the sessions root directory, or None if not found.
        """
        ...

    @staticmethod
    def get_sessions() -> Iterator[Session]:
        """Get all sessions.

        Returns:
            Iterator of Session objects.
        """
        ...

    @staticmethod
    def get_session_from_id(session_id: str) -> Session | None:
        """Get a session by its ID.

        Args:
            session_id: The session ID.

        Returns:
            The Session if found, None otherwise.
        """
        ...

    @staticmethod
    def get_session_from_path(session_path: Path) -> Session | None:
        """Get a session from a file path.

        If agents store sessions in multiple files, a path to any of the files
        should return the Session.

        Args:
            session_path: Path to a session file.

        Returns:
            The Session if found, None otherwise.
        """
        ...

    @staticmethod
    def get_messages(session: Session) -> Iterator[Message]:
        """Get all messages for a session in the correct order.

        Args:
            session: The Session to get messages for.

        Returns:
            Iterator of Message objects in chronological order.
        """
        ...
