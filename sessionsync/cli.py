import logging
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, override

import cyclopts
from rich.console import Console
from rich.logging import RichHandler
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from sessionsync.exporters import json as json_exporter
from sessionsync.exporters import markdown, toon
from sessionsync.git import get_current_branch
from sessionsync.parsers import claude_code, opencode, pi
from sessionsync.schema import (
    AssistantMessage,
    AttachmentMessage,
    ToolResultMessage,
    ToolUseMessage,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from sessionsync.exporters.base import Exporter
    from sessionsync.parsers.base import Parser
    from sessionsync.schema import Message, Session

logger = logging.getLogger("sessionsync")
console = Console()


app = cyclopts.App(
    name="sessionsync",
    help="Sync coding agent sessions to markdown files.",
    version_flags=["--version", "-V"],
    version=version("sessionsync"),
)


ParserName = Literal["claude", "opencode", "pi"]
ENABLED_PARSERS: dict[ParserName, Parser] = {"claude": claude_code, "opencode": opencode, "pi": pi}

ExporterName = Literal["markdown", "toon", "json"]
ENABLED_EXPORTERS: dict[ExporterName, Exporter] = {
    "markdown": markdown,
    "toon": toon,
    "json": json_exporter,
}


def _get_parsers(agent: ParserName | Literal["all"]) -> Iterable[Parser]:
    """Get parsers for the specified agent(s).

    Args:
        agent: Agent name ('claude', 'opencode', 'pi') or 'all'.

    Returns:
        Iterable of parser instances.
    """
    match agent:
        case "all":
            return ENABLED_PARSERS.values()
        case _:
            return [ENABLED_PARSERS[agent]]


def _filter_sessions(
    sessions: Iterator[Session],
    workspace: Path | None,
    branch: str | None,
    *,
    include_subagents: bool,
) -> Iterator[Session]:
    """Filter sessions by workspace, branch, and subagent status.

    Args:
        sessions: Iterator of sessions to filter.
        workspace: Filter sessions by exact workspace path match.
        branch: Filter by git branch. Sessions with None branch are always included.
        include_subagents: Whether to include subagent sessions.

    Yields:
        Sessions matching all filter criteria.
    """
    for session in sessions:
        if workspace is not None and session.workspace != workspace:
            continue
        if branch is not None and session.git_branch is not None and session.git_branch != branch:
            continue
        if not include_subagents and session.parent_session_id is not None:
            continue
        yield session


def _filter_messages(
    messages: Iterator[Message],
    *,
    include_tools: bool,
    include_thinking: bool,
    include_attachments: bool,
) -> Iterator[Message]:
    """Filter messages by type.

    Args:
        messages: Iterator of messages to filter.
        include_tools: Include tool use and result messages.
        include_thinking: Include assistant thinking messages.
        include_attachments: Include file attachment messages.

    Yields:
        Messages matching the filter criteria.
    """
    for msg in messages:
        if isinstance(msg, (ToolUseMessage, ToolResultMessage)) and not include_tools:
            continue
        if isinstance(msg, AssistantMessage) and msg.is_thinking and not include_thinking:
            continue
        if isinstance(msg, AttachmentMessage) and not include_attachments:
            continue
        yield msg


def _sync_session(
    parser: Parser,
    session: Session,
    output: Path,
    exporter: Exporter,
    *,
    include_tools: bool,
    include_thinking: bool,
    include_attachments: bool,
) -> None:
    """Sync a single session to the output directory.

    Args:
        parser: Parser instance to get messages from.
        session: Session to sync.
        output: Output directory path.
        exporter: Exporter instance to write the output.
        include_tools: Include tool use and result messages.
        include_thinking: Include assistant thinking messages.
        include_attachments: Include file attachment messages.
    """
    messages = list(
        _filter_messages(
            parser.get_messages(session),
            include_tools=include_tools,
            include_thinking=include_thinking,
            include_attachments=include_attachments,
        )
    )

    filepath = exporter.export(session, messages, output)
    logger.info("[%s] Exported %s (%d messages) to %s", session.agent.value, session.id, len(messages), filepath.name)


def _sync_sessions(
    parsers: Iterable[Parser],
    output: Path,
    exporter: Exporter,
    workspace: Path | None,
    branch: str | None,
    *,
    include_subagents: bool,
    include_tools: bool,
    include_thinking: bool,
    include_attachments: bool,
) -> None:
    """Sync sessions from parsers to output directory.

    Args:
        parsers: Iterable of parser instances.
        output: Output directory path.
        exporter: Exporter instance to write the output.
        workspace: Filter sessions by exact workspace path match.
        branch: Filter by git branch.
        include_subagents: Include subagent sessions.
        include_tools: Include tool use and result messages.
        include_thinking: Include assistant thinking messages.
        include_attachments: Include file attachment messages.
    """
    output.mkdir(parents=True, exist_ok=True)

    for parser in parsers:
        root = parser.get_root()
        if root is None:
            logger.debug("Parser %s has no root directory", type(parser).__name__)
            continue

        logger.debug("Scanning %s", root)
        sessions = list(
            _filter_sessions(
                parser.get_sessions(),
                workspace,
                branch,
                include_subagents=include_subagents,
            )
        )
        logger.info("[%s] Found %d sessions after filtering", parser.AGENT_TYPE.value, len(sessions))

        for session in sessions:
            _sync_session(
                parser,
                session,
                output,
                exporter,
                include_tools=include_tools,
                include_thinking=include_thinking,
                include_attachments=include_attachments,
            )


class _SessionFileHandler(FileSystemEventHandler):
    """Handle session file changes for watch mode."""

    _output: Path
    _exporter: Exporter
    _workspace: Path | None
    _branch: str | None
    _include_subagents: bool
    _include_tools: bool
    _include_thinking: bool
    _include_attachments: bool
    _parser_by_root: dict[Path, Parser]

    def __init__(
        self,
        parsers: Iterable[Parser],
        output: Path,
        exporter: Exporter,
        workspace: Path | None,
        branch: str | None,
        *,
        include_subagents: bool,
        include_tools: bool,
        include_thinking: bool,
        include_attachments: bool,
    ) -> None:
        """Initialize the handler.

        Args:
            parsers: Iterable of parser instances.
            output: Output directory path.
            exporter: Exporter instance to write the output.
            workspace: Filter sessions by exact workspace path match.
            branch: Filter by git branch.
            include_subagents: Include subagent sessions.
            include_tools: Include tool use and result messages.
            include_thinking: Include assistant thinking messages.
            include_attachments: Include file attachment messages.
        """
        super().__init__()
        self._output = output
        self._exporter = exporter
        self._workspace = workspace
        self._branch = branch
        self._include_subagents = include_subagents
        self._include_tools = include_tools
        self._include_thinking = include_thinking
        self._include_attachments = include_attachments

        self._parser_by_root = {}
        for parser in parsers:
            root = parser.get_root()
            if root is not None:
                self._parser_by_root[root] = parser

    def _find_parser_for_path(self, path: Path) -> Parser | None:
        """Find the parser responsible for a given file path.

        Args:
            path: Path to the file.

        Returns:
            Parser instance if found, None otherwise.
        """
        for root, parser in self._parser_by_root.items():
            if path.is_relative_to(root):
                return parser
        return None

    def _session_passes_filter(self, session: Session) -> bool:
        """Check if a session passes the configured filters.

        Args:
            session: Session to check.

        Returns:
            True if the session passes all filters.
        """
        if self._workspace is not None and session.workspace != self._workspace:
            return False
        if self._branch is not None and session.git_branch != self._branch:
            return False
        return self._include_subagents or session.parent_session_id is None

    def _handle_file_change(self, path: Path) -> None:
        """Reparse and export a changed session file.

        Args:
            path: Path to the changed file.
        """
        logger.debug("File changed: %s", path)
        parser = self._find_parser_for_path(path)
        if parser is None:
            logger.debug("No parser found for %s", path)
            return

        session = parser.get_session_from_path(path)
        if session is None:
            logger.debug("Could not parse session from %s", path)
            return

        if not self._session_passes_filter(session):
            logger.debug("Session %s filtered out", session.id)
            return

        _sync_session(
            parser,
            session,
            self._output,
            self._exporter,
            include_tools=self._include_tools,
            include_thinking=self._include_thinking,
            include_attachments=self._include_attachments,
        )

    @override
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: File system event.
        """
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        self._handle_file_change(Path(src_path))

    @override
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: File system event.
        """
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        self._handle_file_change(Path(src_path))


def _watch_sessions(
    parsers: Iterable[Parser],
    output: Path,
    exporter: Exporter,
    workspace: Path | None,
    branch: str | None,
    *,
    include_subagents: bool,
    include_tools: bool,
    include_thinking: bool,
    include_attachments: bool,
) -> None:
    """Watch for session changes and sync continuously.

    Args:
        parsers: Iterable of parser instances.
        output: Output directory path.
        exporter: Exporter instance to write the output.
        workspace: Filter sessions by exact workspace path match.
        branch: Filter by git branch.
        include_subagents: Include subagent sessions.
        include_tools: Include tool use and result messages.
        include_thinking: Include assistant thinking messages.
        include_attachments: Include file attachment messages.
    """
    handler = _SessionFileHandler(
        parsers,
        output,
        exporter,
        workspace,
        branch,
        include_subagents=include_subagents,
        include_tools=include_tools,
        include_thinking=include_thinking,
        include_attachments=include_attachments,
    )

    observer = Observer()
    roots_scheduled = 0

    for parser in parsers:
        root = parser.get_root()
        if root is not None and root.exists():
            observer.schedule(handler, str(root), recursive=True)
            logger.debug("Watching %s", root)
            roots_scheduled += 1

    if roots_scheduled == 0:
        logger.warning("No session directories found to watch")
        return

    observer.start()
    logger.info("Watching for session changes... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watch...")
        observer.stop()

    observer.join()


CWD = Path().cwd()
CURRENT_BRANCH = get_current_branch()


@app.default
def main(
    *,
    agent: Annotated[
        ParserName | Literal["all"],
        cyclopts.Parameter(
            name="--agent",
            alias="-a",
            help="Agent to sync.",
        ),
    ] = "all",
    output: Annotated[
        Path,
        cyclopts.Parameter(
            name="--output",
            alias="-o",
            help="Output directory.",
        ),
    ] = CWD / ".sessions",
    output_format: Annotated[
        ExporterName,
        cyclopts.Parameter(
            name="--format",
            alias="-f",
            help="Output format.",
        ),
    ] = "markdown",
    workspace: Annotated[
        Path,
        cyclopts.Parameter(
            name="--workspace",
            alias="-w",
            help="Filter sessions by workspace path.",
        ),
    ] = CWD,
    branch: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--branch",
            alias="-b",
            help="Filter by git branch.",
        ),
    ] = None,
    no_subagents: Annotated[
        bool,
        cyclopts.Parameter(
            name="--no-subagents",
            negative=(),
            help="Exclude subagent sessions (only sync main sessions).",
        ),
    ] = False,
    no_tools: Annotated[
        bool,
        cyclopts.Parameter(
            name="--no-tools",
            negative=(),
            help="Exclude tool use and tool result messages.",
        ),
    ] = False,
    no_thinking: Annotated[
        bool,
        cyclopts.Parameter(
            name="--no-thinking",
            negative=(),
            help="Exclude assistant thinking messages.",
        ),
    ] = False,
    no_attachments: Annotated[
        bool,
        cyclopts.Parameter(
            name="--no-attachments",
            negative=(),
            help="Exclude file attachment messages.",
        ),
    ] = False,
    watch: Annotated[
        bool,
        cyclopts.Parameter(
            name="--watch",
            negative=(),
            help="Watch for new sessions and sync continuously.",
        ),
    ] = False,
    verbose: Annotated[
        int,
        cyclopts.Parameter(
            name="--verbose",
            alias="-v",
            count=True,
            help="Increase verbosity (-v for debug).",
        ),
    ] = 0,
) -> None:
    """Sync coding agent sessions to files.

    Args:
        agent: Agent type to sync ('claude', 'opencode', 'pi', or 'all').
        output: Output directory path.
        output_format: Output format ('markdown', 'toon', or 'json').
        workspace: Filter sessions by workspace path.
        branch: Filter by git branch ('all' to sync all branches).
        no_subagents: Exclude subagent sessions.
        no_tools: Exclude tool use and tool result messages.
        no_thinking: Exclude assistant thinking messages.
        no_attachments: Exclude file attachment messages.
        watch: Watch for new sessions and sync continuously.
        verbose: Verbosity level (0=info, 1=debug).
    """
    level = logging.DEBUG if verbose >= 1 else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, show_time=watch)],
    )

    parsers = _get_parsers(agent)
    exporter = ENABLED_EXPORTERS[output_format]

    _sync_sessions(
        parsers=parsers,
        output=output,
        exporter=exporter,
        workspace=workspace,
        branch=branch,
        include_subagents=not no_subagents,
        include_tools=not no_tools,
        include_thinking=not no_thinking,
        include_attachments=not no_attachments,
    )

    if watch:
        _watch_sessions(
            parsers=parsers,
            output=output,
            exporter=exporter,
            workspace=workspace,
            branch=branch,
            include_subagents=not no_subagents,
            include_tools=not no_tools,
            include_thinking=not no_thinking,
            include_attachments=not no_attachments,
        )
