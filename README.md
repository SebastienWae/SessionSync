# SessionSync

SessionSync exports AI coding agent sessions to standardized file formats. It reads session data from Claude Code, OpenCode, and Pi, then converts them to Markdown, JSON, or TOON format.

## Problem

AI coding agents store session data in different formats and locations. This makes it difficult to:

- Review past conversations across multiple agents
- Search and organize session history
- Share, back up or archive coding sessions
- Analyze tool usage patterns

Long sessions also cause agents to enter the "[DUMB ZONE](https://www.youtube.com/watch?v=rmvDxxNubIg&t=355s)". Short sessions avoid this but lose context. SessionSync exports sessions to files so agents can read them back, giving them memory through the filesystem.

## Supported Agents

- Claude Code
- OpenCode
- Pi

## Export Formats

| Format | Description |
|--------|-------------|
| Markdown | Human-readable with YAML frontmatter containing session metadata |
| JSON | Structured data for programmatic access and analysis |
| TOON | Binary format for compact storage |

## Installation

### PyPI

```sh
pip install sessionsync
```

Or with uv:

```sh
uv tool install sessionsync --prerelease=allow
```

## Usage

Export all sessions from the current workspace:

```sh
sessionsync
```

Export sessions from a specific agent:

```sh
sessionsync --agent claude
sessionsync --agent opencode
sessionsync --agent pi
```

Export to a specific format:

```sh
sessionsync --format json
sessionsync --format toon
```

Filter by git branch:

```sh
sessionsync --branch main
```

Watch mode for continuous export:

```sh
sessionsync --watch
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--agent` | `-a` | Agent to sync: `claude`, `opencode`, `pi`, or `all` (default: `all`) |
| `--output` | `-o` | Output directory (default: `.sessions`) |
| `--format` | `-f` | Export format: `markdown`, `json`, or `toon` (default: `markdown`) |
| `--workspace` | `-w` | Filter by workspace path (default: current directory) |
| `--branch` | `-b` | Filter by git branch |
| `--no-subagents` | | Exclude subagent sessions |
| `--no-tools` | | Exclude tool use and result messages |
| `--no-thinking` | | Exclude assistant thinking messages |
| `--no-attachments` | | Exclude file attachments |
| `--watch` | | Watch for changes and sync continuously |
| `--verbose` | `-v` | Enable debug logging |

### Examples

Export only main sessions without subagents:

```sh
sessionsync --no-subagents
```

Export without tool calls and thinking blocks:

```sh
sessionsync --no-tools --no-thinking
```

Export all Claude Code sessions to JSON:

```sh
sessionsync --agent claude --format json
```

## License

[MIT](LICENSE)
