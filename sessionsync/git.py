import subprocess  # noqa: S404


def get_current_branch() -> str | None:
    """Get the current git branch name.

    Returns:
        The current branch name, or None if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        return None
