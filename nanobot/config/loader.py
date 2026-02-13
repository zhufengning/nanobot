"""Configuration loading utilities."""

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from nanobot.config.schema import Config


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOTENV_LOADED = False


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".nanobot" / "config.json"


def get_env_path() -> Path:
    """Get the default dotenv path (~/.nanobot/.env)."""
    return Path.home() / ".nanobot" / ".env"


def get_data_dir() -> Path:
    """Get the nanobot data directory."""
    from nanobot.utils.helpers import get_data_path
    return get_data_path()


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse a single .env line into (key, value)."""
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[7:].strip()
    if "=" not in text:
        return None

    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not _ENV_KEY_RE.match(key):
        return None

    raw_value = raw_value.strip()
    if not raw_value:
        return key, ""

    if raw_value[0] in ('"', "'"):
        # Use shlex to handle quoted strings and escaped characters.
        lexer = shlex.shlex(raw_value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        parts = list(lexer)
        return key, (parts[0] if parts else "")

    value = raw_value.split(" #", 1)[0].strip()
    return key, value


def load_dotenv(env_path: Path | None = None, override: bool = False) -> Path | None:
    """
    Load environment variables from ~/.nanobot/.env into os.environ.

    Existing env vars are kept by default (override=False).
    Returns the loaded path, or None when file does not exist / unreadable.
    """
    global _DOTENV_LOADED

    path = env_path or get_env_path()
    if not path.exists() or not path.is_file():
        return None

    # Default path is loaded once per process to avoid repeated parsing.
    if env_path is None and _DOTENV_LOADED:
        return path

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value

    if env_path is None:
        _DOTENV_LOADED = True
    return path


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Optional path to config file. Uses default if not provided.
    
    Returns:
        Loaded configuration object.
    """
    # Load ~/.nanobot/.env first so skills/tools can consume env vars.
    load_dotenv()

    path = config_path or get_config_path()
    
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            data = _migrate_config(data)
            return Config.model_validate(convert_keys(data))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")
    
    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to camelCase format
    data = config.model_dump()
    data = convert_to_camel(data)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data


def convert_keys(data: Any) -> Any:
    """Convert camelCase keys to snake_case for Pydantic."""
    if isinstance(data, dict):
        return {camel_to_snake(k): convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys(item) for item in data]
    return data


def convert_to_camel(data: Any) -> Any:
    """Convert snake_case keys to camelCase."""
    if isinstance(data, dict):
        return {snake_to_camel(k): convert_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_camel(item) for item in data]
    return data


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])
