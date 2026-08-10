import logging
import sys

# ANSI color codes for terminal output
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "GRAY": "\033[90m",
}

LEVEL_COLORS = {
    "DEBUG": COLORS["GRAY"],
    "INFO": COLORS["GREEN"],
    "WARNING": COLORS["YELLOW"],
    "ERROR": COLORS["RED"],
    "CRITICAL": COLORS["RED"] + COLORS["BOLD"],
}

LEVEL_ICONS = {
    "DEBUG": "🔍",
    "INFO": "✅",
    "WARNING": "⚠️ ",
    "ERROR": "❌",
    "CRITICAL": "🔥",
}


class ApexGuardFormatter(logging.Formatter):
    """Production-grade colored formatter that shows module, function, and context."""

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        level_color = LEVEL_COLORS.get(level, COLORS["WHITE"])
        icon = LEVEL_ICONS.get(level, "")
        reset = COLORS["RESET"]
        cyan = COLORS["CYAN"]
        dim = COLORS["DIM"]
        magenta = COLORS["MAGENTA"]
        blue = COLORS["BLUE"]

        # Build the module path like: app.services.vector_service
        module = record.name

        # Timestamp
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Function name
        func = record.funcName if record.funcName != "<module>" else ""
        func_display = f"{magenta}{func}(){reset}" if func else ""

        # File + line
        file_display = f"{dim}{record.filename}:{record.lineno}{reset}"

        # Core message
        msg = record.getMessage()

        # Final layout:
        # [TIMESTAMP] ICON LEVEL  | module.path | function() | file:line
        #   → message
        header = (
            f"{dim}{timestamp}{reset}  "
            f"{icon} {level_color}{level:<8}{reset} "
            f"{dim}│{reset} {cyan}{module}{reset} "
            f"{dim}│{reset} {func_display} "
            f"{dim}│{reset} {file_display}"
        )
        body = f"  {blue}→{reset} {msg}"

        # If there's exception info, append it
        output = f"{header}\n{body}"
        if record.exc_info and record.exc_info[1]:
            exc_text = self.formatException(record.exc_info)
            output += f"\n{COLORS['RED']}{exc_text}{reset}"

        return output


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with the ApexGuard formatter.

    Usage:
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened", extra={...})
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(ApexGuardFormatter())

        logger.addHandler(handler)
        logger.propagate = False

    return logger
