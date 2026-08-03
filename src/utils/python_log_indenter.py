# -*- encoding: utf-8 -*-
"""
Minimal stub for python_log_indenter.
Provides IndentedLoggerAdapter for environments where the package is not installed.
"""

import logging
from typing import Union


class IndentedLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds indentation to log messages."""

    def __init__(self, logger: logging.Logger, spaces: int = 2, extra=None):
        super().__init__(logger, extra or {})
        self.spaces = spaces
        self._indent_level = 0

    def add(self):
        """Increase indentation level."""
        self._indent_level += 1

    def sub(self):
        """Decrease indentation level."""
        if self._indent_level > 0:
            self._indent_level -= 1

    def process(self, msg, kwargs):
        """Add indentation prefix to message."""
        if self._indent_level > 0:
            indent = ' ' * (self.spaces * self._indent_level)
            msg = indent + msg
        return msg, kwargs

    def log(self, level: Union[int, str], msg: str, *args, **kwargs):
        """Log with indentation."""
        if isinstance(level, str):
            level = getattr(logging, level)
        msg, kwargs = self.process(msg, kwargs)
        self.logger.log(level, msg, *args, **kwargs)

    # Convenience methods matching standard logging levels
    def debug(self, msg, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log(logging.CRITICAL, msg, *args, **kwargs)
