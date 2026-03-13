from __future__ import annotations

import sys
import os
import enum
import time
from typing import Optional, IO

try:
    import colorama  # type: ignore
    colorama.just_fix_windows_console()
    _COLORAMA = True
except Exception:
    _COLORAMA = False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            kernel32.GetConsoleMode(h, ctypes.byref(mode))
            kernel32.SetConsoleMode(h, mode.value | 0x0004)
        except Exception:
            pass

class MessageType(enum.Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    CHECKPOINT = "CHECKPOINT"
    PERFORMANCE = "PERFORMANCE"
    MEMORY = "MEMORY"

_RESET = "\033[0m"
_COLORS = {
    MessageType.INFO: "\033[36m",
    MessageType.SUCCESS: "\033[92m",
    MessageType.WARNING: "\033[33m",
    MessageType.ERROR: "\033[91m",
    MessageType.DEBUG: "\033[35m",
    MessageType.TRAINING: "\033[34m",
    MessageType.EVALUATION: "\033[95m",
    MessageType.CHECKPOINT: "\033[96m",
    MessageType.PERFORMANCE: "\033[94m",
    MessageType.MEMORY: "\033[93m",
}

class Messenger:
    def __init__(
        self,
        use_color: Optional[bool] = None,
        stream: IO[str] = sys.stdout,
        timestamps: bool = True,
        time_format: str = "%H:%M:%S",
        error_to_stderr: bool = True,
    ) -> None:
        self.stream = stream
        self.error_to_stderr = error_to_stderr
        self.timestamps = timestamps
        self.time_format = time_format
        if use_color is None:
            try:
                self.use_color = hasattr(stream, "isatty") and stream.isatty()
            except Exception:
                self.use_color = False
        else:
            self.use_color = use_color

    def log(self, t: MessageType, message: str) -> None:
        out = sys.stderr if (self.error_to_stderr and t is MessageType.ERROR) else self.stream
        ts = time.strftime(self.time_format) if self.timestamps else ""
        if self.timestamps:
            prefix = f"[{ts}][{t.value}]"
        else:
            prefix = f"[{t.value}]"
        text = f"{prefix} {message}"
        if self.use_color:
            color = _COLORS.get(t, "")
            if color:
                text = f"{color}{text}{_RESET}"
        print(text, file=out, flush=True)

    def __call__(self, t: MessageType, message: str) -> None:
        self.log(t, message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(MessageType.INFO, message)

    def success(self, message: str) -> None:
        """Log a success message."""
        self.log(MessageType.SUCCESS, message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(MessageType.WARNING, message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(MessageType.ERROR, message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(MessageType.DEBUG, message)

    def training(self, message: str) -> None:
        """Log a training message."""
        self.log(MessageType.TRAINING, message)

    def evaluation(self, message: str) -> None:
        """Log an evaluation message."""
        self.log(MessageType.EVALUATION, message)

    def checkpoint(self, message: str) -> None:
        """Log a checkpoint message."""
        self.log(MessageType.CHECKPOINT, message)

    def performance(self, message: str) -> None:
        """Log a performance message."""
        self.log(MessageType.PERFORMANCE, message)

    def memory(self, message: str) -> None:
        """Log a memory message."""
        self.log(MessageType.MEMORY, message)

msg = Messenger()
