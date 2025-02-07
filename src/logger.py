''' 
    A logger in Python is used to record (log) messages about a program's execution. It helps in debugging, monitoring, and tracking issues in applications by capturing error messages, warnings, or informational messages.
'''

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # filename for all the logging information name with datetime
LOGS_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(LOGS_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)


if __name__ == "__main__":
    logging.info("Logging has started!!!")


''' 
    (function) def basicConfig(
        *,
        filename: StrPath | None = ...,
        filemode: str = ...,
        format: str = ...,
        datefmt: str | None = ...,
        style: _FormatStyle = ...,
        level: _Level | None = ...,
        stream: SupportsWrite[str] | None = ...,
        handlers: Iterable[Handler] | None = ...,
        force: bool | None = ...,
        encoding: str | None = ...,
        errors: str | None = ...
    ) -> None
'''