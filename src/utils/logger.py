import logging
import colorlog

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

console_handler = colorlog.StreamHandler()

console_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M;%S", reset=True
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("app.log")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.propagate = False

logger.info("Logger initialized successfully")
