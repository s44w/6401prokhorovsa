import logging
import time
from functools import wraps


class PipelineLogger:
    def __init__(self):
        self.logger = logging.getLogger("pipeline_logger")
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler("app.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def timeit(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            self.logger.info(f"Starting '{func.__name__}'...")
            start = time.time()
            try:
                result = await func(*args, **kwargs)
            finally:
                end = time.time()
                self.logger.info(f"Finished '{func.__name__}' in {(end - start): .3f}s")
            return result

        return async_wrapper
