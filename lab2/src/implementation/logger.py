import logging
import time


class PipelineLogger:
    def __init__(self, log_file: str = "pipeline.log"):
        self.logger = logging.getLogger("pipeline_logger")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def timeit(self, func):
        def wrapper(*args, **kwargs):
            self.logger.info(f"Starting '{func.__name__}'...")
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            self.logger.info(f"Finished '{func.__name__}' in {(end - start): .3f}s")
            return result

        return wrapper
