import logging

class Logger:
    def __init__(self,name, file_handler_filename = None) -> None:
        # logger = logging.getLogger(__name__)
        format = "%(asctime)s: %(message)s"
        handlers = [logging.StreamHandler()]
        if file_handler_filename is not None:
            handlers.append(logging.FileHandler(f"{file_handler_filename}.log"))
        logging.basicConfig(
            level=logging.INFO,
            format=format,
            datefmt="%H:%M:%S",
            handlers=handlers
        )
    def get_logger(self):
        return logging