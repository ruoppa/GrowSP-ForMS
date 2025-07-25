import os
import logging
import torch.distributed as dist

from typing import Union
from .misc import prompt_bool

logger_initialized = {}


def get_root_logger(
    log_file: str = None, log_level: int = logging.INFO, name: str = "main"
) -> logging.Logger:
    """Get root logger and add a keyword filter to it. The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will also be added. The name of the root logger is the
    top-level package name, e.g., "mmdet3d".

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger. Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a filter keyword. Defaults to 'main'.
    Returns:
        logging.Logger: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(
    name: str,
    log_file: str = None,
    log_level: int = logging.INFO,
    overwrite: bool = False,
) -> logging.Logger:
    """Initialize and get a logger by name. If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will be directly returned. During initialization, a
    StreamHandler will always be added. If `log_file` is specified and the process rank is 0, a FileHandler will also be added.

    Args:
        name (str): logger name
        log_file (str, optional): The log filename. If specified, a FileHandler will be added to the logger. Defaults to None.
        log_level (int, optional):  The logger level. Note that only the process of rank 0 is affected, and other processes will set the level to "Error" thus be silent most of the time. Defaults to logging.INFO.
        overwrite (bool, optional): If True, overwrite any log that exists in 'log_file'. If False, the user is prompted to pick whether to overwrite. Defaults to False.

    Returns:
        logging.Logger: The obtained logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_mode = "a"  # Append to file
        if os.path.exists(log_file):
            if not overwrite:
                overwrite = prompt_bool(
                    f"Log at '{log_file}' already exists. Do you want to overwrite it? (no --> append to existing log)"
                )
            if overwrite:
                file_mode = "w"

        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(
    msg: str, logger: Union[logging.Logger, str, None] = None, level: int = logging.INFO
):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (Union[logging.Logger, str, None]): The logger to be used. Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages. Defaults to None.
        level (int): Logging level. Only available when `logger` is a Logger object or "root". Defaults to logging.INFO.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            f'logger should be either a logging.Logger object, str, "silent" or None, but got {type(logger)}'
        )
