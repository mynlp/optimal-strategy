import datetime
import os
import subprocess
import sys
from logging import DEBUG, INFO, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from threading import current_thread
from typing import TextIO

# Have to set this environment variable.
# Probably related to the following issue:
# https://github.com/pytorch/pytorch/issues/37377
os.environ["MKL_THREADING_LAYER"] = "GNU"

###########
## Utils ##
###########


def get_thread_id() -> int:
    thread = current_thread()

    ####### Thread by default has a name of "ThreadPoolExecuter-%d-%d", where the first number is the thread number in the process, and the scond is the worker thread number within the pool.

    # Ad-hoc way to obrain thread index.
    # Thread has a name of "Thread_%d", where the first number is the thread number in the process, and the scond is the worker thread number within the pool.

    thread_id: int = int(thread.name.split("_")[-1])

    return thread_id


def thread_log(message: str):
    """Print message to stderr."""
    print(
        "{}: Thread {}: {}".format(datetime.datetime.now(), get_thread_id(), message),
        file=sys.stderr,
    )


def thread_run_subprocess(command_str: str, log_file: Path):
    try:
        with log_file.open(mode="w") as f:
            p = subprocess.run(
                command_str,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            p.check_returncode()
    except Exception as e:
        print(
            "Exception raised in thread {}: {}".format(get_thread_id(), e),
            file=sys.stderr,
        )


############
## Logger ##
############

MAIN_LOGGER_NAME: str = "Joblogger"


def get_main_logger() -> tuple[Logger, "StreamHandler[TextIO]"]:
    # def get_main_logger():
    logger = getLogger(MAIN_LOGGER_NAME)
    logger.setLevel(DEBUG)
    handler = StreamHandler()
    handler.setStream(sys.stderr)
    # handler.setLevel(DEBUG)
    handler.setLevel(INFO)
    fmt = Formatter("%(asctime)s %(name)s: [%(levelname)s]: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False

    return logger, handler


def set_debug_mode(handler: "StreamHandler[TextIO]", debug: bool) -> None:
    if debug == True:
        handler.setLevel(DEBUG)
        fmt = Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
        handler.setFormatter(fmt)


main_logger, main_handler = get_main_logger()
