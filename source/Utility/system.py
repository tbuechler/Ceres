import os

from datetime import datetime
from source.Logger.logger import *


def get_timestamp():
    r""" Returns the current timestamp in form of 'ymd_HMS'."""
    return datetime.now().strftime("%y%m%d_%H%M%S")


def make_dirs(dirs):
    """
    Create multiple directories.

    Args:

    * `dirs (list(str))`: 
        * Path of multiple directories which need to be created.
    """
    try:
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
    except Exception as err:
        log_error("ERROR: make_dirs failed due to {0}.".format(err))

