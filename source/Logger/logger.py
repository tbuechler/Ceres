import os
import logging
from inspect import getframeinfo, stack


def stack_info():
    r""" 
    Returns a string containing the information of the caller filename, caller function name and the line number where the logging was called. 
    """
    snd_caller = getframeinfo(stack()[2][0])
    return str(" [{}/{}:{}]".format(
        os.path.basename(snd_caller.filename),
        snd_caller.function,
        snd_caller.lineno
    ))

def log_info(msg: str, show_stack: bool=False):
    r""" Creates a normal info message. """
    msg += stack_info() if show_stack else ""
    logging.getLogger("").info(msg)

def log_warning(msg: str, show_stack: bool=False):
    r""" Creates a warning message. """
    msg += stack_info() if show_stack else ""
    logging.getLogger("").warning(msg)

def log_error(msg: str):
    r""" Creates an error message and terminates the program. """
    msg += stack_info()
    logging.getLogger("").error(msg)
    exit(-1)


