"""
Utils for creating a Logger that writes info, warnings and so on into a logs file
in the experiment directory. Each experiment has its own independent logs
"""

import os
import traceback
from datetime import datetime

LOGGER = None


def log_function(func):
    """
    Decorator for logging a method in case of raising an exception
    """
    def try_call_log(*args, **kwargs):
        """
        Calling the function but calling the logger in case an exception is raised
        """
        try:
            if(LOGGER is not None):
                message = f"Calling: {func.__name__}..."
                LOGGER.log_info(message=message, message_type="info")
            return func(*args, **kwargs)
        except Exception as e:
            if(LOGGER is None):
                raise e
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()
    return try_call_log


def for_all_methods(decorator):
    """
    Decorator that applies a decorator to all methods inside a class
    """
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def print_(message, message_type="info"):
    """
    Overloads the print method so that the message is written both in logs file and console
    """

    print(message)
    if(LOGGER is not None):
        LOGGER.log_info(message, message_type)
    return


def log_info(message, message_type="info"):
    if(LOGGER is not None):
        LOGGER.log_info(message, message_type)
    return


class Logger():
    """
    Class that instanciates a Logger object to write logs into a file

    Args:
    -----
    exp_path: string
        path to the root directory of an experiment where the logs are saved
    file_name: string
        name of the file where logs are stored
    """

    def __init__(self, exp_path, file_name="logs.txt"):
        """
        Initializer of the logger object
        """

        logs_path = os.path.join(exp_path, file_name)
        self.logs_path = logs_path

        if not os.path.exists(logs_path):
            if(not os.path.exists(exp_path)):
                os.makedirs(exp_path)
            with open(logs_path, 'w') as f:
                f.write("")

        global LOGGER
        LOGGER = self
        return

    def log_info(self, message, message_type="info", **kwargs):
        """
        Logging a message into the file
        """

        if(message_type not in ["new_exp", "info", "warning", "error", "params"]):
            message_type = "info"
        cur_time = self._get_datetime()
        format_message = self._format_message(message=message, cur_time=cur_time,
                                              message_type=message_type)
        with open(self.logs_path, 'a') as f:
            f.write(format_message)

        if(message_type == "error"):
            exit()

        return

    def log_params(self, params):
        """
        Logging parameters so that it is visually appealing
        Args:
        -----
        params: dictionary
            dictionary containing parameters and values
        """

        for param, value in params.items():
            message = f"    {param}:{value}"
            self.log_info(message, message_type="params")

        return

    def _format_message(self, message, cur_time, message_type="info"):
        """
        Formatting the message to have a standarizied template
        """
        pre_string = ""
        if(message_type == "new_exp"):
            pre_string = "\n\n\n"
        form_message = f"{pre_string}{cur_time}    {message_type.upper()}: {message}\n"
        return form_message

    def _get_datetime(self):
        """
        Obtaining current data and time in format YYYY-MM-DD-HH-MM-SS
        """
        time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        return time

#
