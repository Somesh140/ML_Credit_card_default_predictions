# This file contains all the utility functions that might be used in other modules

import os
from datetime import datetime


def get_current_timestamp():
    """this function return current time stamp in %Y-%m-%d-%H-%M-%s format
    """

    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
