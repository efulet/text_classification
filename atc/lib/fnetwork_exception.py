"""
@created_at 2014-11-22
@author Exequiel Fuentes <efulet@gmail.com>
"""


class FNetworkException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
