"""
@created_at 2015-01-18
@author Exequiel Fuentes Lettura <efulet@gmail.com>
"""


class NetworkException(Exception):
    """"""
    
    def __init__(self, value):
        """"""
        self.value = value

    def __str__(self):
        """"""
        return repr(self.value)
