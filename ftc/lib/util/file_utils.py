"""
@created_at 2015-01-18
@author Exequiel Fuentes Lettura <efulet@gmail.com>
"""


import os


class FileUtils:
    """"""
    
    DB_BASE_DIR_NAME = "db"
    
    LOG_BASE_DIR_NAME = "log"
    LOG_FILENAME = "irws.log"
    
    def __init__(self):
        """"""
    
    def db_path(self):
        pathfile = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(pathfile, "..", "..", self.DB_BASE_DIR_NAME)
    
    def log_path(self):
        pathfile = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(pathfile, "..", "..", "..", self.LOG_BASE_DIR_NAME)
    
    def log_file(self):
        return os.path.join(self.log_path(), self.LOG_FILENAME)
