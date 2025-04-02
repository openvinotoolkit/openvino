# type: ignore
from __future__ import annotations
import re as re
__all__ = ['BasicError', 'Error', 'FrameworkError', 'InternalError', 'classify_error_type', 're']
class BasicError(Exception):
    """
     Base class for all exceptions in Model Conversion API
    
            It operates like Exception but when it is converted to str,
            it formats string as args[0].format(*args[1:]), where
            args are arguments provided when an exception instance is
            created.
        
    """
    def __str__(self):
        ...
class Error(BasicError):
    """
     User-friendly error: raised when the error on the user side. 
    """
class FrameworkError(BasicError):
    """
     User-friendly error: raised when the error on the framework side. 
    """
class InternalError(BasicError):
    """
     Not user-friendly error: user cannot fix it and it points to the bug inside MO. 
    """
def classify_error_type(e):
    ...
