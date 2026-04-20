# type: ignore
from __future__ import annotations
"""
openvino exceptions hierarchy. All exceptions are descendants of OVError.
"""
__all__: list[str] = ['OVError', 'OVTypeError', 'UserInputError']
class OVError(Exception):
    """
    Base class for OV exceptions.
    """
class OVTypeError(OVError, TypeError):
    """
    Type mismatch error.
    """
class UserInputError(OVError):
    """
    User provided unexpected input.
    """
