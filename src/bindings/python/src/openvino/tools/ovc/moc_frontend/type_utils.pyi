# type: ignore
from __future__ import annotations
from openvino._pyopenvino import Type
import numpy as np
import openvino as ov
import sys as sys
__all__ = ['Type', 'is_type', 'np', 'ov', 'sys', 'to_ov_type']
def is_type(val):
    ...
def to_ov_type(val):
    ...
