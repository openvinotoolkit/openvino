# type: ignore
"""
Helper functions for validating user input.
"""
from __future__ import annotations
from openvino.exceptions import UserInputError
from typing import Any
import logging as logging
import numpy as np
__all__ = ['Any', 'UserInputError', 'assert_list_of_ints', 'check_valid_attribute', 'check_valid_attributes', 'is_non_negative_value', 'is_positive_value', 'log', 'logging', 'np']
def _check_value(op_name, attr_key, value, val_type, cond = None):
    """
    Check whether provided value satisfies specified criteria.
    
        :param      op_name:        The operator name which attributes are checked.
        :param      attr_key:       The attribute name.
        :param      value:          The value to check.
        :param      val_type:       Required value type.
        :param      cond:           The optional function running additional checks.
    
        :raises     UserInputError:
    
        returns:    True if attribute satisfies all criterias. Otherwise False.
        
    """
def assert_list_of_ints(value_list: typing.Iterable[int], message: str) -> None:
    """
    Verify that the provided value is an iterable of integers.
    """
def check_valid_attribute(op_name, attr_dict, attr_key, val_type, cond = None, required = False):
    """
    Check whether specified attribute satisfies given criteria.
    
        :param  op_name:    The operator name which attributes are checked.
        :param attr_dict:   Dictionary containing key-value attributes to check.
        :param attr_key:    Key value for validated attribute.
        :param val_type:    Value type for validated attribute.
        :param cond:        Any callable wich accept attribute value and returns True or False.
        :param required:    Whether provided attribute key is not required. This mean it may be missing
                            from provided dictionary.
    
        :raises     UserInputError:
    
        returns True if attribute satisfies all criterias. Otherwise False.
        
    """
def check_valid_attributes(op_name, attributes, requirements):
    """
    Perform attributes validation according to specified type, value criteria.
    
        :param  op_name:        The operator name which attributes are checked.
        :param  attributes:     The dictionary with user provided attributes to check.
        :param  requirements:   The list of tuples describing attributes' requirements. The tuple should
                                contain following values:
                                (attr_name: str,
                                is_required: bool,
                                value_type: Type,
                                value_condition: Callable)
    
        :raises     UserInputError:
    
        :returns True if all attributes satisfies criterias. Otherwise False.
        
    """
def is_non_negative_value(value):
    """
    Determine whether the specified x is non-negative value.
    
        :param      value:    The value to check.
    
        returns   True if the specified x is non-negative value, False otherwise.
        
    """
def is_positive_value(value):
    """
    Determine whether the specified x is positive value.
    
        :param      value:    The value to check.
    
        returns   True if the specified x is positive value, False otherwise.
        
    """
log: logging.Logger  # value = <Logger openvino.utils.input_validation (INFO)>
