# type: ignore
from __future__ import annotations
from openvino._pyopenvino import Dimension
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Type
from packaging.version import Version
from packaging.version import parse
import logging as log
import numpy as np
import openvino._pyopenvino
import sys as sys
__all__ = ['Dimension', 'PartialShape', 'Type', 'Version', 'create_generic_function_from_keras_model', 'create_tf_graph_iterator', 'extract_model_graph', 'get_concrete_func', 'get_environment_setup', 'get_imported_module_version', 'get_input_spec_from_model', 'get_signature_from_input', 'get_signature_from_input_signature', 'get_static_shape', 'is_variable', 'log', 'model_is_graph_iterator', 'np', 'parse', 'partial_shape_to_list', 'sys', 'tf_type_to_ov_type', 'trace_tf_model', 'trace_tf_model_if_needed', 'type_supported_by_tf_fe']
def create_generic_function_from_keras_model(keras_model):
    ...
def create_tf_graph_iterator(input_model, placeholder_shapes, placeholder_data_types, example_input, share_weights):
    ...
def extract_model_graph(argv):
    ...
def get_concrete_func(tf_function, example_input, input_needs_packing, error_message, use_example_input = True):
    """
    
        Runs tracing of TF function and returns a concrete function.
    
        :param tf_function: TF function that needs to be traced.
        :param example_input: Example of function input.
        :param input_needs_packing: determines if input needs to be packed in a list before passing to TF function.
        It is used when original function was wrapped in outer TF function, which changes function signature.
        In this case wrapper TF function always expects list of inputs which are unpacked inside subfunction.
        So list/tuple are treated as multiple inputs of original model.
        Non list/tuple are treated as single input, and it needs packing to a list,
        as wrapper function always expect list of inputs.
        :param error_message: Error message which should be shown in case of tracing error.
        :param use_example_input: Determines if example_input should be used.
    
        :returns: Object of type tf.types.experimental.ConcreteFunction.
        
    """
def get_environment_setup(framework):
    """
    
        Get environment setup such as Python version, TensorFlow version
        :param framework: framework name
        :return: a dictionary of environment variables
        
    """
def get_imported_module_version(imported_module):
    """
    
        Get imported module version
        :return: version(str) or raise AttributeError exception
        
    """
def get_input_spec_from_model(model, input_shapes = None):
    ...
def get_signature_from_input(keras_model):
    ...
def get_signature_from_input_signature(keras_model):
    ...
def get_static_shape(shape: [openvino._pyopenvino.PartialShape, list, tuple], dynamic_value = None):
    ...
def is_variable(func_input, captures):
    ...
def model_is_graph_iterator(model):
    ...
def partial_shape_to_list(partial_shape: openvino._pyopenvino.PartialShape):
    ...
def tf_type_to_ov_type(val):
    ...
def trace_tf_model(model, input_shapes, input_types, example_input):
    ...
def trace_tf_model_if_needed(input_model, placeholder_shapes, placeholder_data_types, example_input):
    ...
def type_supported_by_tf_fe(input_model):
    ...
