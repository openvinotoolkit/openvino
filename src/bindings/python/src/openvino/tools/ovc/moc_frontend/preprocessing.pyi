# type: ignore
from __future__ import annotations
from openvino._ov_api import Model
from openvino._pyopenvino import Layout
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino.preprocess import PrePostProcessor
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.moc_frontend.layout_utils import update_layout_to_dict
from openvino.tools.ovc.utils import refer_to_faq_msg
import argparse as argparse
import logging as log
import openvino._ov_api
import openvino._pyopenvino
__all__ = ['Error', 'Layout', 'Model', 'PartialShape', 'PrePostProcessor', 'apply_preprocessing', 'argparse', 'check_keys_valid', 'find_channels_dimension', 'log', 'refer_to_faq_msg', 'update_layout_is_input_flag', 'update_layout_to_dict', 'update_tensor_names_to_first_in_sorted_list']
def apply_preprocessing(ov_function: openvino._ov_api.Model, argv: argparse.Namespace):
    """
    
        Applies pre-processing of model inputs by adding appropriate operations
        On return, 'ov_function' object will be updated
        Expected 'argv.mean_scale_values' formats examples:
            a) Dict: {'inputName': {'mean': [1., 2., 3.], 'scale': [2., 4., 8.]}}
            b) List: list(np.array([(np.array([1., 2., 3.]), np.array([2., 4., 6.])),
                         (np.array([7., 8., 9.]), np.array([5., 6., 7.])))
        Expected 'argv.layout_values' format examples:
            a) Specific layouts for inputs and outputs
            { 'input1': {
                     'source_layout': 'nchw',
                     'target_layout': 'nhwc'
                 },
                 'output2': {
                     'source_layout': 'nhwc'
                 }
            }
            b) Layout for single input: {'': {'source_layout': 'nchw'}}
        :param: ov_function OV function for applying mean/scale pre-processing
        :param: argv Parsed command line arguments
        
    """
def check_keys_valid(ov_function: openvino._ov_api.Model, dict_to_validate: dict, search_outputs: bool):
    """
    
        Internal function: checks if keys from cmd line arguments correspond to ov_function's inputs/outputs
        Throws if some key is not found
        Throws if some different keys point to the same actual input/output
        
    """
def find_channels_dimension(shape: openvino._pyopenvino.PartialShape, num_channels: int, name: str, layout_values):
    """
    
        Internal function. Finds dimension index matching with expected channels number
        Raises error if there is no candidates or number of candidates is > 1
        :param: shape Parameter's partial shape
        :param: num_channels Number of channels to find in shape
        :param: name Parameter's name, used for Error-handling purposes
        :param: layout_values Existing source/target layout items specified by user
        :return: updated layout items with guessed layouts
        
    """
def update_layout_is_input_flag(ov_function: openvino._ov_api.Model, layout_values: dict):
    """
    
        Internal function: updates layout_values with flag whether each layout belongs to input or to output
        
    """
def update_tensor_names_to_first_in_sorted_list(values_dict: dict, ov_function: openvino._ov_api.Model):
    ...
