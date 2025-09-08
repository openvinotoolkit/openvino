# type: ignore
from __future__ import annotations
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Tensor
from openvino.tools.ovc.cli_parser import InputCutInfo as _InputCutInfo
from openvino.tools.ovc.cli_parser import single_input_to_input_cut_info
from openvino.tools.ovc.error import Error
import logging as log
import numpy as np
import pathlib as pathlib
import sys as sys
__all__ = ['Error', 'PartialShape', 'Tensor', 'extract_input_info_from_example', 'extract_module_extensions', 'flatten_inputs', 'get_pytorch_decoder', 'get_pytorch_decoder_for_model_on_disk', 'get_value_from_list_or_dict', 'log', 'np', 'pathlib', 'prepare_torch_inputs', 'single_input_to_input_cut_info', 'sys', 'to_torch_tensor', 'update_list_or_dict']
def extract_input_info_from_example(args, inputs):
    ...
def extract_module_extensions(args):
    ...
def flatten_inputs(inputs, names = None):
    ...
def get_pytorch_decoder(model, example_inputs, args):
    ...
def get_pytorch_decoder_for_model_on_disk(argv, args):
    ...
def get_value_from_list_or_dict(container, name, idx):
    ...
def prepare_torch_inputs(example_inputs):
    ...
def to_torch_tensor(tensor):
    ...
def update_list_or_dict(container, name, idx, value):
    ...
