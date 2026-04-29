# type: ignore
from __future__ import annotations
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Tensor
from openvino.tools.ovc.cli_parser import InputCutInfo as _InputCutInfo
from openvino.tools.ovc.cli_parser import input_to_input_cut_info
from openvino.tools.ovc.cli_parser import single_input_to_input_cut_info
from openvino.tools.ovc.error import Error
import logging as log
import numpy as np
import pathlib as pathlib
import sys as sys
__all__: list[str] = ['Error', 'PartialShape', 'Tensor', 'extract_input_info_from_example', 'extract_module_extensions', 'flatten_inputs', 'get_pytorch_decoder', 'get_pytorch_decoder_for_model_on_disk', 'get_value_from_list_or_dict', 'input_to_input_cut_info', 'log', 'np', 'pathlib', 'prepare_torch_inputs', 'single_input_to_input_cut_info', 'sys', 'to_torch_tensor', 'update_list_or_dict']
def _build_dynamic_shapes(inputs, input_specs = None):
    """
    Build dynamic_shapes for torch.export.export.
    
        If input_specs (list of _InputCutInfo from the 'input' parameter) is provided
        and contains shapes, dimensions marked as -1 (fully dynamic) get Dim.AUTO,
        dimensions with min/max constraints (e.g. Dimension(1, 10)) get
        Dim("dI_D", min=..., max=...), and fixed dimensions stay static.
        When no specs are given returns None so that torch.export.export produces
        a fully static graph.
    
        The input_specs list is flat (one spec per leaf tensor), while inputs may
        contain nested tuples/lists (e.g. past_key_values). pytree is used to
        flatten inputs, pair each leaf with its spec, and then unflatten the
        result back into the original structure that torch.export expects.
        
    """
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
