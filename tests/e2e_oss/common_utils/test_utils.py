import logging as log
from typing import Union
import tensorflow as tf


import numpy as np
from openvino.runtime import Dimension, PartialShape


def get_shapes_from_data(input_data, api_version='1') -> dict:
    shapes = {}
    for input_layer in input_data:
        if api_version == '2':
            shapes[input_layer] = PartialShape(input_data[input_layer].shape)
        else:
            shapes[input_layer] = input_data[input_layer].shape
    return shapes


def convert_shapes_to_partial_shape(shapes: dict) -> dict:
    partial_shape = {}
    for layer, shape in shapes.items():
        dimension_tmp = []
        for item in shape:
            dimension_tmp.append(Dimension(item[0], item[1])) if type(item) == list else dimension_tmp.append(
                Dimension(item))
        partial_shape[layer] = PartialShape(dimension_tmp)
    return partial_shape


def remove_mo_args(mo_args_to_skip: Union[list, str], mo_cmd):
    """
    This function deletes arguments from MO cmd line

    :param mo_args_to_skip: mo arguments to delete
    :param mo_cmd: MO command line that is supposed to be reconfigured
    """
    mo_args_to_skip = mo_args_to_skip if isinstance(mo_args_to_skip, list) else mo_args_to_skip.split(',')

    for mo_arg in mo_args_to_skip:
        if mo_arg in mo_cmd:
            log.info('Deleting argument from MO cmd: {}'.format(mo_arg))
            del mo_cmd[mo_arg]

    return mo_cmd


def prepare_input(input_shape, input_type):
    np.random.seed(0)
    if input_type in [np.float32, np.float64]:
        return np.random.randint(-2, 2, size=input_shape).astype(input_type)
    elif input_type in [np.int8, np.int16, np.int32, np.int64]:
        return np.random.randint(-5, 5, size=input_shape).astype(input_type)
    elif input_type in [np.uint8, np.uint16]:
        return np.random.randint(0, 5, size=input_shape).astype(input_type)
    elif input_type in [str]:
        return np.broadcast_to("Some string", input_shape)
    elif input_type in [bool]:
        return np.random.randint(0, 2, size=input_shape).astype(input_type)
    else:
        assert False, "Unsupported type {}".format(input_type)


def prepare_inputs(inputs_info):
    inputs = []
    for input_shape, input_type in inputs_info:
        inputs.append(prepare_input(input_shape, input_type))
    return inputs


def get_inputs_info(model_obj):
    inputs_info = []
    for input_info in model_obj.inputs:
        input_shape = []
        try:
            for dim in input_info.shape.as_list():
                if dim is None:
                    input_shape.append(1)
                else:
                    input_shape.append(dim)
        except ValueError:
            # unknown rank case
            pass
        type_map = {
            tf.float64: np.float64,
            tf.float32: np.float32,
            tf.int8: np.int8,
            tf.int16: np.int16,
            tf.int32: np.int32,
            tf.int64: np.int64,
            tf.uint8: np.uint8,
            tf.uint16: np.uint16,
            tf.string: str,
            tf.bool: bool,
        }
        if input_info.dtype not in type_map:
            continue
        assert input_info.dtype in type_map, "Unsupported input type: {}".format(input_info.dtype)
        inputs_info.append((input_shape, type_map[input_info.dtype]))

    return inputs_info


def name_aligner(infer_result, reference, xml=None):
    """
    Function name_aligner aligns names for inference and reference outputs if number of their outputs == 1
    """
    if len(infer_result.keys()) == 1 == len(reference.keys()):
        log.info("Renaming inferred output layer {} to referenced output layer {}".format(
            list(infer_result.keys())[0], list(reference.keys())[0]))
        infer_result[next(iter(reference))] = infer_result.pop(next(iter(infer_result)))

    return infer_result, reference
