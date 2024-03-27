# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Iterable, Union

import numpy as np
from openvino.tools.ovc.error import Error

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.ovc.telemetry_stub as tm

dynamic_dimension = np.ma.masked


def refer_to_faq_msg(question_num: int):
    try:
        t = tm.Telemetry()
        t.send_event('ovc', 'error_info', "faq:" + str(question_num))
    except Exception:
        # Telemetry can be not initialized if it is used in MO IR Reader
        pass

    return '\n For more information please refer to Model Conversion API FAQ, question #{0}. ' \
           '(https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html' \
           '?question={0}#question-{0})'.format(question_num)


def get_mo_root_dir():
    """
    Return the absolute path to the Model Conversion API root directory (where mo folder is located)
    :return: path to the MO root directory
    """
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(os.path.realpath(__file__))), os.pardir))


def check_values_equal(val1, val2):
    # This method is needed to check equality of values where some values can be None
    if val1 is None and val2 is None:
        return True
    if val1 is None:
        return False
    if val2 is None:
        return False
    return val1 == val2


np_map_cast = {bool: lambda x: bool_cast(x),
               np.int8: lambda x: np.int8(x),
               np.int16: lambda x: np.int16(x),
               np.int32: lambda x: np.int32(x),
               np.int64: lambda x: np.int64(x),
               np.uint8: lambda x: np.uint8(x),
               np.uint16: lambda x: np.uint16(x),
               np.uint32: lambda x: np.uint32(x),
               np.uint64: lambda x: np.uint64(x),
               np.float16: lambda x: np.float16(x),
               np.float32: lambda x: np.float32(x),
               np.double: lambda x: np.double(x),
               str: lambda x: str(x)}


def bool_cast(x):
    if isinstance(x, str):
        return False if x.lower() in ['false', '0'] else True if x.lower() in ['true', '1'] else 'unknown_boolean_cast'
    else:
        return bool(x)


def mo_array(value: Union[Iterable[Union[float, int]], float, int], dtype=None) -> np.ndarray:
    """
    This function acts in a same way as np.array except for the case when dtype is not provided
    and np.array return fp64 array this function returns fp32 array
    """
    x = np.array(value, dtype=dtype)
    if not isinstance(value, np.ndarray) and x.dtype == np.float64 and dtype != np.float64:
        x = x.astype(np.float32)
    return x


def validate_batch_in_shape(shape, layer_name: str):
    """
    Raises Error #39 if shape is not valid for setting batch size
    Parameters
    ----------
    shape: current shape of layer under validation
    layer_name: name of layer under validation
    """
    if len(shape) == 0 or (shape[0] is not dynamic_dimension and shape[0] not in (-1, 0, 1)):
        raise Error(('The input layer {} has a shape {} defined in the model. \n\n' +
                     'When you use "batch" option, Model Conversion API applies its value to the first ' +
                     'element of the shape if it is equal to -1, 0 or 1. Otherwise, this is the ambiguous ' +
                     'situation - it is not known in advance whether the layer has the batch ' +
                     'dimension or not.\n\n For example, you want to set batch dimension equals 100 ' +
                     'for the input layer "data" with shape (10,34). Although you can not use "batch", ' +
                     'you should pass "input_shape=[100,34]" instead of "batch=100". \n\n' +
                     'You can also specify batch dimension by setting "layout". \n\n')
                    .format(layer_name, shape))


def get_ir_version():
    """
    Default IR version.
    :return: the IR version
    """
    return 11
