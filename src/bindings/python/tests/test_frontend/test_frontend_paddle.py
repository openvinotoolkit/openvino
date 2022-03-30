# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from cv2 import threshold
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec
import paddle.nn.functional as F
import numpy as np
import pytest
from pathlib import Path
from itertools import chain

from openvino.frontend import FrontEndManager

PADDLE_FRONTEND_NAME = 'paddle'
paddle_relu6_model_basename = 'relu6'
paddle_relu6_model_filename = paddle_relu6_model_basename + '.pdmodel'
paddle_concat_model_basename = 'concat'
paddle_concat_model_filename = paddle_concat_model_basename + '.pdmodel'
fem = FrontEndManager()


def skip_if_paddle_frontend_is_disabled():
    front_ends = fem.get_available_front_ends()
    if PADDLE_FRONTEND_NAME not in front_ends:
        pytest.skip()


def create_paddle_model():
    @to_static()
    def test(x):
        return F.relu6(x)
    x_spec = InputSpec(shape=[None, 3], dtype='float32', name='x')
    paddle.jit.save(test, path=paddle_relu6_model_basename, input_spec=[x_spec, ])


def create_concat_model():
    @to_static()
    def test(x, y):
        return paddle.concat([x, y], axis=0)
    x_spec = InputSpec(shape=[None, 3], dtype='float32', name='x')
    y_spec = InputSpec(shape=[None, 3], dtype='float32', name='y')
    paddle.jit.save(test, path=paddle_concat_model_basename, input_spec=[x_spec, y_spec])


def setup_module():
    create_paddle_model()
    create_concat_model()


def teardown_module():
    os.remove(paddle_relu6_model_filename)
    os.remove(paddle_concat_model_filename)


def test_paddle_conversion_extension():
    skip_if_paddle_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.paddle import ConversionExtension
    from openvino.frontend import NodeContext
    import openvino.runtime.opset8 as ops

    fe = fem.load_by_model(paddle_relu6_model_filename)
    assert fe
    assert fe.get_name() == "paddle"

    invoked = False

    def custom_converter(node: NodeContext):
        nonlocal invoked
        invoked = True
        x = node.get_input("X")
        threshold = node.get_attribute("threshold")
        add = ops.clamp(x, 0.0, threshold)
        return {"Out": [add.output(0)]}

    fe.add_extension(ConversionExtension("relu6", custom_converter))
    input_model = fe.load(paddle_relu6_model_filename)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked


def test_op_extension_via_paddle_extension_set_attrs_values():
    skip_if_paddle_frontend_is_disabled()

    # use specific (openvino.frontend.paddle) import here
    from openvino.frontend.paddle import OpExtension
    from openvino.runtime import Core

    ie = Core()

    # check the model is valid
    model = ie.read_model(paddle_relu6_model_filename)
    assert model

    # add extensions
    ie.add_extension(OpExtension("Clamp", "relu6", ["X"], ["Out"], {}, {"min": 0.0, "max": 6.0}))

    model = ie.read_model(paddle_relu6_model_filename)
    assert model


def test_op_extension_via_frontend_extension_set_attrs_values():
    skip_if_paddle_frontend_is_disabled()

    # use common (openvino.frontend) import here
    from openvino.frontend import OpExtension
    from openvino.runtime import Core

    ie = Core()
    # check the model is valid
    model = ie.read_model(paddle_relu6_model_filename)
    assert model

    # add extensions
    ie.add_extension(OpExtension("Clamp", "relu6", ["X"], ["Out"], {}, {"min": 0.0, "max": 6.0}))

    model = ie.read_model(paddle_relu6_model_filename)
    assert model


# could not find a suitable case because attribute type almost is not same
# such as ov use int64 but paddle use int32, ov use double but paddle use float
# def test_op_extension_via_frontend_extension_map_attributes():
#     skip_if_paddle_frontend_is_disabled()
#
#     # use common (openvino.frontend) import here
#     from openvino.frontend import OpExtension
#     from openvino.runtime import Core
#
#     ie = Core()
#     # check the model is valid
#     model = ie.read_model(paddle_concat_model_filename)
#     assert model
#
#     # add extensions
#     ie.add_extension(OpExtension("Concat", "concat", ["X"], ["Out"], {"axis": "axis"}))
#
#     model = ie.read_model(paddle_concat_model_filename)
#     assert model


def get_builtin_extensions_path():
    win_folder_path = Path(__file__).parent.parent.parent.parent
    linux_folder_path = win_folder_path.joinpath("lib")
    for lib_path in chain(win_folder_path.glob("*.dll"), linux_folder_path.glob("*.so")):
        if "libtest_builtin_extensions_1" in lib_path.name:
            return str(lib_path)
    return ""


@pytest.mark.skipif(len(get_builtin_extensions_path()) == 0,
                    reason="The extension library path was not found")
def test_so_extension_via_frontend_convert_input_model():
    skip_if_paddle_frontend_is_disabled()

    def load_model():
        fe = fem.load_by_framework(framework=PADDLE_FRONTEND_NAME)
        fe.add_extension(get_builtin_extensions_path())
        in_model = fe.load(paddle_relu6_model_filename)
        return fe.convert(in_model)

    model = load_model()

    assert any(op.get_type_name() == "Relu" for op in model.get_ops())
    assert all(op.get_type_name() != "Clamp" for op in model.get_ops())


@pytest.mark.skipif(len(get_builtin_extensions_path()) == 0,
                    reason="The extension library path was not found")
def test_so_extension_via_frontend_decode_input_model():
    skip_if_paddle_frontend_is_disabled()

    def load_decoded_model():
        fe = fem.load_by_framework(framework=PADDLE_FRONTEND_NAME)
        fe.add_extension(get_builtin_extensions_path())
        in_model = fe.load(paddle_relu6_model_filename)
        return fe.decode(in_model)

    decoded_model = load_decoded_model()
    assert decoded_model
