# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import glob
import re
import sys

# ticket 95904
#import paddle
#from paddle.jit import to_static
#from paddle.static import InputSpec

import pytest

from openvino.frontend import FrontEndManager
from openvino import shutdown

PADDLE_FRONTEND_NAME = "paddle"
paddle_relu6_model_basename = "relu6"
paddle_relu6_model_filename = paddle_relu6_model_basename + ".pdmodel"
paddle_concat_model_basename = "concat"
paddle_concat_model_filename = paddle_concat_model_basename + ".pdmodel"
fem = FrontEndManager()


def skip_if_paddle_frontend_is_disabled():
    front_ends = fem.get_available_front_ends()
    if PADDLE_FRONTEND_NAME not in front_ends:
        pytest.skip()


# ticket 95904
#def create_paddle_model():
    #@to_static()
    #def test(x):
    #    return paddle.nn.functional.relu6(x)
    #x_spec = InputSpec(shape=[None, 3], dtype="float32", name="x")
    #paddle.jit.save(test, path=paddle_relu6_model_basename, input_spec=[x_spec, ])


# ticket 95904
#def create_concat_model():
    #@to_static()
    #def test(x, y):
    #    return paddle.concat([x, y], axis=0)
    #x_spec = InputSpec(shape=[None, 3], dtype="float32", name="x")
    #y_spec = InputSpec(shape=[None, 3], dtype="float32", name="y")
    #paddle.jit.save(test, path=paddle_concat_model_basename, input_spec=[x_spec, y_spec])


# ticket 95904
#def setup_module():
    #create_paddle_model()
    #create_concat_model()


def teardown_module():
    os.remove(paddle_relu6_model_filename)
    os.remove(paddle_concat_model_filename)
    shutdown()


@pytest.mark.skip(reason="Paddlepaddle has incompatible protobuf. Ticket: 95904")
def test_paddle_conversion_extension():
    skip_if_paddle_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.paddle import ConversionExtension
    from openvino.frontend import NodeContext
    import openvino.opset8 as ops

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


@pytest.mark.skip(reason="Paddlepaddle has incompatible protobuf. Ticket: 95904")
def test_op_extension_via_paddle_extension_set_attrs_values():
    skip_if_paddle_frontend_is_disabled()

    # use specific (openvino.frontend.paddle) import here
    from openvino.frontend.paddle import OpExtension
    from openvino import Core

    ie = Core()

    # check the model is valid
    model = ie.read_model(paddle_relu6_model_filename)
    assert model

    # add extensions
    ie.add_extension(OpExtension("Clamp", "relu6", ["X"], ["Out"], {}, {"min": 0.0, "max": 6.0}))

    model = ie.read_model(paddle_relu6_model_filename)
    assert model


@pytest.mark.skip(reason="Paddlepaddle has incompatible protobuf. Ticket: 95904")
def test_op_extension_via_frontend_extension_set_attrs_values():
    skip_if_paddle_frontend_is_disabled()

    # use common (openvino.frontend) import here
    from openvino.frontend import OpExtension
    from openvino import Core

    ie = Core()
    # check the model is valid
    model = ie.read_model(paddle_relu6_model_filename)
    assert model

    # add extensions
    ie.add_extension(OpExtension("Clamp", "relu6", ["X"], ["Out"], {}, {"min": 0.0, "max": 6.0}))

    model = ie.read_model(paddle_relu6_model_filename)
    assert model


def get_builtin_extensions_path():
    base_paths = [Path(__file__).parent.parent.parent.parent]
    repo_dir = os.environ.get("REPO_DIR")
    if repo_dir:
        base_paths.append(repo_dir)

    for base_path in base_paths:
        paths = glob.glob(os.path.join(base_path, "bin", "*", "*", "*test_builtin_extensions*"))
        for path in paths:
            if re.search(r"(lib)?test_builtin_extensions.?\.(dll|so)", path):
                return path
    raise RuntimeError("Unable to find test_builtin_extensions")


@pytest.mark.skip(reason="Paddlepaddle has incompatible protobuf. Ticket: 95904")
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


@pytest.mark.skip(reason="Paddlepaddle has incompatible protobuf. Ticket: 95904")
def test_so_extension_via_frontend_decode_input_model():
    skip_if_paddle_frontend_is_disabled()

    def load_decoded_model():
        fe = fem.load_by_framework(framework=PADDLE_FRONTEND_NAME)
        fe.add_extension(get_builtin_extensions_path())
        in_model = fe.load(paddle_relu6_model_filename)
        return fe.decode(in_model)

    decoded_model = load_decoded_model()
    assert decoded_model
