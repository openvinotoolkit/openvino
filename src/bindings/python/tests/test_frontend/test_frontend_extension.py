# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino.frontend import FrontEndManager

mock_available = True
try:
    from openvino.pybind_mock_frontend import FrontEndWrapperPaddle, FrontEndWrapperTensorflow
except Exception:
    print("No mock frontend available")
    mock_available = False

# FrontEndManager shall be initialized and destroyed after all tests finished
# This is because destroy of FrontEndManager will unload all plugins, no objects shall exist after this
fem = FrontEndManager()
ONNX_FRONTEND_NAME = "onnx"
TENSORFLOW_FRONTEND_NAME = "tf"
PADDLE_FRONTEND_NAME = "paddle"

mock_needed = pytest.mark.skipif(not mock_available, reason="mock fe is not available")


def skip_if_frontend_is_disabled(frontend_name):
    front_ends = fem.get_available_front_ends()
    if frontend_name not in front_ends:
        pytest.skip()


@mock_needed
def test_tensorflow_conversion_extension_fe_wrapper():
    skip_if_frontend_is_disabled(TENSORFLOW_FRONTEND_NAME)

    from openvino.frontend.tensorflow import ConversionExtension
    from openvino.frontend import NodeContext

    fe = FrontEndWrapperTensorflow()

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))
    assert fe.check_conversion_extension_registered("CustomConverter")


@mock_needed
def test_paddle_conversion_extension_fe_wrapper():
    skip_if_frontend_is_disabled(PADDLE_FRONTEND_NAME)

    from openvino.frontend.paddle import ConversionExtension
    from openvino.frontend import NodeContext

    fe = FrontEndWrapperPaddle()

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))
    assert fe.check_conversion_extension_registered("CustomConverter")


def test_paddle_conversion_extension():
    skip_if_frontend_is_disabled(PADDLE_FRONTEND_NAME)

    from openvino.frontend.paddle import FrontEnd
    from openvino.frontend.paddle import ConversionExtension
    from openvino.frontend import NodeContext

    fe = FrontEnd()

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))


def test_tensorflow_conversion_extension():
    skip_if_frontend_is_disabled(TENSORFLOW_FRONTEND_NAME)

    from openvino.frontend.tensorflow import FrontEnd
    from openvino.frontend.tensorflow import ConversionExtension
    from openvino.frontend import NodeContext

    fe = FrontEnd()

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))


def test_common_onnx_conversion_extension():
    skip_if_frontend_is_disabled(ONNX_FRONTEND_NAME)

    from openvino.frontend import FrontEndManager
    from openvino.frontend import ConversionExtension
    from openvino.frontend import NodeContext

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))


def test_common_tensorflow_conversion_extension():
    skip_if_frontend_is_disabled(TENSORFLOW_FRONTEND_NAME)

    from openvino.frontend import FrontEndManager
    from openvino.frontend import ConversionExtension
    from openvino.frontend import NodeContext

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework=TENSORFLOW_FRONTEND_NAME)

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))


def test_common_paddle_conversion_extension():
    skip_if_frontend_is_disabled(PADDLE_FRONTEND_NAME)

    from openvino.frontend import FrontEndManager
    from openvino.frontend import ConversionExtension
    from openvino.frontend import NodeContext

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework=PADDLE_FRONTEND_NAME)

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))
