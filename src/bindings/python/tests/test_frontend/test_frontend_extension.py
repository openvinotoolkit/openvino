# -*- coding: utf-8 -*-  noqa: E999
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
from openvino.frontend import FrontEndManager

TENSORFLOW_FRONTEND_NAME = "tf"
PADDLE_FRONTEND_NAME = "paddle"

imported_frontends = []

try:
    from pybind_mock_frontend import FrontEndWrapperTensorflow
    imported_frontends.append(TENSORFLOW_FRONTEND_NAME)
except Exception:
    pass

try:
    from pybind_mock_frontend import FrontEndWrapperPaddle
    imported_frontends.append(PADDLE_FRONTEND_NAME)
except Exception:
    pass

# FrontEndManager shall be initialized and destroyed after all tests finished
# This is because destroy of FrontEndManager will unload all plugins, no objects shall exist after this
fem = FrontEndManager()


def skip_if_frontend_is_disabled(frontend):
    if frontend not in imported_frontends:
        return pytest.mark.skip(
            reason=f"Cannot import frontend {frontend}.  Check paths in:"
                   f" LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','')}"
                   f", PYTHONPATH={os.environ.get('PYTHONPATH','')}")

    return pytest.mark.skipif(frontend not in fem.get_available_front_ends(), reason=f"Frontend {frontend} is disabled")


def skip_if_tensorflow_not_install_by_wheel_pkg():
    import_failed = False
    try:
        from openvino.frontend.tensorflow import ConversionExtension
    except ImportError:
        import_failed = True

    return pytest.mark.skipif(import_failed, reason="Tensorflow conversion not installed by wheel pkg.")


@skip_if_frontend_is_disabled(TENSORFLOW_FRONTEND_NAME)
@skip_if_tensorflow_not_install_by_wheel_pkg()
def test_tensorflow_conversion_extension_fe_wrapper():
    from openvino.frontend.tensorflow import ConversionExtension
    from openvino.frontend import NodeContext

    fe = FrontEndWrapperTensorflow()

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))
    assert fe.check_conversion_extension_registered("CustomConverter")


@skip_if_frontend_is_disabled(PADDLE_FRONTEND_NAME)
def test_paddle_conversion_extension_fe_wrapper():
    from openvino.frontend.paddle import ConversionExtension
    from openvino.frontend import NodeContext

    fe = FrontEndWrapperPaddle()

    def custom_converter(node: NodeContext):
        node.get_input(0)
        node.get_attribute("alpha")

    fe.add_extension(ConversionExtension("CustomConverter", custom_converter))
    assert fe.check_conversion_extension_registered("CustomConverter")
