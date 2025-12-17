# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest


def pytest_configure(config):

    # register additional markers
    config.addinivalue_line("markers", "skip_on_cpu: Skip test on CPU")
    config.addinivalue_line("markers", "skip_on_gpu: Skip test on GPU")
    config.addinivalue_line("markers", "skip_on_hetero: Skip test on HETERO")
    config.addinivalue_line("markers", "skip_on_template: Skip test on TEMPLATE")
    config.addinivalue_line("markers", "onnx_coverage: Collect ONNX operator coverage")
    config.addinivalue_line("markers", "template_extension")
    config.addinivalue_line("markers", "dynamic_library: Runs tests only in dynamic libraries case")

    if sys.platform == "win32":
        # Adjust 'bin/intel64/Release' to your actual relative path
        # From src/bindings/python/tests/conftest.py to bin/intel64/Release
        # src/bindings/python/tests -> src/bindings/python -> src/bindings -> src -> root -> bin
        build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "bin", "intel64", "Release"))

        if os.path.exists(build_dir):
            print(f"DEBUG: Adding DLL directory: {build_dir}")
            try:
                os.add_dll_directory(build_dir)
            except AttributeError:
                # Python < 3.8, fallback to env var
                os.environ['PATH'] = build_dir + os.pathsep + os.environ['PATH']
        else:
            print(f"WARNING: Build directory not found at {build_dir}")


@pytest.fixture(scope="session")
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
