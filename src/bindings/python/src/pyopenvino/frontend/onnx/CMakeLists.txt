# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${PYTHON_SOURCE_DIR}/pyopenvino/frontend/frontend_module.cmake)
frontend_module(py_onnx_frontend onnx pyopenvino_${PYTHON_VERSION})
