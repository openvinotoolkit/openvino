# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${pyopenvino_SOURCE_DIR}/frontend/frontend_module.cmake)
frontend_module(py_onnx_frontend onnx ${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion})
