# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(generic)

if (NGRAPH_PDPD_FRONTEND_ENABLE)
    add_subdirectory(paddlepaddle)
endif()

if (NGRAPH_TF_FRONTEND_ENABLE)
    add_subdirectory(tensorflow)
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE)
    add_subdirectory(onnx_common)
    add_subdirectory(onnx_import)
endif()

if (NGRAPH_ONNX_EDITOR_ENABLE)
    add_subdirectory(onnx_editor)
endif()
