# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FRONTEND_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend")

add_subdirectory(frontend_manager)

if (NGRAPH_ONNX_IMPORT_ENABLE)
    add_subdirectory(onnx)
endif()

if (NGRAPH_PDPD_FRONTEND_ENABLE)
    add_subdirectory(paddlepaddle)
endif()
