# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

add_subdirectory(frontend/shared)

if (ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(onnx)
endif()
