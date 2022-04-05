# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

add_subdirectory(common)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(onnx)
endif()

if(ENABLE_OV_PADDLE_FRONTEND)
    add_subdirectory(paddle)
endif()

if(ENABLE_OV_IR_FRONTEND)
    add_subdirectory(ir)
endif()

if(ENABLE_OV_TF_FRONTEND)
    add_subdirectory(tensorflow)
endif()
