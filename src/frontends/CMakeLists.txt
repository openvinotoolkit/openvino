# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") 
    ie_add_compiler_flags(-Wno-sign-compare)
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

if(ENABLE_OV_PYTORCH_FRONTEND)
    add_subdirectory(pytorch)
endif()

if(ENABLE_OV_IR_FRONTEND)
    add_subdirectory(ir)
endif()

if(ENABLE_OV_TF_FRONTEND OR ENABLE_OV_TF_LITE_FRONTEND)
    add_subdirectory(tensorflow_common)
endif()

if(ENABLE_OV_TF_FRONTEND)
    add_subdirectory(tensorflow)
endif()

if (ENABLE_OV_TF_LITE_FRONTEND)
    add_subdirectory(tensorflow_lite)
endif()