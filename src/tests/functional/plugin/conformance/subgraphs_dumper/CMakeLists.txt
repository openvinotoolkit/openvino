# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_subgraphs_dumper)

list(APPEND LIBRARIES
        gflags
        op_conformance_utils
        openvino::core::dev
)

if(ENABLE_OV_IR_FRONTEND)
    list(APPEND COMPILE_DEFINITIONS ENABLE_OV_IR_FRONTEND)
endif()

if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND COMPILE_DEFINITIONS ENABLE_OV_ONNX_FRONTEND)
endif()

if(ENABLE_OV_PADDLE_FRONTEND)
    list(APPEND COMPILE_DEFINITIONS ENABLE_OV_PADDLE_FRONTEND)
endif()

if(ENABLE_OV_TF_FRONTEND)
    list(APPEND COMPILE_DEFINITIONS ENABLE_OV_TF_FRONTEND)
endif()

if(ENABLE_OV_TF_LITE_FRONTEND)
    list(APPEND COMPILE_DEFINITIONS ENABLE_OV_TF_LITE_FRONTEND)
endif()

if(ENABLE_OV_PYTORCH_FRONTEND)
    list(APPEND COMPILE_DEFINITIONS ENABLE_OV_PYTORCH_FRONTEND)
endif()

# add subgraphs_dumpers tool
ov_add_test_target(
        NAME ${TARGET_NAME}
        TYPE EXECUTABLE
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src
        INCLUDES
            PRIVATE
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PRIVATE
                ${LIBRARIES}
        DEPENDENCIES
            ov_frontends
            ${LIBRARIES}
        DEFINES
            ${COMPILE_DEFINITIONS}
        ADD_CPPLINT
)

# add subgraphs_dumpers lib to get API
ov_add_target(
        NAME "${TARGET_NAME}_util"
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PUBLIC
                ${LIBRARIES}
        EXCLUDED_SOURCE_PATHS
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/src/main.cpp
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/include/gflag_config.hpp
        DEPENDENCIES
            ov_frontends
            ${LIBRARIES}
        DEFINES
            ${COMPILE_DEFINITIONS}
        ADD_CPPLINT
)

ov_build_target_faster(${TARGET_NAME} UNITY)
