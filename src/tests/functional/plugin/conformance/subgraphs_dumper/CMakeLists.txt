# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_subgraphs_dumper)
include(sources.cmake)

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
        SOURCES
            ${OV_SUBGRAPHS_DUMPER_COMMON_SOURCES}
            ${OV_SUBGRAPHS_DUMPER_CLI_SOURCES}
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
)

# add subgraphs_dumpers lib to get API
ov_add_target(
        NAME "${TARGET_NAME}_util"
        TYPE STATIC
        SOURCES ${OV_SUBGRAPHS_DUMPER_COMMON_SOURCES}
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PUBLIC
                ${LIBRARIES}
        DEPENDENCIES
            ov_frontends
            ${LIBRARIES}
        DEFINES
            ${COMPILE_DEFINITIONS}
)

ov_build_target_faster(${TARGET_NAME} UNITY)
