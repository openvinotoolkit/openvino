# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_onnx_common")

add_library(${TARGET_NAME} STATIC

    # Sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_model_validator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp

    # Headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_common/onnx_model_validator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_common/parser.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_common/utils.hpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

set(ONNX_COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONNX_COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(${TARGET_NAME}
    PUBLIC $<BUILD_INTERFACE:${ONNX_COMMON_INCLUDE_DIR}>
           $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
    PRIVATE ${ONNX_COMMON_SRC_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime openvino::util)

ov_link_system_libraries(${TARGET_NAME} PUBLIC onnx_proto onnx)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
