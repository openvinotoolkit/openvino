# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_npu_transformations)
set(TARGET_ALIAS openvino::npu_transformations)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shared_weights_assigner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shared_weights_context_extractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shared_weights_producer.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shared_weights_assigner.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shared_weights_context_extractor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shared_weights_producer.hpp
)

# OBJECT library: objects are embedded into consumers, so nothing has to be installed or exported
add_library(${TARGET_NAME} OBJECT ${SOURCES} ${HEADERS})
add_library(${TARGET_ALIAS} ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev
    openvino::core::dev
    openvino::util
    openvino::npu_logger_utils
)

set_target_properties(${TARGET_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO}
)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
