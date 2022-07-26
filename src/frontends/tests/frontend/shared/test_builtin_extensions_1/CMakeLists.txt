# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "test_builtin_extensions_1")

file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

set(DEPENDENCIES openvino::runtime::dev offline_transformations)
set(DEFINITIONS)

if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEPENDENCIES openvino::frontend::onnx)
    list(APPEND DEFINITIONS ENABLE_OV_ONNX_FRONTEND)
endif()

if (ENABLE_OV_TF_FRONTEND)
    list(APPEND DEPENDENCIES openvino::frontend::tensorflow)
    list(APPEND DEFINITIONS ENABLE_OV_TF_FRONTEND)
endif()

if (ENABLE_OV_PADDLE_FRONTEND)
    list(APPEND DEPENDENCIES openvino::frontend::paddle)
    list(APPEND DEFINITIONS ENABLE_OV_PADDLE_FRONTEND)
endif()

# Create library
add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS})
target_compile_definitions(${TARGET_NAME} PRIVATE ${DEFINITIONS})
target_link_libraries(${TARGET_NAME} PRIVATE ${DEPENDENCIES})

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

if(WIN32)
    install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
else()
    install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION tests/lib COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests/lib COMPONENT tests EXCLUDE_FROM_ALL)
endif()
