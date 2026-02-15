# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "test_builtin_extensions")

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4217")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /ignore:4217")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4217")
endif()

file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

set(DEPENDENCIES openvino::runtime::dev)

if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEPENDENCIES openvino::frontend::onnx)
    list(APPEND DEFINITIONS ENABLE_OV_ONNX_FRONTEND)
endif()

if (ENABLE_OV_TF_FRONTEND)
    list(APPEND DEPENDENCIES openvino::frontend::tensorflow)
    list(APPEND DEFINITIONS ENABLE_OV_TF_FRONTEND)
endif()

if (ENABLE_OV_TF_LITE_FRONTEND)
    list(APPEND DEPENDENCIES openvino::frontend::tensorflow_lite)
    list(APPEND DEFINITIONS ENABLE_OV_TF_LITE_FRONTEND)
endif()

if (ENABLE_OV_PADDLE_FRONTEND)
    list(APPEND DEPENDENCIES openvino::frontend::paddle)
    list(APPEND DEFINITIONS ENABLE_OV_PADDLE_FRONTEND)
endif()

if (ENABLE_OV_PYTORCH_FRONTEND)
    list(APPEND DEPENDENCIES openvino::frontend::pytorch)
    list(APPEND DEFINITIONS ENABLE_OV_PYTORCH_FRONTEND)
endif()

# Create library

add_library(${TARGET_NAME} MODULE ${LIBRARY_SRC} ${LIBRARY_HEADERS})

target_compile_definitions(${TARGET_NAME} PRIVATE ${DEFINITIONS})
target_link_libraries(${TARGET_NAME} PRIVATE ${DEPENDENCIES})

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
