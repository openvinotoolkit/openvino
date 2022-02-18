# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_FE_NAME "openvino_mock_py_frontend")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

add_library(${TARGET_FE_NAME} ${LIBRARY_SRC} ${LIBRARY_HEADERS})

target_include_directories(${TARGET_FE_NAME}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(DEPENDENCIES frontend_common::static)
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

target_compile_definitions(${TARGET_FE_NAME} PRIVATE ${DEFINITIONS})
target_link_libraries(${TARGET_FE_NAME} PRIVATE ${DEPENDENCIES})

add_clang_format_target(${TARGET_FE_NAME}_clang FOR_TARGETS ${TARGET_FE_NAME})

install(TARGETS ${TARGET_FE_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
