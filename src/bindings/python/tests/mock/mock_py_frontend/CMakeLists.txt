# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_FE_NAME "openvino_mock_py_frontend")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

add_library(${TARGET_FE_NAME} ${LIBRARY_SRC} ${LIBRARY_HEADERS})

target_include_directories(${TARGET_FE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIBRARIES openvino::runtime)
set(DEFINITIONS IMPLEMENT_OPENVINO_API)

if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND LIBRARIES openvino::frontend::onnx)
    list(APPEND DEFINITIONS ENABLE_OV_ONNX_FRONTEND)
endif()

if (ENABLE_OV_TF_FRONTEND)
    list(APPEND LIBRARIES openvino::frontend::tensorflow)
    list(APPEND DEFINITIONS ENABLE_OV_TF_FRONTEND)
endif()

if (ENABLE_OV_PADDLE_FRONTEND)
    list(APPEND LIBRARIES openvino::frontend::paddle)
    list(APPEND DEFINITIONS ENABLE_OV_PADDLE_FRONTEND)
endif()

target_compile_definitions(${TARGET_FE_NAME} PRIVATE ${DEFINITIONS})
target_link_libraries(${TARGET_FE_NAME} PUBLIC ${LIBRARIES})

add_clang_format_target(${TARGET_FE_NAME}_clang FOR_TARGETS ${TARGET_FE_NAME})

install(TARGETS ${TARGET_FE_NAME}
        RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
