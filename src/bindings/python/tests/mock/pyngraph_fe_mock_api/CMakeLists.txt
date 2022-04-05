# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(PYBIND_FE_NAME "pybind_mock_frontend")

set(PYBIND_FE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pyngraph_mock_frontend_api.cpp)

source_group("src" FILES ${PYBIND_FE_SRC})

pybind11_add_module(${PYBIND_FE_NAME} MODULE ${PYBIND_FE_SRC})

set(DEPENDENCIES openvino_mock_py_frontend openvino::runtime)
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

target_compile_definitions(${PYBIND_FE_NAME} PRIVATE ${DEFINITIONS})
target_link_libraries(${PYBIND_FE_NAME} PRIVATE ${DEPENDENCIES})

add_clang_format_target(${PYBIND_FE_NAME}_clang FOR_TARGETS ${PYBIND_FE_NAME})

install(TARGETS ${PYBIND_FE_NAME}
        DESTINATION python/${PYTHON_VERSION}
        COMPONENT tests EXCLUDE_FROM_ALL)
