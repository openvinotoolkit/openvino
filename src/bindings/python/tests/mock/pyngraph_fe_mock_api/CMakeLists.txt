# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(PYBIND_FE_NAME "pybind_mock_frontend")

set(PYBIND_FE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pyngraph_mock_frontend_api.cpp)

source_group("src" FILES ${PYBIND_FE_SRC})

pybind11_add_module(${PYBIND_FE_NAME} MODULE ${PYBIND_FE_SRC})

set(DEPENDENCIES ov_mock_py_frontend openvino::runtime)

if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEPENDENCIES ov_onnx_frontend)
    add_compile_definitions(ENABLE_OV_ONNX_FRONTEND)
endif()

if (ENABLE_OV_TF_FRONTEND)
    list(APPEND DEPENDENCIES ov_tensorflow_frontend)
    add_compile_definitions(ENABLE_OV_TF_FRONTEND)
endif()

if (ENABLE_OV_PADDLE_FRONTEND)
    list(APPEND DEPENDENCIES ov_paddle_frontend)
    add_compile_definitions(ENABLE_OV_PADDLE_FRONTEND)
endif()

target_link_libraries(${PYBIND_FE_NAME} PRIVATE ${DEPENDENCIES})

add_clang_format_target(${PYBIND_FE_NAME}_clang FOR_TARGETS ${PYBIND_FE_NAME})

install(TARGETS ${PYBIND_FE_NAME}
        DESTINATION python/${PYTHON_VERSION}
        COMPONENT tests EXCLUDE_FROM_ALL)
