# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_FE_NAME "mock_py_ngraph_frontend")
set(PYBIND_FE_NAME "pybind_mock_frontend")

set(PYBIND_FE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pyngraph_mock_frontend_api.cpp)

source_group("src" FILES ${PYBIND_FE_SRC})

pybind11_add_module(${PYBIND_FE_NAME} MODULE ${PYBIND_FE_SRC})

target_link_libraries(${PYBIND_FE_NAME} PRIVATE ${TARGET_FE_NAME})

add_clang_format_target(${PYBIND_FE_NAME}_clang FOR_TARGETS ${PYBIND_FE_NAME})
