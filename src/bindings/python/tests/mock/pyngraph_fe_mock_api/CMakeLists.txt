# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(PYBIND_FE_NAME "pybind_mock_frontend")

set(PYBIND_FE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pyngraph_mock_frontend_api.cpp)

source_group("src" FILES ${PYBIND_FE_SRC})

pybind11_add_module(${PYBIND_FE_NAME} MODULE NO_EXTRAS ${PYBIND_FE_SRC})

target_link_libraries(${PYBIND_FE_NAME} PRIVATE openvino::frontend::mock_py)

ov_add_clang_format_target(${PYBIND_FE_NAME}_clang FOR_TARGETS ${PYBIND_FE_NAME})

install(TARGETS ${PYBIND_FE_NAME} DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
