# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(PYBIND_FE_NAME "pybind_mock_frontend")

set(PYBIND_FE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pyngraph_mock_frontend_api.cpp)

source_group("src" FILES ${PYBIND_FE_SRC})

pybind11_add_module(${PYBIND_FE_NAME} MODULE NO_EXTRAS ${PYBIND_FE_SRC})

target_compile_definitions(${PYBIND_FE_NAME} PRIVATE
        IMPLEMENT_OPENVINO_API
        $<$<BOOL:${ENABLE_OV_PADDLE_FRONTEND}>:ENABLE_OV_PADDLE_FRONTEND>
        $<$<BOOL:${ENABLE_OV_TF_FRONTEND}>:ENABLE_OV_TF_FRONTEND>
)
target_link_libraries(${PYBIND_FE_NAME} PRIVATE openvino_mock_py_frontend)

add_clang_format_target(${PYBIND_FE_NAME}_clang FOR_TARGETS ${PYBIND_FE_NAME})

install(TARGETS ${PYBIND_FE_NAME}
        DESTINATION ${OV_CPACK_PYTHONDIR}/${pyversion}
        COMPONENT tests EXCLUDE_FROM_ALL)
