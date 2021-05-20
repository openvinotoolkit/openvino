# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_FE_NAME "mock_py_ngraph_frontend")

set(SRC mock_py_frontend.cpp)
add_library(${TARGET_FE_NAME} SHARED ${SRC})

target_compile_definitions(${TARGET_FE_NAME} PRIVATE)

target_include_directories(${TARGET_FE_NAME} PRIVATE ".")

target_include_directories(${TARGET_FE_NAME} PRIVATE ${FRONTEND_INCLUDE_PATH} ${NGRAPH_INCLUDE_PATH})
target_link_libraries(${TARGET_FE_NAME} PRIVATE frontend_manager)
target_link_libraries(${TARGET_FE_NAME} PUBLIC ngraph  PRIVATE ngraph::builder)

add_clang_format_target(${TARGET_FE_NAME}_clang FOR_TARGETS ${TARGET_FE_NAME})
