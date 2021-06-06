# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/mock_frontend.cpp)
add_library(mock1_ngraph_frontend SHARED ${SRC})

target_compile_definitions(mock1_ngraph_frontend PRIVATE "-DMOCK_VARIANT=\"1\"")

target_include_directories(mock1_ngraph_frontend PRIVATE ".")

target_include_directories(mock1_ngraph_frontend PRIVATE ${FRONTEND_INCLUDE_PATH} ${NGRAPH_INCLUDE_PATH})
target_link_libraries(mock1_ngraph_frontend PRIVATE frontend_manager)
add_dependencies(unit-test mock1_ngraph_frontend)

add_clang_format_target(mock1_ngraph_frontend_clang FOR_TARGETS mock1_ngraph_frontend)
