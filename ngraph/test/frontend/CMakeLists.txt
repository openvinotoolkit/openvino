# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/mock_frontend.cpp)
add_library(mock1_frontend SHARED ${SRC})

target_compile_definitions(mock1_frontend PRIVATE "-DMOCK_VARIANT=\"1\"")

target_include_directories(mock1_frontend PRIVATE ".")

target_include_directories(mock1_frontend PRIVATE ${FRONTEND_INCLUDE_PATH} ${NGRAPH_INCLUDE_PATH})
target_link_libraries(mock1_frontend PRIVATE frontend_manager)
