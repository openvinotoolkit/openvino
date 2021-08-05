# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_shared_test_classes")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${LIBRARY_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(${TARGET_NAME} PUBLIC frontend_manager
        ngraph::ngraph cnpy ie_backend ngraph_test_util commonTestUtils)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
