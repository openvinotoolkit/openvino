# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_FE_NAME "mock_mo_ngraph_frontend")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

# Create shared library
add_library(${TARGET_FE_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS})

target_include_directories(${TARGET_FE_NAME} PRIVATE ".")

target_link_libraries(${TARGET_FE_NAME} PRIVATE frontend_manager)
target_link_libraries(${TARGET_FE_NAME} PUBLIC ngraph PRIVATE ngraph::builder)

add_clang_format_target(${TARGET_FE_NAME}_clang FOR_TARGETS ${TARGET_FE_NAME})
