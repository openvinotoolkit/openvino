# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ngraph_function_creation_sample")

file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

ie_add_sample(NAME ngraph_function_creation_sample
              SOURCES ${MAIN_SRC}
              HEADERS ${MAIN_HEADERS}
              DEPENDENCIES format_reader)

find_package(ngraph REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES})
