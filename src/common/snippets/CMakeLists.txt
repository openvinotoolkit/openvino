# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine_snippets")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/snippets/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

ie_faster_build(${TARGET_NAME}
    UNITY
)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime
                                     PRIVATE ngraph_reference openvino::runtime::dev)

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
                                          PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

ie_mark_target_as_cc(${TARGET_NAME})

# LTO

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install

ov_install_static_lib(${TARGET_NAME} core)
