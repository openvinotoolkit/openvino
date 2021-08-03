# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ngraph_reference")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(REF_IMPL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

if(COMMAND ie_faster_build)
    ie_faster_build(${TARGET_NAME}
        UNITY
        PCH PRIVATE "src/precomp.hpp"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${TARGET_NAME} PUBLIC /wd4146)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE XBYAK_NO_OP_NAMES XBYAK64)

# Defines macro in C++ to load backend plugin
target_include_directories(${TARGET_NAME} PUBLIC ${REF_IMPL_INCLUDE_DIR} ${NGRAPH_INCLUDE_PATH})

link_system_libraries(${TARGET_NAME} PRIVATE xbyak)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(ngraph::reference ALIAS ${TARGET_NAME})

# developer package
openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph::reference)
