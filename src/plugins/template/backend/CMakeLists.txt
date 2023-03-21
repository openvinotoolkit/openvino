# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (SRC
    backend.cpp
    backend.hpp
    executable.cpp
    executable.hpp
    int_backend.cpp 
    int_executable.cpp 
    evaluates_map.cpp
    )

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4018)
endif()


add_library(interpreter_backend STATIC EXCLUDE_FROM_ALL ${SRC})
add_library(openvino::interpreter_backend ALIAS interpreter_backend)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wno-missing-declarations)
endif()

ie_faster_build(interpreter_backend UNITY)

target_compile_definitions(interpreter_backend
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
target_link_libraries(interpreter_backend PRIVATE ngraph::builder ngraph::reference openvino::util openvino::runtime::dev)

target_include_directories(interpreter_backend PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

file(GLOB_RECURSE all_backends_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(interpreter_backend_clang FOR_SOURCES ${all_backends_src})

# developer package

openvino_developer_export_targets(COMPONENT core TARGETS interpreter_backend)

# install

ov_install_static_lib(interpreter_backend template)
