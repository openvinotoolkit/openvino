# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_definitions(-DIN_OV_COMPONENT)
ov_deprecated_no_errors()

file(GLOB OPS_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp"
    )

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
    ov_add_compiler_flags(/wd4244)
    ov_add_compiler_flags(/wd4018)
endif()

add_library(interpreter_backend STATIC EXCLUDE_FROM_ALL ${OPS_SRC} ${SRC})
add_library(openvino::interpreter_backend ALIAS interpreter_backend)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-missing-declarations)
endif()

ov_build_target_faster(interpreter_backend UNITY)

target_compile_definitions(interpreter_backend
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
target_link_libraries(interpreter_backend PRIVATE openvino::builders openvino::reference openvino::util openvino::runtime::dev openvino::shape_inference)

target_include_directories(interpreter_backend PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ops/>)

file(GLOB_RECURSE all_backends_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
ov_add_clang_format_target(interpreter_backend_clang FOR_SOURCES ${all_backends_src})

# developer package

openvino_developer_export_targets(COMPONENT core TARGETS interpreter_backend)

# install

ov_install_static_lib(interpreter_backend ${OV_CPACK_COMP_CORE})
