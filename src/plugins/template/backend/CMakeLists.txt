# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_interpreter_backend)

add_definitions(-DIN_OV_COMPONENT)

file(GLOB OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp")

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

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${OPS_SRC} ${SRC})

add_library(openvino::interpreter_backend ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME interpreter_backend)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-missing-declarations)
endif()

ov_build_target_faster(${TARGET_NAME} UNITY)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::reference openvino::util openvino::runtime::dev openvino::shape_inference)

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ops/>
                                                 $<INSTALL_INTERFACE:developer_package/include/${TARGET_NAME}>)

file(GLOB_RECURSE all_backends_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_SOURCES ${all_backends_src})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET openvino::interpreter_backend)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION developer_package/include/${TARGET_NAME}
        COMPONENT developer_package EXCLUDE_FROM_ALL
        FILES_MATCHING PATTERN "*.hpp")
