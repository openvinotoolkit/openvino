# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# liburing (io_uring) is a Linux-only library. It ships an autotools-style
# `configure` script that generates a few host-specific headers
# (config-host.h, compat.h, io_uring_version.h). Both that configure step and
# the actual compilation are driven from CMake here, so liburing is built as a
# static library as part of the OpenVINO build.

project(liburing C)

set(TARGET_NAME "openvino_liburing")

set(LIBURING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/liburing")

#
# Run liburing's own configure script to generate config-host.h, compat.h and
# io_uring_version.h. '--use-libc' avoids the nolibc path so the sources build
# with the standard toolchain. The generated files are gitignored by liburing,
# so the submodule working tree stays clean.
#
if(NOT EXISTS "${LIBURING_SOURCE_DIR}/src/include/liburing/compat.h")
    message(STATUS "Configuring liburing")
    execute_process(
        COMMAND sh configure --use-libc --cc=${CMAKE_C_COMPILER}
        WORKING_DIRECTORY "${LIBURING_SOURCE_DIR}"
        RESULT_VARIABLE _liburing_configure_result
        OUTPUT_VARIABLE _liburing_configure_output
        ERROR_VARIABLE _liburing_configure_output)
    if(NOT _liburing_configure_result EQUAL 0)
        message(FATAL_ERROR "liburing configure failed:\n${_liburing_configure_output}")
    endif()
endif()

#
# liburing static library
#
set(liburing_srcs
    "${LIBURING_SOURCE_DIR}/src/setup.c"
    "${LIBURING_SOURCE_DIR}/src/queue.c"
    "${LIBURING_SOURCE_DIR}/src/register.c"
    "${LIBURING_SOURCE_DIR}/src/syscall.c"
    "${LIBURING_SOURCE_DIR}/src/version.c")

add_library(${TARGET_NAME} STATIC ${liburing_srcs})
add_library(openvino::liburing ALIAS ${TARGET_NAME})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_C_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-all)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE
    _GNU_SOURCE
    _LARGEFILE_SOURCE
    _FILE_OFFSET_BITS=64
    LIBURING_INTERNAL)

# Force-include the generated config-host.h, like liburing's own build does.
target_compile_options(${TARGET_NAME} PRIVATE "-include" "${LIBURING_SOURCE_DIR}/config-host.h")

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${LIBURING_SOURCE_DIR}/src/include>)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER thirdparty)
