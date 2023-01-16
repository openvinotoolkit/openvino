# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(EXISTS "$ENV{MV_COMMON_BASE}")
    set(XLINK_ROOT_DIR "$ENV{MV_COMMON_BASE}/components/XLink")
else()
    set(XLINK_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
endif(EXISTS "$ENV{MV_COMMON_BASE}")

set(XLINK_INCLUDE
        ${XLINK_ROOT_DIR}/shared/include
        ${XLINK_ROOT_DIR}/pc/protocols)

set(XLINK_PRIVATE_INCLUDE)

file(GLOB PC_SRC             "${XLINK_ROOT_DIR}/pc/*.c")
file(GLOB PC_PROTO_SRC       "${XLINK_ROOT_DIR}/pc/protocols/*.c")
file(GLOB_RECURSE SHARED_SRC "${XLINK_ROOT_DIR}/shared/*.c")

list(APPEND XLINK_SOURCES ${PC_SRC} ${PC_PROTO_SRC} ${SHARED_SRC})

if(WIN32)
    set(XLINK_PLATFORM_INCLUDE ${XLINK_ROOT_DIR}/pc/Win/include)

    file(GLOB XLINK_PLATFORM_SRC "${XLINK_ROOT_DIR}/pc/Win/src/*.c")
    list(APPEND XLINK_SOURCES ${XLINK_PLATFORM_SRC})
else()
    find_package(Threads REQUIRED)

    # TODO: need to pre-detect libusb before enabling ENABLE_INTEL_MYRIAD_COMMON

    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND AND NOT ANDROID)
        pkg_search_module(libusb REQUIRED
                          IMPORTED_TARGET
                          libusb-1.0)
        if(libusb_FOUND)
            set(LIBUSB_LIBRARY "PkgConfig::libusb" CACHE STRING "libusb-1.0 imported target")
            set(LIBUSB_INCLUDE_DIR "" CACHE PATH "libusb-1.0 include dirs")

            message(STATUS "libusb-1.0 (${libusb_VERSION}) is found at ${libusb_PREFIX}")
        endif()
    else()
        find_path(LIBUSB_INCLUDE_DIR NAMES libusb.h PATH_SUFFIXES "include" "libusb" "libusb-1.0")
        find_library(LIBUSB_LIBRARY NAMES usb-1.0 PATH_SUFFIXES "lib")

        if(NOT LIBUSB_INCLUDE_DIR OR NOT LIBUSB_LIBRARY)
            message(FATAL_ERROR "libusb is required, please install it")
        endif()
    endif()

    set(XLINK_PLATFORM_INCLUDE ${XLINK_ROOT_DIR}/pc/MacOS)
    list(APPEND XLINK_SOURCES "${XLINK_ROOT_DIR}/pc/MacOS/pthread_semaphore.c")
endif()

# This is for the Movidius team
set(XLINK_INCLUDE_DIRECTORIES ${XLINK_INCLUDE})
