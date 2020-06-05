#
# Copyright (C) 2005-2019 Intel Corporation
#
# SPDX-License-Identifier: GPL-2.0-only OR BSD-3-Clause
#

cmake_minimum_required(VERSION 2.8)

project(ittapi)

OPTION(FORCE_32 "Force a 32bit compile on 64bit" OFF)

IF(FORCE_32 AND UNIX)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
ENDIF()

if(CMAKE_SIZEOF_VOID_P MATCHES "8" AND NOT(FORCE_32))
    set(ARCH_64 1)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-D_DEBUG)
    if (NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    endif()
else()
    if (NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
    endif()
    add_definitions(-DNDEBUG)
endif()

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG}     ${LIBRARY_OUTPUT_PATH} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

set(ITT_PUBLIC_HDRS
    include/ittnotify.h
    include/jitprofiling.h
    include/libittnotify.h
)

file(GLOB ITT_SRCS "src/ittnotify/*.c" "src/ittnotify/*.h")

add_library(ittnotify STATIC ${ITT_SRCS} ${ITT_PUBLIC_HDRS})

if(APPLE)
    SET_TARGET_PROPERTIES(ittnotify PROPERTIES OUTPUT_NAME ittnotify)
endif()

if(WIN32)
    if(ARCH_64)
        SET_TARGET_PROPERTIES(ittnotify PROPERTIES OUTPUT_NAME libittnotify64)
    else()
        SET_TARGET_PROPERTIES(ittnotify PROPERTIES OUTPUT_NAME libittnotify32)
    endif()
else()
    if(ARCH_64)
        SET_TARGET_PROPERTIES(ittnotify PROPERTIES OUTPUT_NAME ittnotify64)
    else()
        SET_TARGET_PROPERTIES(ittnotify PROPERTIES OUTPUT_NAME ittnotify32)
    endif()
endif()

if (NOT WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    TARGET_LINK_LIBRARIES(ittnotify dl)
endif()

SET_TARGET_PROPERTIES(ittnotify PROPERTIES LINKER_LANGUAGE C)

target_include_directories(ittnotify
    PUBLIC include src/ittnotify
)

set(CMAKE_SUPPRESS_REGENERATION true)
