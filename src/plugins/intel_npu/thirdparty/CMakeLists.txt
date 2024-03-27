# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(BUILD_SHARED_LIBS OFF)

#
# zeroApi
#

if(ENABLE_ZEROAPI_BACKEND)
    # We have to avoid linking against loader with debug postfix due it's a part of driver
    # and the name will be the same for release and debug configurations
    set(OLD_CMAKE_DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    set(CMAKE_DEBUG_POSTFIX "")
    set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        string(REPLACE "/WX" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=undef -Wno-error=missing-declarations -Wno-error=stringop-truncation -Wno-error=unused-but-set-variable")
    endif()
    add_subdirectory(level-zero EXCLUDE_FROM_ALL)
    set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
    set(CMAKE_DEBUG_POSTFIX ${OLD_CMAKE_DEBUG_POSTFIX})
endif()
