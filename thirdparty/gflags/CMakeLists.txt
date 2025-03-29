# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(GFLAGS_IS_SUBPROJECT TRUE)
set(HAVE_SYS_STAT_H 1)
set(HAVE_INTTYPES_H 1)
set(INTTYPES_FORMAT C99)
set(BUILD_TESTING OFF)
set(BUILD_SHARED_LIBS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
endif()

add_subdirectory(gflags EXCLUDE_FROM_ALL)
set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
