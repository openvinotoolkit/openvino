# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# RPATH is always enabled for unit tests
set(CMAKE_SKIP_RPATH OFF)
# because unit tests use plugins object files compiled with LTO
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

add_subdirectory(functional)
add_subdirectory(unit)