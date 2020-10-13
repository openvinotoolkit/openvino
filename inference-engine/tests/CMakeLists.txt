# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    ie_enable_lto()
endif()

set(IE_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_subdirectory(ngraph_functions)
add_subdirectory(unit)

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(ie_test_utils)
endif()

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()