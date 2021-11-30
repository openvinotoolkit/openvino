# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(IE_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_subdirectory(ngraph_helpers)
add_subdirectory(unit)
add_subdirectory(ie_test_utils)

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()