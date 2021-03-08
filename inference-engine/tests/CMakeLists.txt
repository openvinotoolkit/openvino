# Copyright (C) 2019-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(IE_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_subdirectory(ngraph_helpers)
add_subdirectory(unit)

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(ie_test_utils)
    add_subdirectory(functional)
endif()