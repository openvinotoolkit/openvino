# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(add_ie_target)

#####################################################################################################
#                                         SETUP TESTS                                               #
#####################################################################################################
set(IE_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_subdirectory(ngraph_functions)
add_subdirectory(unit)

add_subdirectory(ie_test_utils)

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()