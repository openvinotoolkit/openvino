# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_TESTS)
    add_subdirectory(common_test_utils)
    add_subdirectory(unit_test_utils)
endif()

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional_test_utils)
endif()
