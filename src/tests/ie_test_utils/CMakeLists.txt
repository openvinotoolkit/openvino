# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_TESTS)
    add_subdirectory(unit_test_utils)
endif()

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional_test_utils)
endif()
