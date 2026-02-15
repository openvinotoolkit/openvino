# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
