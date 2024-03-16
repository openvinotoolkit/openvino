# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(OpenVINO_C_API)

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
