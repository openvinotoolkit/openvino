# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(InferenceEngine)

add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(ie_bridges/c)

if(ENABLE_PYTHON)
    add_subdirectory(ie_bridges/python)
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests_deprecated)
    add_subdirectory(tests)
endif()
