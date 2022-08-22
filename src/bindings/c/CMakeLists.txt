# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(OpenVINO_C_API)

# TODO: fix it
if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wno-error=sign-compare)
    ie_add_compiler_flags(-Wno-error=missing-declarations)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
