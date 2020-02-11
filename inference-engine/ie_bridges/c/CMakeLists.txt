# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(InferenceEngine_C_API)

add_subdirectory(src)

if(ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()
