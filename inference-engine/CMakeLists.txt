# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(InferenceEngine)

if(ENABLE_PYTHON)
    add_subdirectory(ie_bridges/python)
endif()
