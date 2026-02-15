# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_JS)
    return()
endif()

project(OpenVINO_JS_API)

add_subdirectory(node)
