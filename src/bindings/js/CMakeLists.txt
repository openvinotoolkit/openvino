# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Disable JS API by default
if(NOT ENABLE_JS)
    message(WARNING "JS API is disabled by default, specify -DENABLE_JS=ON to switch it on")
    return()
endif()

project(OpenVINO_JS_API)

add_subdirectory(node)
