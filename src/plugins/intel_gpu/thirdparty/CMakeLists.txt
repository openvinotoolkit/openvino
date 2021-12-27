# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_library(rapidjson INTERFACE)

set_target_properties(rapidjson PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rapidjson>
)

ov_install_static_lib(rapidjson gpu)
