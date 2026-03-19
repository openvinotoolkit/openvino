# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(logger)
add_subdirectory(vcl)

if(ENABLE_NPU_PLUGIN_ENGINE)
    add_subdirectory(zero)
endif()
