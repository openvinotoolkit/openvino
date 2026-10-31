# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(logger)

add_subdirectory(zero)
add_subdirectory(vm)
if(ENABLE_NPU_PLUGIN_ENGINE)
    add_subdirectory(vcl)
endif()
