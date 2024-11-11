# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(logger)

if(ENABLE_NPU_PLUGIN_ENGINE)
    add_subdirectory(zero)
endif()
