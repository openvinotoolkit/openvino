#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_subdirectory(utils)

add_subdirectory(al)

if(ENABLE_DRIVER_COMPILER_ADAPTER)
    add_subdirectory(compiler)
endif()

if(ENABLE_ZEROAPI_BACKEND)
    add_subdirectory(backend)
endif()

if (NOT BUILD_COMPILER_FOR_DRIVER)
    add_subdirectory(plugin)
endif()