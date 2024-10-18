# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Close LTO temporarily for add_symbol_to_partition_1 issue
set(ENABLE_LTO OFF)

add_subdirectory(utils)

add_subdirectory(al)

add_subdirectory(compiler_adapter)

add_subdirectory(backend)

if (NOT BUILD_COMPILER_FOR_DRIVER)
    add_subdirectory(plugin)
endif()
