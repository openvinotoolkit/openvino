# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Close LTO temporarily for add_symbol_to_partition_1 issue
set(ENABLE_LTO OFF)

add_subdirectory(utils)

add_subdirectory(al)

if (ENABLE_NPU_PLUGIN_ENGINE)
    add_subdirectory(common)
    add_subdirectory(compiler_adapter)
    add_subdirectory(backend)
    add_subdirectory(plugin)
endif()
