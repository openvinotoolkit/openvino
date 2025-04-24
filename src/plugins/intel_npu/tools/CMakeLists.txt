#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_subdirectory(common)
add_subdirectory(compile_tool)
add_subdirectory(single-image-test)

if (ENABLE_INTEL_NPU_PROTOPIPE)
    add_subdirectory(protopipe)
endif()
