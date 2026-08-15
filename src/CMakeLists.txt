# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_definitions(-DIN_OV_COMPONENT)

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wmissing-declarations)
endif()

# The standalone Vulkan core only. frontends / plugins / inference / bindings /
# tests depend on the removed ov::Model graph layer and are intentionally gone.
add_subdirectory(core)
