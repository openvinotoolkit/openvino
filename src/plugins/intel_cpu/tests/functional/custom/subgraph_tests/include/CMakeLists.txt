# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(include_SRC
    conv_with_zero_point_fuse.hpp
    fuse_muladd_ewsimple.hpp
    undefined_et.hpp
)

# Convert to full paths
foreach(src_file ${include_SRC})
    list(APPEND include_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${include_SRC_FULL})
