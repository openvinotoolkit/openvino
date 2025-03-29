# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (MSVC)
    # NPU plugin tests don't have macros that have `defined`
    # in their expansion
    ov_add_compiler_flags(/wd5105)
endif()
add_subdirectory(functional)
add_subdirectory(unit)
