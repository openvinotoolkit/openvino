# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(invalid_cases_SRC
    proposal.cpp
)

# Convert to full paths
foreach(src_file ${invalid_cases_SRC})
    list(APPEND invalid_cases_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${invalid_cases_SRC_FULL})
