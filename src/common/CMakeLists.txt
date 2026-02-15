# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(itt)
add_subdirectory(conditional_compilation)
add_subdirectory(util)

if(ENABLE_INTEL_CPU)
    add_subdirectory(snippets)
endif()

add_subdirectory(transformations)
add_subdirectory(offline_transformations)
add_subdirectory(low_precision_transformations)
