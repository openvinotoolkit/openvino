# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(preprocessing_SRC
    resize_tests.cpp
    yuv_to_grey_tests.cpp
)

# Convert to full paths
foreach(src_file ${preprocessing_SRC})
    list(APPEND preprocessing_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${preprocessing_SRC_FULL})
