# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(transformations_SRC
    insert_fake_quantize.cpp
    insert_fake_quantize.hpp
    insert_requantize.cpp
    insert_requantize.hpp
)

# Convert to full paths
foreach(src_file ${transformations_SRC})
    list(APPEND transformations_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${transformations_SRC_FULL})
