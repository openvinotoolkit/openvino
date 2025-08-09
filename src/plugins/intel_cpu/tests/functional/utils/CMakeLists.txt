# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(utils_SRC
    convolution_params.hpp
    cpu_test_utils.cpp
    cpu_test_utils.hpp
    filter_cpu_info.hpp
    fusing_test_utils.cpp
    fusing_test_utils.hpp
    properties_test.hpp
    quantization_utils.hpp
)

# Convert to full paths
foreach(src_file ${utils_SRC})
    list(APPEND utils_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${utils_SRC_FULL})

add_subdirectory(transformations)

# Use priority-based architecture selection to avoid multiple definitions
if(X86_64 OR X86)
    add_subdirectory(x64)
elseif(AARCH64 OR ARM)
    add_subdirectory(arm)
elseif(RISCV64)
    add_subdirectory(riscv64)
endif()
