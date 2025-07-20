# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(src_SRC
    disable_gathercompressed_quantized_model.cpp
    quantized_matmuls_with_shared_weights.cpp
    subgraph_select_pd.cpp
)

# Convert to full paths
foreach(src_file ${src_SRC})
    list(APPEND src_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${src_SRC_FULL})

# Use priority-based architecture selection to avoid multiple definitions
if(X86_64 OR X86)
    add_subdirectory(x64)
elseif(AARCH64 OR ARM)
    add_subdirectory(arm)
endif()

add_subdirectory(classes)
add_subdirectory(common)
