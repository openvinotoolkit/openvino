# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(shared_tests_instances_SRC
    core_config.cpp
    set_device_name.cpp
    skip_tests_config.cpp
)

# Convert to full paths
foreach(src_file ${shared_tests_instances_SRC})
    list(APPEND shared_tests_instances_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${shared_tests_instances_SRC_FULL})

add_subdirectory(behavior)
add_subdirectory(execution_graph_tests)
add_subdirectory(low_precision_transformations)
add_subdirectory(preprocessing)
add_subdirectory(single_layer_tests)
add_subdirectory(snippets)
add_subdirectory(subgraph_tests)

if(ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(onnx)
endif()
