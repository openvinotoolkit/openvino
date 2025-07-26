# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(execution_graph_tests_SRC
    add_output.cpp
    disable_lowering_precision.cpp
    duplicate_inputs_outputs_names.cpp
    keep_assign.cpp
    nms_transformation_for_last_node.cpp
    num_inputs_fusing_bin_conv.cpp
    remove_parameter.cpp
    runtime_precision.cpp
)

# Convert to full paths
foreach(src_file ${execution_graph_tests_SRC})
    list(APPEND execution_graph_tests_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${execution_graph_tests_SRC_FULL})
