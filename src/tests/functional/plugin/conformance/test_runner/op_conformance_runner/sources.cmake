# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_OP_CONFORMANCE_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/core_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/op_impl_check/op_impl_check.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/op_impl_check/single_op_graph.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/read_ir/read_ir.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils/generate_static_shapes.cpp
)
