# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_SUBGRAPHS_DUMPER_COMMON_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/cache/cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/cache/graph_cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/cache/op_cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/matchers/single_op/convolutions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/matchers/single_op/manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/matchers/single_op/single_op.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/matchers/subgraph/fused_names.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/matchers/subgraph/manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/matchers/subgraph/read_value_assign.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/matchers/subgraph/repeat_pattern.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils/cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils/memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils/model_comparator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils/node.cpp
)

set(OV_SUBGRAPHS_DUMPER_CLI_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/include/gflag_config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp
)
