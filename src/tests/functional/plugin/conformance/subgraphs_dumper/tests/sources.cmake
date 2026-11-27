# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_SUBGRAPHS_DUMPER_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/base_test.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cache/cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cache/graph_cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cache/meta.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cache/op_cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/single_op/convolutions_matcher.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/single_op/generic_single_op.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/single_op/manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/single_op/matchers_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/subgraph/fused_names.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/subgraph/manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/subgraph/read_value.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/subgraph/repeat_pattern.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matchers/subgraph/subgraph.cpp
    ${CMAKE_CURRENT_LIST_DIR}/skip_tests_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_models/model_0.hpp
    ${CMAKE_CURRENT_LIST_DIR}/test_models/model_1.hpp
    ${CMAKE_CURRENT_LIST_DIR}/test_models/model_2.hpp
    ${CMAKE_CURRENT_LIST_DIR}/test_models/model_3.hpp
    ${CMAKE_CURRENT_LIST_DIR}/test_models/model_4.hpp
    ${CMAKE_CURRENT_LIST_DIR}/test_models/model_5.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/model_comparator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/node.cpp
)
