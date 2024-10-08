// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/rotary_pos_emb.hpp"

namespace ov {
namespace test {

inline void CheckNumberOfNodesWithType(std::shared_ptr<const ov::Model> function,
                                       const std::unordered_set<std::string>& nodeTypes,
                                       size_t expectedCount) {
    ASSERT_NE(nullptr, function);
    int num_ops = 0;
    for (const auto& node : function->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type = rt_info.find("layerType")->second.as<std::string>();
        if (nodeTypes.count(layer_type)) {
            num_ops++;
        }
    }
    ASSERT_EQ(num_ops, expectedCount);
}

TEST_P(RoPETestLlama2StridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestChatGLMStridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestQwen7bStridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestGPTJStridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestRotateHalfWithoutTranspose, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
}

TEST_P(RoPETestLlama2Slice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestChatGLMSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestQwen7bSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestGPTJSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestChatGLM2DRoPEStridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

}  // namespace test
}  // namespace ov
