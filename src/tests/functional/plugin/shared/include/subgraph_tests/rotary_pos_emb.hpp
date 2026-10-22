// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/rotary_pos_emb.hpp"

namespace ov {
namespace test {

TEST_P(RoPETestFlux, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestQwenVL, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

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
    if (std::get<2>(GetParam()) && targetDevice == ov::test::utils::DEVICE_GPU) {
        CheckNumberOfNodesWithType(function, {"rms"}, 0);
        size_t sin_cos_count = 0;
        for (const auto& node : function->get_ops()) {
            const auto& info = node->get_rt_info();
            if (info.at("layerType").as<std::string>() == "RoPE") {
                auto producer = node->get_input_node_shared_ptr(0);
                while (producer->get_rt_info().at("layerType").as<std::string>() == "Reorder") {
                    producer = producer->get_input_node_shared_ptr(0);
                }
                EXPECT_EQ(producer->get_rt_info().at("layerType").as<std::string>(), "Input");
            }
            const auto names = info.at("originalLayersNames").as<std::string>();
            if (names.find("position_cos") != std::string::npos || names.find("position_sin") != std::string::npos) {
                EXPECT_EQ(node->get_output_element_type(0), ov::element::f32) << names;
                ++sin_cos_count;
            }
        }
        EXPECT_EQ(sin_cos_count, 2);
    }
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

TEST_P(RoPETestChatGLMHF, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestGPTOSS, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestLtxVideo, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

TEST_P(RoPETestCohere, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

}  // namespace test
}  // namespace ov
