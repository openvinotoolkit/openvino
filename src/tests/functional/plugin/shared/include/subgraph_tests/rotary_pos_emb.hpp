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
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestLlama2StridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestChatGLMStridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestQwen7bStridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestGPTJStridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestRotateHalfWithoutTranspose, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
}

TEST_P(RoPETestLlama2Slice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestChatGLMSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestQwen7bSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestGPTJSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestChatGLM2DRoPEStridedSlice, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestChatGLMHF, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestGPTOSS, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

TEST_P(RoPETestLtxVideo, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(function, {"RoPE"}, 1);
};

}  // namespace test
}  // namespace ov
