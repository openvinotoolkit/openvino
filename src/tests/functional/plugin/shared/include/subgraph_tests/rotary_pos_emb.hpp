// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/rotary_pos_emb.hpp"

namespace ov {
namespace test {

TEST_P(RoPETestLlama2, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckNumberOfNodesWithType(compiledModel, {"RoPE"}, 1);
};

TEST_P(RoPETestChatGLM, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckNumberOfNodesWithType(compiledModel, {"RoPE"}, 1);
};

TEST_P(RoPETestQwen7b, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckNumberOfNodesWithType(compiledModel, {"RoPE"}, 1);
};

TEST_P(RoPETestGPTJ, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckNumberOfNodesWithType(compiledModel, {"RoPE"}, 1);
};

TEST_P(RoPETestRotateHalfWithoutTranspose, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckNumberOfNodesWithType(compiledModel, {"RoPE"}, 1);
};

}  // namespace test
}  // namespace ov
