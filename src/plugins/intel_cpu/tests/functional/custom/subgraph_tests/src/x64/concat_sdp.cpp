// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

#include "custom/subgraph_tests/src/classes/concat_sdp.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {
const std::vector<std::vector<InputShape>> inputShapes = {
    // greedy search
    {
        // B, H, L1, S
        {{1, 8, -1, 64}, {{1, 8, 10, 64}, {1, 8, 1, 64}, {1, 8, 1, 64}, {1, 8, 20, 64}, {1, 8, 1, 64}}},
        // B, H, L0, S
        {{1, 8, -1, 64}, {{1, 8, 0, 64}, {1, 8, 10, 64}, {1, 8, 11, 64}, {1, 8, 12, 64}, {1, 8, 32, 64}}},
    },
    // beam search
    {
        // B, H, L1, S
        {{-1, 8, -1, 64}, {{4, 8, 10, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}}},
        // B, H, L0, S
        {{-1, 8, -1, 64}, {{4, 8, 0, 64}, {4, 8, 10, 64}, {4, 8, 11, 64}, {4, 8, 12, 64}, {4, 8, 13, 64}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTest,
        ConcatSDPTest,
        ::testing::Combine(::testing::Values(ElementType::bf16, ElementType::f16),
                           ::testing::ValuesIn(inputShapes),
                           ::testing::Values(true, false),
                           ::testing::Values(true, false),
                           ::testing::Values(true, false)),
        ConcatSDPTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
