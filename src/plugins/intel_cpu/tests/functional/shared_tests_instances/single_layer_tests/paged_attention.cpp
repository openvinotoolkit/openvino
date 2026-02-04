// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/test_constants.hpp"

#include "single_op_tests/paged_attention.hpp"
#include "shared_test_classes/single_op/paged_attention.hpp"

namespace {
using ov::test::PagedAttentionLayerTest;
using ElementType = ov::element::Type_t;
using InputShapes = std::vector<ov::test::InputShape>;

const std::vector<InputShapes> input_shapes_ref = {  // greedy search
{
    // L1, B, H, S
    {{-1, 1, 8, 64}, {{10, 1, 8, 64}, {1, 1, 8, 64}}},
    // B, L0, H, S
    {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {10, 1, 8, 64}}},
}};

const std::vector<ov::AnyMap> additional_configs_ref = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_ref),
                                            ::testing::Values(true, false),
                                            ::testing::Values(false),  // Xattn = false
                                            ::testing::Values(false),  // sinkInput = false
                                            ::testing::Values(0),      // sliding_window = 0
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);
}  // namespace