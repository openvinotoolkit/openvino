// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/reshape_permute_conv_permute_reshape_act.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConvReshapeAct;

std::vector<std::array<size_t, 4>> input_shapes {
    {1, 1, 166, 2},
    {1, 1, 144, 2},
    {1, 1, 288, 2},
    {1, 1, 144, 4},
};

std::vector<std::array<size_t, 2>> kernel_shapes {
    {1, 7},
    {1, 15},
};

std::vector<size_t> output_channels {
    16,
    8,
    4,
};

std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
};

ov::AnyMap additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_basic, ConvReshapeAct,
    ::testing::Combine(
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(kernel_shapes),
        ::testing::ValuesIn(output_channels),
        ::testing::Values(additional_config)),
    ConvReshapeAct::getTestCaseName);

}  // namespace
