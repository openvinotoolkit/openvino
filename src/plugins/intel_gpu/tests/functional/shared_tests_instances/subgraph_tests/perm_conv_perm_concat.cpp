// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/perm_conv_perm_concat.hpp"

#include <vector>

namespace {
std::vector<ov::Shape> input_shapes{
    {1, 1, 7, 32},
    {1, 1, 8, 16},
};

std::vector<ov::Shape> kernel_shapes{
    {1, 3},
    {1, 5},
};

std::vector<size_t> output_channels{
    32,
    64,
};

std::vector<ov::element::Type> netPrecisions = {ov::element::f32};

ov::AnyMap additional_config = {};
}  // namespace

namespace ov {
namespace test {
INSTANTIATE_TEST_SUITE_P(smoke_basic,
                         PermConvPermConcat,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(kernel_shapes),
                                            ::testing::ValuesIn(output_channels),
                                            ::testing::Values(additional_config)),
                         PermConvPermConcat::getTestCaseName);
}  // namespace test
}  // namespace ov
