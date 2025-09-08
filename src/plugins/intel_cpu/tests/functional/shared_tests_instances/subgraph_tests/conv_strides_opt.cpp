// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/conv_strides_opt.hpp"

#include <vector>

using namespace ov::test;

namespace {

std::vector<ov::Shape> input_shapes{
    ov::Shape{1, 1, 4, 4},
    ov::Shape{1, 64, 56, 56},
};

std::vector<ov::op::PadType> pads{
    ov::op::PadType::SAME_UPPER,
    ov::op::PadType::SAME_LOWER,
    ov::op::PadType::EXPLICIT,
};

INSTANTIATE_TEST_SUITE_P(smoke_Convolution_StridesOpt,
                         ConvStridesOpt,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(pads),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvStridesOpt::getTestCaseName);

}  // namespace
