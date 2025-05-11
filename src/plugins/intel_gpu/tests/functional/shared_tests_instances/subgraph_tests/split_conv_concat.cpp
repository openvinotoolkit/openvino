// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_conv_concat.hpp"

#include <vector>

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> input_types = {ov::element::f32, ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape,
                         SplitConvConcat,
                         ::testing::Combine(::testing::ValuesIn(input_types),
                                            ::testing::Values(ov::Shape{1, 6, 40, 40}),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         SplitConvConcat::getTestCaseName);

}  // namespace

