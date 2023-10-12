// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_conv_concat.hpp"

#include <vector>

using namespace ov::test;
const std::vector<ov::element::Type> netPrecisions = {ov::element::f32};

std::vector<ov::Shape> inputShapes = {{1, 32, 1, 130}, {1, 64, 1, 170}, {1, 32, 1, 1026}};

INSTANTIATE_TEST_SUITE_P(smoke_SplitConvConcat,
                         SplitConvConcat,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         SplitConvConcat::getTestCaseName);
