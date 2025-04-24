// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/multiply_add.hpp"

#include <vector>

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> input_type = {ov::element::f32};

const std::vector<ov::Shape> inputShapes = {
    {1, 3, 2, 2, 4, 5},
    {1, 3, 2, 2, 2, 4, 5},
    {1, 3, 2, 2, 2, 2, 4, 5},
    {1, 3, 2, 2, 2, 2, 2, 4, 5},
    {1, 3, 2, 2, 2, 2, 2, 2, 4, 5},
};

INSTANTIATE_TEST_SUITE_P(smoke_MultipleAdd_Nd,
                         MultiplyAddLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(input_type),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MultiplyAddLayerTest::getTestCaseName);

}  // namespace
