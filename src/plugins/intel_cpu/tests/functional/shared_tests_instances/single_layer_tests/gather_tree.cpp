// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/gather_tree.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/test_enums.hpp"

namespace {
using ov::test::GatherTreeLayerTest;
using ov::test::utils::InputLayerType;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::i32
};

const std::vector<ov::Shape> input_shapes = {{5, 1, 10}, {1, 1, 10}, {20, 1, 10}, {20, 20, 10}};

const std::vector<InputLayerType> secondary_input_types = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER
};

INSTANTIATE_TEST_SUITE_P(smoke_GatherTree, GatherTreeLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(input_shapes),
                            ::testing::ValuesIn(secondary_input_types),
                            ::testing::ValuesIn(model_types),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GatherTreeLayerTest::getTestCaseName);

}  // namespace
