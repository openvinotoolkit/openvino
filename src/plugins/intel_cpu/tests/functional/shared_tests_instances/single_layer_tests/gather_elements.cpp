// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/gather_elements.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GatherElementsLayerTest;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
        ov::element::i64,
        ov::element::i16,
};
const std::vector<ov::element::Type> indices_types = {
        ov::element::i32,
        ov::element::i64
};

INSTANTIATE_TEST_SUITE_P(smoke_set1, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 2}})),
                            ::testing::Values(ov::Shape{2, 2}),
                            ::testing::ValuesIn(std::vector<int>{-1, 0, 1}),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(indices_types),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set2, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 2, 1}})),
                            ::testing::Values(ov::Shape{4, 2, 1}),
                            ::testing::ValuesIn(std::vector<int>({0, -3})),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(indices_types),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set3, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 2, 3, 5}})),
                            ::testing::Values(ov::Shape{2, 2, 3, 7}),
                            ::testing::Values(3, -1),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(indices_types),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set4, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{3, 2, 3, 8}})),
                            ::testing::Values(ov::Shape{2, 2, 3, 8}),
                            ::testing::Values(0, -4),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(indices_types),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set5, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{3, 2, 3, 4, 8}})),
                            ::testing::Values(ov::Shape{3, 2, 3, 5, 8}),
                            ::testing::Values(3, -2),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(indices_types),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);
}  // namespace
