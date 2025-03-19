// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/opsets/opset6.hpp"
#include "single_op_tests/gather_elements.hpp"

namespace {
using ov::test::GatherElementsLayerTest;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::element::Type> indices_types = {
        ov::element::i32,
        ov::element::i64,
};

INSTANTIATE_TEST_SUITE_P(smoke_set1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 2}})),
        ::testing::Values(ov::Shape{2, 2}),
        ::testing::ValuesIn(std::vector<int>{-1, 0, 1}),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 2, 1}})),
        ::testing::Values(ov::Shape{4, 2, 1}),
        ::testing::ValuesIn(std::vector<int>({0, -3})),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 2, 3, 5}})),
        ::testing::Values(ov::Shape{2, 2, 3, 7}),
        ::testing::Values(3, -1),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set4, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{3, 2, 3, 8}})),
        ::testing::Values(ov::Shape{2, 2, 3, 8}),
        ::testing::Values(0, -4),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set5, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{3, 2, 3, 4, 8}})),
        ::testing::Values(ov::Shape{3, 2, 3, 5, 8}),
        ::testing::Values(3, -2),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank4axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{7, 7, 8, 4}})),
        ::testing::Values(ov::Shape{2, 7, 8, 4}),
        ::testing::Values(0),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank4axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{6, 1, 8, 4}})),
        ::testing::Values(ov::Shape{6, 8, 8, 4}),
        ::testing::Values(1, -3),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank4axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{6, 7, 4, 4}})),
        ::testing::Values(ov::Shape{6, 7, 2, 4}),
        ::testing::Values(2, -2),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank4axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{6, 5, 8, 7}})),
        ::testing::Values(ov::Shape{6, 5, 8, 7}),
        ::testing::Values(1, -3),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 3, 9, 4, 9}})),
        ::testing::Values(ov::Shape{1, 3, 9, 4, 9}),
        ::testing::Values(0),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 3, 5, 4, 7}})),
        ::testing::Values(ov::Shape{2, 9, 5, 4, 7}),
        ::testing::Values(1, -4),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{1, 2, 6, 8, 9}})),
        ::testing::Values(ov::Shape{1, 2, 6, 8, 9}),
        ::testing::Values(2, -3),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 2, 4, 7, 7}})),
        ::testing::Values(ov::Shape{2, 2, 4, 3, 7}),
        ::testing::Values(3, -2),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis4, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{1, 3, 9, 3, 2}})),
        ::testing::Values(ov::Shape{1, 3, 9, 3, 9}),
        ::testing::Values(4, -1),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{3, 3, 2, 4, 4, 3}})),
        ::testing::Values(ov::Shape{7, 3, 2, 4, 4, 3}),
        ::testing::Values(0),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{1, 6, 2, 3, 5, 9}})),
        ::testing::Values(ov::Shape{1, 6, 2, 3, 5, 9}),
        ::testing::Values(1, -5),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 3, 9, 7, 2, 1}})),
        ::testing::Values(ov::Shape{2, 3, 5, 7, 2, 1}),
        ::testing::Values(2, -4),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{1, 3, 4, 5, 1, 3}})),
        ::testing::Values(ov::Shape{1, 3, 4, 4, 1, 3}),
        ::testing::Values(3, -3),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis4, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{1, 3, 2, 4, 3, 3}})),
        ::testing::Values(ov::Shape{1, 3, 2, 4, 6, 3}),
        ::testing::Values(4, -2),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis5, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{2, 1, 7, 8, 1, 6}})),
        ::testing::Values(ov::Shape{2, 1, 7, 8, 1, 5}),
        ::testing::Values(5, -1),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

}  // namespace
