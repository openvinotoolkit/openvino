// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/identity.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::IdentityLayerTest;

namespace {
const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i64,
        ov::element::i32,
        ov::element::i16,
        ov::element::i8,
        ov::element::u8,
};

std::vector<std::vector<ov::Shape>> input_shape_static_2D = {{{2, 10}}, {{10, 2}}, {{10, 10}}};

INSTANTIATE_TEST_SUITE_P(smoke_Identity2D, IdentityLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_2D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                IdentityLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> input_shape_static_3D = {{{2, 8, 10}}, {{10, 8, 2}}, {{8, 2, 10}}};

INSTANTIATE_TEST_SUITE_P(smoke_Identity3D, IdentityLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_3D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                IdentityLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> input_shape_static_4D = {{{2, 2, 2, 2}}, {{1, 10, 2, 3}}, {{2, 3, 4, 5}}};

INSTANTIATE_TEST_SUITE_P(smoke_Identity4D, IdentityLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_4D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                IdentityLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> input_shape_static_5D = {{{2, 2, 2, 2, 2}}, {{1, 10, 2, 3, 4}}, {{2, 3, 4, 5, 6}}};

INSTANTIATE_TEST_SUITE_P(smoke_Identity5D, IdentityLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_5D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                IdentityLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> input_shape_static_6D = {{{2, 2, 2, 2, 2, 2}}, {{1, 10, 2, 3, 4, 5}}, {{2, 3, 4, 5, 6, 7}}};

INSTANTIATE_TEST_SUITE_P(smoke_Identity6D, IdentityLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_6D)),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        IdentityLayerTest::getTestCaseName);

}  // namespace
