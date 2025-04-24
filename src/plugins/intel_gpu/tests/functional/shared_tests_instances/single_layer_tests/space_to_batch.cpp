// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/space_to_batch.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::SpaceToBatchLayerTest;
using ov::test::spaceToBatchParamsTuple;

auto stb_only_test_cases = []() {
    return std::vector<spaceToBatchParamsTuple>{
        spaceToBatchParamsTuple({1, 2, 2},
                                {0, 0, 0},
                                {0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 2, 2}})),
                                ov::element::f32,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 2, 2}})),
                                ov::element::f32,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 3, 2, 2}})),
                                ov::element::f32,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 4, 4}})),
                                ov::element::f32,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 2},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{2, 1, 2, 4}})),
                                ov::element::f32,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 3, 2, 2},
                                {0, 0, 1, 0, 3},
                                {0, 0, 2, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 3, 2, 1}})),
                                ov::element::f32,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 2, 2}})),
                                ov::element::i8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 3, 2, 2}})),
                                ov::element::i8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 4, 4}})),
                                ov::element::i8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 2},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{2, 1, 2, 4}})),
                                ov::element::i8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 3, 2, 2},
                                {0, 0, 1, 0, 3},
                                {0, 0, 2, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 3, 2, 1}})),
                                ov::element::i8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 2, 2}})),
                                ov::element::u8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 3, 2, 2}})),
                                ov::element::u8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 4, 4}})),
                                ov::element::u8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 2, 2},
                                {0, 0, 0, 2},
                                {0, 0, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{2, 1, 2, 4}})),
                                ov::element::u8,
                                ov::test::utils::DEVICE_GPU),
        spaceToBatchParamsTuple({1, 1, 3, 2, 2},
                                {0, 0, 1, 0, 3},
                                {0, 0, 2, 0, 0},
                                ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>({{1, 1, 3, 2, 1}})),
                                ov::element::u8,
                                ov::test::utils::DEVICE_GPU),
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_CLDNN, SpaceToBatchLayerTest, ::testing::ValuesIn(stb_only_test_cases()),
                        SpaceToBatchLayerTest::getTestCaseName);
}  // namespace
