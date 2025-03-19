// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/batch_to_space.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::BatchToSpaceLayerTest;
using ov::test::batchToSpaceParamsTuple;

auto bts_only_test_cases = []() {
    return std::vector<batchToSpaceParamsTuple>{batchToSpaceParamsTuple({1, 2, 2},
                                                                        {0, 0, 0},
                                                                        {0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 1, 1}})),
                                                                        ov::element::f32,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 1, 1, 1}})),
                                                                        ov::element::f32,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 3, 1, 1}})),
                                                                        ov::element::f32,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 1, 2, 2}})),
                                                                        ov::element::f32,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {8, 1, 1, 2}})),
                                                                        ov::element::f32,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 3, 2, 2},
                                                                        {0, 0, 1, 0, 3},
                                                                        {0, 0, 2, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {24, 1, 2, 1, 2}})),
                                                                        ov::element::f32,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 1, 1, 1}})),
                                                                        ov::element::i8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 3, 1, 1}})),
                                                                        ov::element::i8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 1, 2, 2}})),
                                                                        ov::element::i8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {8, 1, 1, 2}})),
                                                                        ov::element::i8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 3, 2, 2},
                                                                        {0, 0, 1, 0, 3},
                                                                        {0, 0, 2, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {24, 1, 2, 1, 2}})),
                                                                        ov::element::i8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 1, 1, 1}})),
                                                                        ov::element::u8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 3, 1, 1}})),
                                                                        ov::element::u8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {4, 1, 2, 2}})),
                                                                        ov::element::u8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 2, 2},
                                                                        {0, 0, 0, 0},
                                                                        {0, 0, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {8, 1, 1, 2}})),
                                                                        ov::element::u8,
                                                                        ov::test::utils::DEVICE_GPU),
                                                batchToSpaceParamsTuple({1, 1, 3, 2, 2},
                                                                        {0, 0, 1, 0, 3},
                                                                        {0, 0, 2, 0, 0},
                                                                        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                                                        {24, 1, 2, 1, 2}})),
                                                                        ov::element::u8,
                                                                        ov::test::utils::DEVICE_GPU)};
};

INSTANTIATE_TEST_SUITE_P(smoke_CLDNN, BatchToSpaceLayerTest, ::testing::ValuesIn(bts_only_test_cases()),
                        BatchToSpaceLayerTest::getTestCaseName);
}  // namespace
