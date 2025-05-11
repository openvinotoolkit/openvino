// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/shuffle_channels.hpp"

using ov::test::ShuffleChannelsLayerTest;

namespace {

const std::vector<ov::element::Type> model_types = {
        ov::element::u8,
        ov::element::u16,
        ov::element::f32
};

const std::vector<int> axes = {-4, -3, -2, -1, 0, 1, 2, 3};
const std::vector<int> groups = {1, 2, 3, 6};

const auto shuffle_channels_params_4D = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(groups)
);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels4D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                shuffle_channels_params_4D,
                ::testing::ValuesIn(model_types),
                ::testing::Values(ov::test::static_shapes_to_test_representation({{12, 18, 30, 36}})),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

// ND support tests
INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels6D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(2, 3)),
                ::testing::ValuesIn(model_types),
                ::testing::Values(ov::test::static_shapes_to_test_representation({{24, 6, 12, 18, 30, 36}})),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels5D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(2, 3)),
                ::testing::ValuesIn(model_types),
                ::testing::Values(ov::test::static_shapes_to_test_representation({{6, 12, 18, 30, 36}})),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels3D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(1, 3)),
                ::testing::ValuesIn(model_types),
                ::testing::Values(ov::test::static_shapes_to_test_representation({{18, 30, 36}})),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels2D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(1, 3)),
                ::testing::ValuesIn(model_types),
                ::testing::Values(ov::test::static_shapes_to_test_representation({{18, 30}})),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels1D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(0, 3)),
                ::testing::ValuesIn(model_types),
                ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape{30}})),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

}  // namespace
