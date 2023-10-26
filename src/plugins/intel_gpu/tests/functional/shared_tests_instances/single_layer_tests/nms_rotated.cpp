// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/nms_rotated.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::test;

const std::vector<int64_t> maxOutBoxPerClass = {5, 20};
const std::vector<float> threshold = {0.3f, 0.7f};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<ElementType> outType = {ElementType::i32, ElementType::i64};
const std::vector<bool> clockwise = {true, false};

const std::vector<ElementType> inputPrecisions = {ElementType::f32, ElementType::f16};

static const std::vector<std::vector<InputShape>> input_shapes = {
    {
        { {}, {{2, 50, 5}} },
        { {}, {{2, 50, 50}} }
    },
    {
        { {}, {{9, 10, 5}} },
        { {}, {{9, 10, 10}} }
    }
};

const ov::AnyMap empty_plugin_config{};

INSTANTIATE_TEST_SUITE_P(smoke_NmsRotatedLayerTest, NmsRotatedLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(input_shapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(ElementType::i32),
                ::testing::Values(ElementType::f32),
                ::testing::ValuesIn(outType),
                ::testing::ValuesIn(maxOutBoxPerClass),
                ::testing::ValuesIn(threshold),
                ::testing::ValuesIn(threshold),
                ::testing::ValuesIn(sortResDesc),
                ::testing::ValuesIn(clockwise),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::Values(true),
                ::testing::Values(true),
                ::testing::Values(empty_plugin_config),
                ::testing::Values(utils::DEVICE_GPU)),
        NmsRotatedLayerTest::getTestCaseName);
