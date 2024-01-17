// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/nms_rotated.hpp"

namespace {
using LayerTestsDefinitions::NmsRotatedOpTest;

const std::vector<std::vector<ov::test::InputShape>> inShapeParams = {
    {
        { {}, {{2, 50, 5}} },
        { {}, {{2, 50, 50}} }
    },
    {
        { {}, {{9, 10, 5}} },
        { {}, {{9, 10, 10}} }
    }
};

const std::vector<ov::element::Type_t> outType = {ov::element::i32, ov::element::i64};
const std::vector<ov::element::Type_t> inputPrecisions = {ov::element::f32, ov::element::f16};
const ov::AnyMap empty_plugin_config{};

INSTANTIATE_TEST_SUITE_P(smoke_NmsRotatedLayerTest,
                         NmsRotatedOpTest,
                         ::testing::Combine(::testing::ValuesIn(inShapeParams),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(ov::element::i32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(outType),
                                            ::testing::Values(5, 20),
                                            ::testing::Values(0.3f, 0.7f),
                                            ::testing::Values(0.3f, 0.7f),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         NmsRotatedOpTest::getTestCaseName);

} // namespace
