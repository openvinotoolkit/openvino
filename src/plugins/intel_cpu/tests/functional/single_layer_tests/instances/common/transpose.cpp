// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/transpose.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Transpose {

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {}, {}, {}};
const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {}, {}, {}};

const std::vector<ElementType> netPrecisions = {
        ElementType::i8,
        ElementType::f32
};

const std::vector<std::vector<size_t>> inputOrderPerChannels4D = {
        std::vector<size_t>{0, 1, 2, 3},
        std::vector<size_t>{0, 2, 1, 3},
        std::vector<size_t>{1, 0, 2, 3},
        std::vector<size_t>{},
};

const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nchw,
};

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC16_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC16()),
                                 ::testing::ValuesIn(inputOrder4D()),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(emptyPluginConfig),
                                 ::testing::ValuesIn(CPUParams4D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC32_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC32()),
                                 ::testing::ValuesIn(inputOrder4D()),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(emptyPluginConfig),
                                 ::testing::ValuesIn(CPUParams4D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4D()),
                                 ::testing::ValuesIn(inputOrder4D()),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(emptyPluginConfig),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC16_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC16()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(emptyPluginConfig),
                                 ::testing::Values(cpuParams_nhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC32_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC32()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(emptyPluginConfig),
                                 ::testing::Values(cpuParams_nhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4D()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(emptyPluginConfig),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

} // namespace Transpose
} // namespace CPULayerTestsDefinitions
