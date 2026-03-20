// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/transpose.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Transpose {
std::vector<ov::AnyMap> additional_config = {
        {{ov::hint::inference_precision.name(), ov::element::f32.to_string()}},
        {{ov::hint::inference_precision.name(), ov::element::f16.to_string()}}
};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {}, {}, {}};
const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {}, {}, {}};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::i8,
        ov::element::f32
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
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(additional_config),
                                 ::testing::ValuesIn(CPUParams4D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC32_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC32()),
                                 ::testing::ValuesIn(inputOrder4D()),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(additional_config),
                                 ::testing::ValuesIn(CPUParams4D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4D()),
                                 ::testing::ValuesIn(inputOrder4D()),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(additional_config),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC16_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC16()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(additional_config),
                                 ::testing::Values(cpuParams_nhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC32_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC32()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(additional_config),
                                 ::testing::Values(cpuParams_nhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4D()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(additional_config),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

// ─── String Transpose ────────────────────────────────────────────────────────
// Tests cover the executeString() fallback introduced to handle element::string,
// because the JIT permute kernel's switch(data_size) has no case for
// sizeof(std::string)==32 and would silently produce empty output strings.

const std::vector<InputShape> stringInputShapes2D = {
    InputShape{{3, 4}, {{3, 4}}},                        // static 2-D
    InputShape{{-1, -1}, {{2, 3}, {4, 2}, {1, 5}}},     // dynamic 2-D
};
const std::vector<std::vector<size_t>> stringOrders2D = {{1, 0}};

const std::vector<InputShape> stringInputShapes3D = {
    InputShape{{2, 3, 4}, {{2, 3, 4}}},                  // static 3-D
    InputShape{{-1, -1, -1}, {{2, 3, 4}, {1, 5, 2}}},   // dynamic 3-D
};
const std::vector<std::vector<size_t>> stringOrders3D = {{2, 0, 1}, {1, 0, 2}, {0, 2, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_String2D_Transpose, TransposeStringLayerCPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(stringInputShapes2D),
                             ::testing::ValuesIn(stringOrders2D),
                             ::testing::Values(ov::element::string),
                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                             ::testing::Values(ov::AnyMap{}),
                             ::testing::Values(CPUSpecificParams{})),
                         TransposeStringLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_String3D_Transpose, TransposeStringLayerCPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(stringInputShapes3D),
                             ::testing::ValuesIn(stringOrders3D),
                             ::testing::Values(ov::element::string),
                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                             ::testing::Values(ov::AnyMap{}),
                             ::testing::Values(CPUSpecificParams{})),
                         TransposeStringLayerCPUTest::getTestCaseName);

}  // namespace Transpose
}  // namespace test
}  // namespace ov