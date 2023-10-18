// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/nms_rotated.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace NmsRotated {

static const std::vector<std::vector<InputShape>> input_shapes = {
    {
        { {}, {{1, 5, 5}} },
        { {}, {{1, 7, 5}} }
    },
    {
        { {}, {{2, 9, 5}} },
        { {}, {{2, 15, 9}} }
    },
    {
        { {}, {{5, 17, 5}} },
        { {}, {{5, 7, 17}} }
    },
    {
        { {}, {{9, 75, 5}} },
        { {}, {{9, 55, 75}} }
    },
    {
        { {-1, -1,  5}, {{5, 20, 5},  {3, 50,  5},  {2, 99,  5}} },
        { {-1, -1, -1}, {{5, 30, 20}, {3, 100, 50}, {2, 133, 99}} }
    }
};

static const std::vector<std::vector<InputShape>> input_shapes_nightly = {
    {
        { {}, {{3, 11, 5}} },
        { {}, {{3, 15, 11}} }
    },
    {
        { {}, {{15, 29, 5}} },
        { {}, {{15, 31, 29}} }
    },
    {
        { {}, {{21, 64, 5}} },
        { {}, {{21, 32, 64}} }
    },
    {
        { {-1, -1,  5}, {{7, 35, 5},  {7, 35,  5},  {7, 35,  5}} },
        { {-1, -1, -1}, {{7, 30, 35}, {7, 100, 35}, {7, 133, 35}} }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_, NmsRotatedLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(input_shapes),          // Input shapes
                ::testing::Values(ElementType::f32),        // Boxes and scores input precisions
                ::testing::Values(ElementType::i32),        // Max output boxes input precisions
                ::testing::Values(ElementType::f32),        // Thresholds precisions
                ::testing::Values(ElementType::i32),        // Output type
                ::testing::Values(5, 20),                   // Max output boxes per class
                ::testing::Values(0.3f, 0.7f),              // IOU threshold
                ::testing::Values(0.3f, 0.7f),              // Score threshold
                ::testing::Values(true, false),             // Sort result descending
                ::testing::Values(true, false),             // Clockwise
                ::testing::Values(false),                   // Is 1st input constant
                ::testing::Values(false),                   // Is 2nd input constant
                ::testing::Values(false),                   // Is 3rd input constant
                ::testing::Values(false),                   // Is 4th input constant
                ::testing::Values(false),                   // Is 5th input constant
                ::testing::Values(emptyCPUSpec),            // CPU specific params
                ::testing::Values(empty_plugin_config)),    // Additional plugin configuration
        NmsRotatedLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_, NmsRotatedLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(input_shapes_nightly),
                ::testing::Values(ElementType::f16, ElementType::bf16),
                ::testing::Values(ElementType::i64),
                ::testing::Values(ElementType::f16, ElementType::bf16),
                ::testing::Values(ElementType::i64),
                ::testing::Values(10),
                ::testing::Values(0.5f),
                ::testing::Values(0.4f),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true),
                ::testing::Values(true),
                ::testing::Values(true),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        NmsRotatedLayerTestCPU::getTestCaseName);

} // namespace NmsRotated
} // namespace CPULayerTestsDefinitions
