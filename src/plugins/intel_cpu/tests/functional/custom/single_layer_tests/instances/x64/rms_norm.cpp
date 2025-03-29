// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/rms_norm.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace RMSNorm {
const auto cpuSpec = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};

const std::vector<std::vector<InputShape>> shapes{
    // normal
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 1024 + 16 + 1},
            {ov::Shape{1, 8, 1024 + 16 + 1}, ov::Shape{2, 3, 1024 + 16 + 1}}}
        },
        // scale shape
        {ov::test::InputShape{ov::PartialShape{1024 + 16 + 1},
            {ov::Shape{1024 + 16 + 1}, ov::Shape{1024 + 16 + 1}}}
        },
    },
    // small data size
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 31},
            {ov::Shape{1, 8, 31}, ov::Shape{2, 3, 31}}}
        },
        // scale shape
        {ov::test::InputShape{ov::PartialShape{31},
            {ov::Shape{31}, ov::Shape{31}}}
        },
    },
    // scale is scalar
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 1094},
            {ov::Shape{1, 8, 1094}, ov::Shape{2, 3, 1094}}}
        },
        // scale shape
        {ov::test::InputShape{ov::PartialShape{1},
            {ov::Shape{1}, ov::Shape{1}}}
        },
    },
    // decomposition path
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 64},
            {ov::Shape{1, 8, 64}}}
        },
        // scale shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 64},
            {ov::Shape{1, 8, 64}}}
        },
    },
};

const auto params = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16, ElementType::f16),
                                                 testing::ValuesIn(shapes),
                                                 testing::Values(ov::test::utils::DEVICE_CPU),
                                                 testing::Values(cpuSpec));

INSTANTIATE_TEST_SUITE_P(smoke_RMSNorm_CPU,
                         RMSNormLayerCPUTest,
                         params,
                         RMSNormLayerCPUTest::getTestCaseName);

}  // namespace RMSNorm
}  // namespace test
}  // namespace ov
