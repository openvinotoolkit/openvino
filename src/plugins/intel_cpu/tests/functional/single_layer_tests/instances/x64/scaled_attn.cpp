// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/scaled_attn.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace ScaledAttn {
const auto cpuSpec = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};

const std::vector<std::vector<InputShape>> shapes{
    // normal case, shapes of q,k,v are same
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // attn shape: [B, 1, 1, L0+L1]
        {ov::test::InputShape{ov::PartialShape{-1, 1, 1, -1},
            {ov::Shape{1, 1, 1, 100}, ov::Shape{1, 1, 1, 1}, ov::Shape{2, 1, 1, 10}}}
        },
    },
    // heads number of kv is 1, attn mask: [B, H, L1, L0+L1]
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, 64},
            {ov::Shape{1, 1, 100, 64}, ov::Shape{1, 1, 1, 64}, ov::Shape{2, 1, 10, 64}}}
        },
        // attn shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, -1},
            {ov::Shape{1, 8, 100, 100}, ov::Shape{1, 8, 1, 1}, ov::Shape{2, 8, 10, 10}}}
        },
    },
    // heads number of kv is 1, attn mask: [H, L1, L0+L1]
    {
        // q shape
        {ov::test::InputShape{ov::PartialShape{-1, 8, -1, 64},
            {ov::Shape{1, 8, 100, 64}, ov::Shape{1, 8, 1, 64}, ov::Shape{2, 8, 10, 64}}}
        },
        // kv shape
        {ov::test::InputShape{ov::PartialShape{-1, 1, -1, 64},
            {ov::Shape{1, 1, 100, 64}, ov::Shape{1, 1, 1, 64}, ov::Shape{2, 1, 10, 64}}}
        },
        // attn shape
        {ov::test::InputShape{ov::PartialShape{8, -1, -1},
            {ov::Shape{8, 100, 100}, ov::Shape{8, 1, 1}, ov::Shape{8, 10, 10}}}
        },
    },
};

const auto params = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16),
                                                 testing::ValuesIn(shapes),
                                                 testing::Values(true, false),
                                                 testing::Values(true, false),
                                                 testing::Values(true, false),
                                                 testing::Values(ov::test::utils::DEVICE_CPU),
                                                 testing::Values(cpuSpec));

INSTANTIATE_TEST_SUITE_P(smoke_ScaledAttn_CPU,
                         ScaledAttnLayerCPUTest,
                         params,
                         ScaledAttnLayerCPUTest::getTestCaseName);

} // namespace ScaledAttn
} // namespace CPULayerTestsDefinitions