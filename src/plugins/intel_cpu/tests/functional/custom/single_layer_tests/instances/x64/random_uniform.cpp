// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/random_uniform.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace RandomUniform {

static const std::vector<ElementType> output_prc_nightly = {
        ElementType::f32,
        ElementType::f16,
        ElementType::bf16,
        ElementType::i32,
        ElementType::i64
};

// Need to validate the Kernel corner cases.
static const std::vector<ov::Shape> output_shapes_nightly = {
        {1}, {2}, {24}, {20}, {36}, {624}, {625},
        {2, 2}, {5}, {2, 3}, {7}, {2, 2, 2}, {3, 3}, {2, 5}, {11}, {2, 3, 2}, {13}, {2, 7}, {3, 5},
        {4, 4}, {1, 17}, {2, 9}, {19}, {4, 5}, {21}, {11, 2}, {23, 1}, {4, 2, 3}, {5, 5}, {26}, {1, 27}, {14, 2},
        {29}, {10, 3}, {31}, {2, 8, 2}, {33}, {17, 2}, {5, 7}, {2, 3, 2, 3}, {37}, {2, 19}, {2, 20}, {41}, {42},
        {43}, {22, 2}, {3, 5, 3}, {5, 2, 5}, {1, 3, 1, 17, 1}, {26, 2}, {53}, {54}, {55}, {56}, {57}, {58}, {59},
        {2, 32}, {99}, {127}, {128}, {129}, {199}, {255}, {499}, {997}, {1753}, {2899}
};

INSTANTIATE_TEST_SUITE_P(nightly_Param, RandomUniformLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(output_shapes_nightly),
                ::testing::Values(std::tuple<double, double>{-31, 17}),
                ::testing::Values(ElementType::i32),
                ::testing::ValuesIn(output_prc_nightly),
                ::testing::Values(3),
                ::testing::Values(1),
                ::testing::Values(ov::op::PhiloxAlignment::TENSORFLOW,
                                  ov::op::PhiloxAlignment::PYTORCH),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        RandomUniformLayerTestCPU::getTestCaseName);

}  // namespace RandomUniform
}  // namespace test
}  // namespace ov
