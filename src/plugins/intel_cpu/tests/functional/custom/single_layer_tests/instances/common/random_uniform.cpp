// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/random_uniform.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace RandomUniform {

static const std::vector<ElementType> shape_prc = {
        ElementType::i32,
        ElementType::i64
};

static const std::vector<ov::Shape> output_shapes = {
        {500},
        {4, 3, 210}
};

static const std::vector<uint64_t> global_seed = {
        0, 8
};

static const std::vector<uint64_t> operational_seed = {
        0, 3, 5
};

static const std::vector<std::tuple<double, double>> min_max = {
        {0, 50},
        {-50, 50},
        {-50, 0}
};

static const std::vector<ov::op::PhiloxAlignment> alignment = {
        ov::op::PhiloxAlignment::TENSORFLOW,
        ov::op::PhiloxAlignment::PYTORCH
};

INSTANTIATE_TEST_SUITE_P(smoke_Param, RandomUniformLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(output_shapes),
                ::testing::ValuesIn(min_max),
                ::testing::ValuesIn(shape_prc),
                ::testing::Values(ElementType::f32, ElementType::i32),
                ::testing::ValuesIn(global_seed),
                ::testing::ValuesIn(operational_seed),
                ::testing::ValuesIn(alignment),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        RandomUniformLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ParamConst, RandomUniformLayerTestCPU,
        ::testing::Combine(
                ::testing::Values(output_shapes[0]),
                ::testing::Values(min_max[0]),
                ::testing::Values(ElementType::i32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(1),
                ::testing::Values(0),
                ::testing::ValuesIn(alignment),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        RandomUniformLayerTestCPU::getTestCaseName);

}  // namespace RandomUniform
}  // namespace test
}  // namespace ov
