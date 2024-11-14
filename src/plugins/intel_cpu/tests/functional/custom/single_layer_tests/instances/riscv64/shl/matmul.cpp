// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/matmul.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace MatMul {
namespace {
std::vector<CPUSpecificParams> filterSpecificParams_SHL() {
    // replace with shl primitive type
    std::vector<CPUSpecificParams> specificParams;
    specificParams.push_back(CPUSpecificParams{{}, {}, {"gemm_shl"}, "gemm_shl"});
    return specificParams;
}

const std::vector<ShapeRelatedParams>& FC_2DParams() {
    static const std::vector<ShapeRelatedParams> params = {
        {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, true}},
        {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, true}},
        {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, true}},
        {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, true}},

        {
            {
                {{-1, -1}, {{20, 60}, {20, 60}}},
                {{60, 120}, {{60, 120}, {60, 120}}}
            },
            {false, true}
        },
        {
            {
                {{{0, 100}, {0, 12}}, {{20, 1}, {14, 1}, {20, 1}, {14, 1}}},
                {{1, 120}, {{1, 120}, {1, 120}, {1, 120}, {1, 120}}}
            },
            {false, true}
        },
    };
    return params;
}


const auto testParams3D_SHL_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(FC_2DParams()),
                                                        ::testing::Values(ElementType::f32),
                                                        ::testing::Values(ElementType::undefined),
                                                        ::testing::Values(ElementType::undefined),
                                                        ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                                        ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                        ::testing::Values(emptyAdditionalConfig())),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::Values(emptyFusingSpec),
                                             ::testing::ValuesIn(filterSpecificParams_SHL()));
INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_SHL, MatMulLayerCPUTest, testParams3D_SHL_smoke, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams>& FC_3DParams() {
    static const std::vector<ShapeRelatedParams> params = {
        {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, true}},
        {static_shapes_to_test_representation({{1, 1, 120}, {120, 120}}), {false, true}},
        {static_shapes_to_test_representation({{3, 1, 120}, {120, 120}}), {false, true}},
        {static_shapes_to_test_representation({{2, 32, 1}, {1, 50}}), {false, true}},

        {
            {
                {{1, 5, 32}, {{1, 5, 32}, {1, 5, 32}}},
                {{32, 3}, {{32, 3}, {32, 3}}}
            },
            {false, true}
        },

        {
            {
                {{{0, 60}, {0, 60}, {0, 60}}, {{1, 3, 14}, {1, 7, 14}}},
                {{14, 10}, {{14, 10}, {14, 10}}}
            },
            {false, true}
        },
    };
    return params;
}

const auto testParams2D_SHL_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(FC_3DParams()),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values(emptyAdditionalConfig())),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::Values(emptyFusingSpec),
                                             ::testing::ValuesIn(filterSpecificParams_SHL()));
INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_SHL, MatMulLayerCPUTest, testParams2D_SHL_smoke, MatMulLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace MatMul
}  // namespace test
}  // namespace ov