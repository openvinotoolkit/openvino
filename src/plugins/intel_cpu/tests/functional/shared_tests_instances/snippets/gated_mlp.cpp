// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/gated_mlp.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

static std::vector<utils::ActivationTypes> activations {
    utils::ActivationTypes::Relu, utils::ActivationTypes::Swish, utils::ActivationTypes::Gelu,
};

static std::vector<std::pair<InputShape, std::vector<Shape>>> shapes {
    {
        InputShape{{-1, -1, 1024}, {{2, 128, 1024}, {2, 1, 1024}, {16, 128, 1024}, {2, 1, 1024}}},
        std::vector<Shape>{{256, 1024}, {256, 1024}, {1024, 256}}
    },
    {
        InputShape{{-1, -1, 757}, {{21, 39, 757}, {12, 121, 757}, {12, 3, 757}}},
        std::vector<Shape>{{127, 757}, {127, 757}, {757, 127}}
    },
    {
        InputShape{{-1, -1, 1024}, {{3, 64, 1024}, {3, 1, 1024}}},
        std::vector<Shape>{{256, 1024}, {256, 1024}, {1024, 256}}
    },
    {
        InputShape{{}, {{1, 32, 1024}}},
        std::vector<Shape>{{256, 1024}, {256, 1024}, {1024, 256}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GatedMLP_f32,
                         GatedMLP,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(GatedMLPFunction::WeightFormat::FP32),
                                            ::testing::ValuesIn(activations),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(1),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GatedMLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GatedMLP_bf16,
                         GatedMLP,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(GatedMLPFunction::WeightFormat::FP32),
                                            ::testing::ValuesIn(activations),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::Values(3), // 3 x Subgraphs
                                            ::testing::Values(3), // 3 Subgraphs - In/Out Converts + gMLP
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GatedMLP::getTestCaseName);



}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
