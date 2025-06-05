// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

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
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_f32,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(MLPFunction::WeightFormat::FP32),
                                            ::testing::ValuesIn(activations),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(1),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_bf16,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(MLPFunction::WeightFormat::FP32),
                                            ::testing::ValuesIn(activations),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::Values(4),
                                            ::testing::Values(4),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MLP::getTestCaseName);



}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
