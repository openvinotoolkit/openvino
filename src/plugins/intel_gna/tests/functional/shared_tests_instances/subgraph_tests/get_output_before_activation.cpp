// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/get_output_before_activation.hpp"

namespace ov {
namespace test {
namespace {
std::vector<size_t> input_sizes = {80, 32, 64, 100};

std::vector<midOutputType> midLayerTypes{midOutputType::Mul, midOutputType::Sub, midOutputType::Sum};

std::vector<ov::AnyMap> configs = {{
                                       {"GNA_COMPACT_MODE", "NO"},
                                       {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                                       {"GNA_SCALE_FACTOR_0", "1638.4"},
                                       {"GNA_SCALE_FACTOR_1", "1638.4"},
                                   },
                                   {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};
}  // namespace

INSTANTIATE_TEST_SUITE_P(OutputBeforeActivation,
                         OutputBeforeActivation,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(input_sizes),
                                            ::testing::ValuesIn(midLayerTypes),
                                            ::testing::ValuesIn(configs)),
                         OutputBeforeActivation::getTestCaseName);
}  // namespace test
}  // namespace ov
