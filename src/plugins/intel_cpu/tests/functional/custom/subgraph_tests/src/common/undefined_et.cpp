// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "custom/subgraph_tests/include/undefined_et.hpp"

namespace ov {
namespace test {
namespace {

static const std::vector<ElementType> data_et = {
        element::f32,
        element::f16,
        element::bf16
};

static const std::vector<ov::AnyMap> plugin_config{{{hint::execution_mode.name(), hint::ExecutionMode::ACCURACY}},
                                                   {{hint::execution_mode.name(), hint::ExecutionMode::PERFORMANCE},
                                                    {hint::inference_precision.name(), element::dynamic}}};

INSTANTIATE_TEST_SUITE_P(smoke_, UndefinedEtSubgraphTest,
        ::testing::Combine(
                ::testing::ValuesIn(data_et),
                ::testing::ValuesIn(plugin_config)),
        UndefinedEtSubgraphTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
