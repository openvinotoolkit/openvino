// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/enforce_precision.hpp"
#include <gtest/gtest.h>
#include <ngraph/ngraph.hpp>

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<ov::PartialShape>> input_shapes = {
    {{ 1, 16, 384, 64 }, { 1, 16, 64, 384 }},
};

namespace platform_bf16 {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_EnforcePrecision_bf16, EnforcePrecisionTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(input_shapes),
                            ::testing::Values(7),   // 3 Roll + 3 Reorder + Subgraph
                            ::testing::Values(1),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        EnforcePrecisionTest::getTestCaseName);
} // namespace platform_bf16

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov
