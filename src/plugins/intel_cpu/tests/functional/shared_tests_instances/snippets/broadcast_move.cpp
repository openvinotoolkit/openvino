// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/check_broadcast.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov::test::snippets {

namespace {

// Data types that work with snippets fusion (BroadcastMove emitter properly integrated)
const std::vector<ov::element::Type> working_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::u8,
    ov::element::i8
};

const std::vector<CheckBroadcastTestCaseParams> working_test_cases = {
    {
        {{{}, {{1, 3, 4, 4}}}, {{}, {{4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY),
        1,  // 1 node expected (snippets fusion works)
        1   // 1 subgraph expected
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastMove_Working, CheckBroadcast,
                         ::testing::Combine(
                                 ::testing::ValuesIn(working_types),
                                 ::testing::ValuesIn(working_test_cases),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         CheckBroadcast::getTestCaseName);

} // namespace
} // namespace ov::test::snippets
