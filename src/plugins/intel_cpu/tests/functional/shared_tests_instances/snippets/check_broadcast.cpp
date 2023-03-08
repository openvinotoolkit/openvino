// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/check_broadcast.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<ov::element::Type> input_types = {
    ov::element::i32,
    ov::element::f32,
    ov::element::i64,
};

const std::vector<CheckBroadcastTestCaseParams> test_cases = {
    {
        {{1, 3, 4, 4}, {4, 4}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, -1),
        1,
        0
    },
    {
        {{1, 3, 4, 4}, {1, 3, 4, 4}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, -1),
        1,
        1
    },
    {
        {{1, 3, 4, 4}, {4, 4}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY, -1),
        1,
        1
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_CheckBroadcast, CheckBroadcast,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_types),
                                 ::testing::ValuesIn(test_cases),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         CheckBroadcast::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov