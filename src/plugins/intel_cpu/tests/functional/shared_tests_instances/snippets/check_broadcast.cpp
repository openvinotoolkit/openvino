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
    // TODO: 105804
    //ov::element::i32,
    ov::element::f32
};

const std::vector<CheckBroadcastTestCaseParams> test_cases = {
    // broadcast is neccessary
    {
        {{{}, {{1, 3, 4, 4}}}, {{}, {{4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, -1),
        1,
        0
    },
    {
        {{{}, {{1, 3, 4, 4}}}, {{}, {{4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 2),
        1,
        0
    },
    // DS
    {
       {{{1, 3, -1, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}, {{-1, {1, 4}}, {{4, 4}, {1, 3}, {4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, -1),
        1,
        0
    },
    {
       {{{1, 3, -1, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}, {{-1, {1, 4}}, {{4, 4}, {1, 3}, {4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 2),
        1,
        0
    },

    // broadcast is not neccessary
    {
        {{{}, {{1, 3, 4, 4}}}, {{}, {{1, 3, 4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, -1),
        1,
        1
    },
    {
        {{{}, {{1, 3, 4, 4}}}, {{}, {{1, 3, 4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0),
        1,
        1
    },
    // DS
    {
       {{{1, 3, -1, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}, {{-1, 3, {1, 4}, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, -1),
        1,
        1
    },
    {
       {{{1, 3, -1, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}, {{-1, 3, {1, 4}, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0),
        1,
        1
    },

    // any other PDPD
    {
        {{{}, {{1, 3, 4, 4}}}, {{}, {{4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY, -1),
        1,
        1
    },
    {
        {{{}, {{1, 3, 4, 4}}}, {{}, {{4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY, 0),
        1,
        1
    },
    {
        {{{}, {{1, 3, 4, 4}}}, {{}, {{4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY, 2),
        1,
        1
    },
    // DS
    {
       {{{1, 3, -1, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}, {{-1, {1, 4}}, {{4, 4}, {1, 3}, { 4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY, -1),
        1,
        1
    },
    {
       {{{1, 3, -1, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}, {{-1, {1, 4}}, {{4, 4}, {1, 3}, {4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY, 0),
        1,
        1
    },
    {
       {{{1, 3, -1, {1, 4}}, {{1, 3, 4, 4}, {1, 3, 1, 3}, {1, 3, 4, 4}}}, {{-1, {1, 4}}, {{4, 4}, {1, 3}, {4, 4}}}},
        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY, 2),
        1,
        1
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_CheckBroadcast, CheckBroadcast,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_types),
                                 ::testing::ValuesIn(test_cases),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         CheckBroadcast::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov
