// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_exec_graph_info.hpp"

using namespace ov::test;
namespace {
const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16};

const std::vector<std::map<std::string, std::string>> configs = {
    {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVExecGraphTests,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         OVExecGraphTests::getTestCaseName);

}  // namespace
