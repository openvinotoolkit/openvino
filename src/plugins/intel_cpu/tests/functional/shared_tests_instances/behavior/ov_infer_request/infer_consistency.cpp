// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/infer_consistency.hpp"

using namespace ov::test::behavior;

namespace {
// for deviceConfigs, the deviceConfigs[0] is target device which need to be tested.
// deviceConfigs[1], deviceConfigs[2],deviceConfigs[n] are the devices which will
// be compared with target device, the result of target should be in one of the compared
// device.
using Configs = std::vector<std::pair<std::string, ov::AnyMap>>;

std::vector<Configs> configs = {{{ov::test::utils::DEVICE_CPU, {}}, {ov::test::utils::DEVICE_CPU, {}}}};

INSTANTIATE_TEST_SUITE_P(BehaviorTests,
                         OVInferConsistencyTest,
                         ::testing::Combine(::testing::Values(10),  // inferRequest num
                                            ::testing::Values(10),  // infer counts
                                            ::testing::ValuesIn(configs)),
                         OVInferConsistencyTest::getTestCaseName);
}  // namespace
