// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_run.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> remoteConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         RemoteRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(remoteConfigs)),
                         RemoteRunTests::getTestCaseName);
