// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/wait.hpp"

#include <vector>

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestWaitTests::getTestCaseName);

}  // namespace
