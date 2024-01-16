// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/callback.hpp"

#include <vector>

using namespace BehaviorTestsDefinitions;

namespace {
const std::vector<std::map<std::string, std::string>> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         InferRequestCallbackTests::getTestCaseName);
}  // namespace
