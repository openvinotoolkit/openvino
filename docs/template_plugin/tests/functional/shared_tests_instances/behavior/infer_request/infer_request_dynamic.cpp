// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/infer_request_dynamic.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        InferRequestDynamicTests::getTestCaseName);

}  // namespace

