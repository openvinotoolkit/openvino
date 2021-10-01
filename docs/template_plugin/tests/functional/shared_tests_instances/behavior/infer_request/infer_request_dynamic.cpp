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

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(ngraph::builder::subgraph::makeSplitConvConcat()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{{{1, 4, 20, 20}, {1, 10, 18, 18}},
                                                                                                                   {{2, 4, 20, 20}, {2, 10, 18, 18}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        InferRequestDynamicTests::getTestCaseName);

}  // namespace
