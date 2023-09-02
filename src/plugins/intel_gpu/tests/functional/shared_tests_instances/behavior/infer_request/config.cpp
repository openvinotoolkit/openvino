// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
auto configs = []() {
    return std::vector<std::map<std::string, std::string>>{{}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestConfigTest,
                        ::testing::Combine(
                                ::testing::Values(1u),
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                         InferRequestConfigTest::getTestCaseName);
}  // namespace
