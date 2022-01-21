// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

const std::vector<std::map<std::string, std::string>> HeteroConfigs = {
            {{"TARGET_FALLBACK", CommonTestUtils::DEVICE_TEMPLATE}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(HeteroConfigs)),
                        OVInferenceChaining::getTestCaseName);

}  // namespace
