// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {
    {}
};

const std::vector<ov::AnyMap> HeteroConfigs = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}
};

const std::vector<ov::AnyMap> AutoConfigs = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                        OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(HeteroConfigs)),
                        OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoConfigs)),
                        OVInferenceChaining::getTestCaseName);
}  // namespace
