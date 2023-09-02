// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/callback.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> multiConfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU)},
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestCallbackTests,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::DEVICE_MULTI),
                ::testing::ValuesIn(multiConfigs)),
        OVInferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestCallbackTests,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::DEVICE_AUTO),
                ::testing::ValuesIn(multiConfigs)),
        OVInferRequestCallbackTests::getTestCaseName);
}  // namespace
