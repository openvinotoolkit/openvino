// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/callback.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {
    {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCallbackTests,
        ::testing::Combine(
            ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
            ::testing::ValuesIn(configs)),
        OVInferRequestCallbackTests::getTestCaseName);
}  // namespace
