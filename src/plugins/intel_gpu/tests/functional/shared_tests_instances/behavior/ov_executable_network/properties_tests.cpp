// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/properties_tests.hpp"

namespace {
using ov::test::behavior::InferRequestPropertiesTest;

const std::vector<ov::AnyMap> configs = {{}};
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestPropertiesTest,
                         ::testing::Combine(::testing::Values(1u),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPropertiesTest::getTestCaseName);
}  // namespace
