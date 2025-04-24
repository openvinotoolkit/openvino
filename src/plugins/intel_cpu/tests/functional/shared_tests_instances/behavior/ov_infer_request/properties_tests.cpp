// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/properties_tests.hpp"

using namespace ov::test::behavior;
namespace {
const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestPropertiesTest,
                         ::testing::Combine(::testing::Values(1u),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPropertiesTest::getTestCaseName);
}  // namespace
