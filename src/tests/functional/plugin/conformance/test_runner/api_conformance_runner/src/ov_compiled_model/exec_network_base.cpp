// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"
#include "ov_api_conformance_helpers.hpp"


namespace {
using namespace ov::test::behavior;
using namespace ov::test::conformance;

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVCompiledModelBaseTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::Values(ov::AnyMap({}))),
                        OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model, OVCompiledModelBaseTestOptional,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::Values(ov::AnyMap({}))),
                        OVCompiledModelBaseTestOptional::getTestCaseName);
}  // namespace
