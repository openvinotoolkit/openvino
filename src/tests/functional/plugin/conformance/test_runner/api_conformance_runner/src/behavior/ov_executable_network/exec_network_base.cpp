// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"
#include "ie_plugin_config.hpp"
#include "ov_api_conformance_helpers.hpp"


namespace {
using namespace ov::test::behavior;
using namespace ov::test::conformance;

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVCompiledModelBaseTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::Values(pluginConfig)),
                        OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model, OVCompiledModelBaseTestOptional,
                        ::testing::Combine(
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::Values(pluginConfig)),
                        OVCompiledModelBaseTestOptional::getTestCaseName);
}  // namespace
