// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"
#include "ie_plugin_config.hpp"

using namespace ov::test::behavior;
namespace {

const std::vector<ov::AnyMap> configs = {
    {},
};
const std::vector<ov::AnyMap> swPluginConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(swPluginConfigs)),
                         OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(swPluginConfigs)),
                         OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(swPluginConfigs)),
                         OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelBaseTestOptional,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledModelBaseTestOptional::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVCompiledModelBaseTestOptional,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(swPluginConfigs)),
                         OVCompiledModelBaseTestOptional::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVCompiledModelBaseTestOptional,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(swPluginConfigs)),
                         OVCompiledModelBaseTestOptional::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledModelBaseTestOptional,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(swPluginConfigs)),
                         OVCompiledModelBaseTestOptional::getTestCaseName);
}  // namespace
