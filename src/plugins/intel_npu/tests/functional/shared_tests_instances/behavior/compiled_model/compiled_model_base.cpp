// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;
namespace {

const std::vector<ov::element::Type> modelTypes{ov::element::f16, ov::element::f32};

const std::vector<ov::AnyMap> compiledModelConfigs = {{}};

// Hetero configs

auto heteroCompiledModelConfigs = []() -> std::vector<ov::AnyMap> {
    std::vector<ov::AnyMap> heteroPluginConfigs(compiledModelConfigs.size());
    for (auto it = compiledModelConfigs.cbegin(); it != compiledModelConfigs.cend(); ++it) {
        auto&& distance = it - compiledModelConfigs.cbegin();
        heteroPluginConfigs.at(distance) = {ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                            ov::device::properties(ov::test::utils::DEVICE_NPU, *it)};
    }
    return heteroPluginConfigs;
}();

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledModelBaseTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(heteroCompiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledModelBaseTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelBaseTestOptional,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledModelBaseTestOptional>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledModelBaseTestOptional,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(heteroCompiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledModelBaseTestOptional>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVAutoExecutableNetworkTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVAutoExecutableNetworkTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVAutoExecutableNetworkTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(heteroCompiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVAutoExecutableNetworkTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         CompiledModelSetType,
                         ::testing::Combine(::testing::ValuesIn(modelTypes),
                                            ::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(heteroCompiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<CompiledModelSetType>);

}  // namespace
