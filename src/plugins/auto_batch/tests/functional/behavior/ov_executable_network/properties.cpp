// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/compiled_model/properties.hpp"

#include "behavior/ov_plugin/properties_tests.hpp"

namespace ov {
namespace test {
namespace behavior {

const std::vector<ov::AnyMap> auto_batch_inproperties = {
    {ov::num_streams(-100)},
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_batch_properties = {
    {{ov::device::priorities.name(), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"}},
    {{ov::device::priorities.name(), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
     {ov::auto_batch_timeout(1)}},
    {{ov::device::priorities.name(), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
     {ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("BATCH:GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("BATCH:GPU"));

const std::vector<ov::AnyMap> batchCorrectConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(nightly_Auto_Batch_OVClassCompileModelWithCorrectPropertiesAutoBatchingTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("BATCH:GPU"), ::testing::ValuesIn(batchCorrectConfigs)));

const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_GPU = {
    {"BATCH:GPU", std::make_pair(ov::AnyMap{}, "GPU.0")}};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::ValuesIn(GetMetricTest_ExecutionDevice_GPU));

INSTANTIATE_TEST_SUITE_P(nightly_HeteroAutoBatchOVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("BATCH"));
}  // namespace behavior
}  // namespace test
}  // namespace ov
