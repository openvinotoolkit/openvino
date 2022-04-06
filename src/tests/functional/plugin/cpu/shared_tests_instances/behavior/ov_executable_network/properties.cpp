// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "ie_system_conf.h"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
        {ov::num_streams(-100)},
};

const std::vector<ov::AnyMap> hetero_inproperties = {
        {ov::num_streams(-100)},
};

const std::vector<ov::AnyMap> multi_inproperties = {
        {ov::num_streams(-100)},
};


const std::vector<ov::AnyMap> auto_inproperties = {
        {ov::num_streams(-100)},
};


const std::vector<ov::AnyMap> auto_batch_inproperties = {
        {ov::num_streams(-100)},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(CommonTestUtils::DEVICE_CPU) + "(4)"}, {ov::auto_batch_timeout(-1)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::ValuesIn(inproperties)),
                        OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVCompiledModelPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(hetero_inproperties)),
                        OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVCompiledModelPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(multi_inproperties)),
                        OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVCompiledModelPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(auto_inproperties)),
                        OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVCompiledModelPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                ::testing::ValuesIn(auto_batch_inproperties)),
                        OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

#if (defined(__APPLE__) || defined(_WIN32))
auto default_affinity = [] {
        auto numaNodes = InferenceEngine::getAvailableNUMANodes();
        if (numaNodes.size() > 1) {
                return ov::Affinity::NUMA;
        } else {
                return ov::Affinity::NONE;
        }
}();
#else
auto default_affinity = ov::Affinity::CORE;
#endif

const std::vector<ov::AnyMap> default_properties = {
        {ov::affinity(default_affinity)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelPropertiesDefaultTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::ValuesIn(default_properties)),
        OVCompiledModelPropertiesDefaultTests::getTestCaseName);

const std::vector<ov::AnyMap> properties = {
        {ov::num_streams(ov::streams::NUMA)},
        {ov::num_streams(ov::streams::AUTO)},
        {ov::num_streams(0), ov::inference_num_threads(1)},
        {ov::num_streams(1), ov::inference_num_threads(1)},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
};

const std::vector<ov::AnyMap> hetero_properties = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::num_streams(ov::streams::AUTO)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
};


const std::vector<ov::AnyMap> multi_properties = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::num_streams(ov::streams::AUTO)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
};

const std::vector<ov::AnyMap> auto_batch_properties = {
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(CommonTestUtils::DEVICE_CPU) + "(4)"}},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(CommonTestUtils::DEVICE_CPU) + "(4)"}, {CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "1"}},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(CommonTestUtils::DEVICE_CPU) + "(4)"}, {ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::ValuesIn(properties)),
        OVCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                ::testing::ValuesIn(hetero_properties)),
        OVCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(multi_properties)),
        OVCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                ::testing::ValuesIn(auto_batch_properties)),
        OVCompiledModelPropertiesTests::getTestCaseName);
} // namespace