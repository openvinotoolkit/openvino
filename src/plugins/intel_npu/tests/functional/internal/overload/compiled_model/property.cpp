// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/intel_npu/properties.hpp>
#include <vector>

#include "behavior/compiled_model/properties.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;

namespace {

std::vector<std::pair<std::string, ov::Any>> exe_network_supported_properties = {
    {ov::hint::num_requests.name(), ov::Any(8)},
    {ov::hint::enable_cpu_pinning.name(), ov::Any(true)},
    {ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::enable_profiling.name(), ov::Any(true)},
    {ov::device::id.name(), ov::Any(ov::test::utils::getDeviceNameID(ov::test::utils::getDeviceName()))},
    {ov::optimal_number_of_infer_requests.name(), ov::Any(2)},
};

std::vector<std::pair<std::string, ov::Any>> exe_network_immutable_properties = {
    {std::make_pair(ov::optimal_number_of_infer_requests.name(), ov::Any(2))},
    {std::make_pair(ov::hint::enable_cpu_pinning.name(), ov::Any(false))},
    {std::make_pair(ov::supported_properties.name(), ov::Any("deadbeef"))},
    {std::make_pair(ov::model_name.name(), ov::Any("deadbeef"))}};

// ExecutableNetwork Properties tests
class ClassExecutableNetworkGetPropertiesTestNPU
    : public OVCompiledModelPropertiesBase,
      public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, ov::Any>>> {
protected:
    std::string deviceName;
    std::string configKey;
    ov::Any configValue;
    ov::Core ie;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVCompiledModelPropertiesBase::SetUp();
        deviceName = std::get<0>(GetParam());
        std::tie(configKey, configValue) = std::get<1>(GetParam());

        model = ov::test::utils::make_conv_pool_relu();
    }
    static std::string getTestCaseName(
        testing::TestParamInfo<std::tuple<std::string, std::pair<std::string, ov::Any>>> obj) {
        std::string targetDevice;
        std::pair<std::string, ov::Any> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        std::ostringstream result;
        result << "targetDevice=" << ov::test::utils::getDeviceNameTestCase(targetDevice) << "_";
        result << "config=(" << configuration.first << "=" << configuration.second.as<std::string>() << ")";
        result << "_targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        return result.str();
    }
};

using ClassExecutableNetworkTestSuite1NPU = ClassExecutableNetworkGetPropertiesTestNPU;

TEST_P(ClassExecutableNetworkTestSuite1NPU, PropertyIsSupportedAndImmutableAndGet) {
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel exeNetwork = ie.compile_model(model, deviceName);
    OV_ASSERT_NO_THROW(properties = exeNetwork.get_property(ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());

    OV_ASSERT_NO_THROW(exeNetwork.get_property(configKey));
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassExecutableNetworkGetPropertiesTestNPU,
                         ClassExecutableNetworkTestSuite1NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(exe_network_supported_properties)),
                         ClassExecutableNetworkTestSuite1NPU::getTestCaseName);

using ClassExecutableNetworkTestSuite2NPU = ClassExecutableNetworkGetPropertiesTestNPU;

TEST_P(ClassExecutableNetworkTestSuite2NPU, PropertyIsSupportedAndImmutableAndCanNotSet) {
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel exeNetwork = ie.compile_model(model, deviceName);
    OV_ASSERT_NO_THROW(properties = exeNetwork.get_property(ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());

    ASSERT_THROW(exeNetwork.set_property({{configKey, configValue}}), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassExecutableNetworkTestSuite2NPU,
                         ClassExecutableNetworkTestSuite2NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(exe_network_immutable_properties)),
                         ClassExecutableNetworkTestSuite2NPU::getTestCaseName);

}  // namespace

namespace {

std::vector<std::pair<std::string, ov::Any>> plugin_public_mutable_properties = {
    {ov::hint::num_requests.name(), ov::Any(5)},
    {ov::enable_profiling.name(), ov::Any(true)},
    {ov::compilation_num_threads.name(), ov::Any(1)},
    {ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::enable_cpu_pinning.name(), ov::Any(true)},
    {ov::log::level.name(), ov::Any(ov::log::Level::ERR)},
    {ov::device::id.name(), ov::Any(ov::test::utils::getDeviceNameID(ov::test::utils::getDeviceName()))},
};

std::vector<std::pair<std::string, ov::Any>> plugin_internal_mutable_properties = {
    {ov::intel_npu::compilation_mode_params.name(), ov::Any("use-user-precision=false propagate-quant-dequant=0")},
    {ov::intel_npu::dma_engines.name(), ov::Any(1)},
    {ov::intel_npu::platform.name(), ov::Any(ov::intel_npu::Platform::AUTO_DETECT)},
    {ov::intel_npu::compilation_mode.name(), ov::Any("DefaultHW")},
    {ov::intel_npu::max_tiles.name(), ov::Any(8)},
    {ov::intel_npu::stepping.name(), ov::Any(4)},
    {ov::intel_npu::dpu_groups.name(), ov::Any(2)},
    {ov::intel_npu::defer_weights_load.name(), ov::Any(true)},
};

std::vector<std::pair<std::string, ov::Any>> plugin_public_immutable_properties = {
    {ov::device::uuid.name(), ov::Any("deadbeef")},
    {ov::supported_properties.name(), {ov::device::full_name.name()}},
    {ov::num_streams.name(), ov::Any(ov::streams::Num(4))},
    {ov::available_devices.name(), ov::Any(std::vector<std::string>{"deadbeef"})},
    {ov::device::capabilities.name(), ov::Any(std::vector<std::string>{"deadbeef"})},
    {ov::range_for_async_infer_requests.name(),
     ov::Any(std::tuple<unsigned int, unsigned int, unsigned int>{0, 10, 1})},
    {ov::range_for_streams.name(), ov::Any(std::tuple<unsigned int, unsigned int>{0, 10})},
    {ov::optimal_number_of_infer_requests.name(), ov::Any(4)},
    {ov::intel_npu::device_alloc_mem_size.name(), ov::Any(2)},
    {ov::intel_npu::device_total_mem_size.name(), ov::Any(2)},
};

// Plugin Properties tests
class ClassPluginPropertiesTestNPU
    : public OVCompiledModelPropertiesBase,
      public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, ov::Any>>> {
protected:
    std::string deviceName;
    std::string configKey;
    ov::Any configValue;
    ov::Core ie;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVCompiledModelPropertiesBase::SetUp();
        deviceName = std::get<0>(GetParam());
        std::tie(configKey, configValue) = std::get<1>(GetParam());
    }
    static std::string getTestCaseName(
        testing::TestParamInfo<std::tuple<std::string, std::pair<std::string, ov::Any>>> obj) {
        std::string targetDevice;
        std::pair<std::string, ov::Any> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        std::ostringstream result;
        result << "targetDevice=" << ov::test::utils::getDeviceNameTestCase(targetDevice) << "_";
        result << "config=(" << configuration.first << "=" << configuration.second.as<std::string>() << ")";
        result << "_targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
        return result.str();
    }
};

using ClassPluginPropertiesTestSuite0NPU = ClassPluginPropertiesTestNPU;

TEST_P(ClassPluginPropertiesTestSuite0NPU, CanSetGetPublicMutableProperty) {
    std::vector<ov::PropertyName> properties;

    OV_ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_TRUE(it->is_mutable());

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, {{configKey, configValue}}));

    ov::Any retrieved_value;
    OV_ASSERT_NO_THROW(retrieved_value = ie.get_property(deviceName, configKey));

    ASSERT_EQ(retrieved_value.as<std::string>(), configValue.as<std::string>());
}

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_ClassPluginPropertiesTestNPU,
                         ClassPluginPropertiesTestSuite0NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_mutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

using ClassPluginPropertiesTestSuite1NPU = ClassPluginPropertiesTestNPU;

TEST_P(ClassPluginPropertiesTestSuite1NPU, CanSetGetInternalMutableProperty) {
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, {{configKey, configValue}}));

    ov::Any retrieved_value;
    OV_ASSERT_NO_THROW(retrieved_value = ie.get_property(deviceName, configKey));

    ASSERT_EQ(retrieved_value.as<std::string>(), configValue.as<std::string>());
}

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_ClassPluginPropertiesTestNPU,
                         ClassPluginPropertiesTestSuite1NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_internal_mutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

using ClassPluginPropertiesTestSuite2NPU = ClassPluginPropertiesTestNPU;

TEST_P(ClassPluginPropertiesTestSuite2NPU, CanNotSetImmutableProperty) {
    std::vector<ov::PropertyName> properties;

    OV_ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());

    ov::Any orig_value;
    OV_ASSERT_NO_THROW(orig_value = ie.get_property(deviceName, configKey));

    ASSERT_THROW(ie.set_property(deviceName, {{configKey, configValue}}), ov::Exception);

    ov::Any after_value;
    OV_ASSERT_NO_THROW(after_value = ie.get_property(deviceName, configKey));

    ASSERT_EQ(orig_value.as<std::string>(), after_value.as<std::string>());
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassPluginPropertiesTest,
                         ClassPluginPropertiesTestSuite2NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_immutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

using ClassPluginPropertiesTestSuite3NPU = ClassPluginPropertiesTestNPU;

TEST_P(ClassPluginPropertiesTestSuite3NPU, CanGetPropertyWithOptionsNotAffectingCore) {
    std::vector<ov::PropertyName> properties;

    OV_ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());

    ov::Any retrieved_value;
    OV_ASSERT_NO_THROW(retrieved_value = ie.get_property(deviceName, configKey));

    ov::Any retrieved_value_with_options;
    OV_ASSERT_NO_THROW(retrieved_value_with_options = ie.get_property(
                           deviceName,
                           configKey,
                           {{ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::THROUGHPUT)}}));

    ov::Any retrieved_value2;
    OV_ASSERT_NO_THROW(retrieved_value2 = ie.get_property(deviceName, configKey));

    ASSERT_EQ(retrieved_value.as<std::string>(), retrieved_value2.as<std::string>());
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassPluginPropertiesOptsTest1NPU,
                         ClassPluginPropertiesTestSuite3NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_immutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_ClassPluginPropertiesOptsTest2NPU,
                         ClassPluginPropertiesTestSuite3NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_mutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

using ClassPluginPropertiesTestSuite4NPU = ClassExecutableNetworkGetPropertiesTestNPU;

TEST_P(ClassPluginPropertiesTestSuite4NPU, CanNotSetGetInexistentProperty) {
    // ie.set_property won't call plugin Engine::SetConfig due to empty string-ov::Plugin map from core_impl
    // workaround to overcome this is to call first ie.get_property which calls get_plugin() from core_impl and
    // populates plugin map
    std::vector<ov::PropertyName> properties;
    OV_ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    ASSERT_THROW(ie.set_property(deviceName, {{configKey, configValue}}), ov::Exception);

    ASSERT_THROW(auto property1 = ie.get_property(deviceName, configKey), ov::Exception);

    ASSERT_THROW(ov::CompiledModel compiled_model1 = ie.compile_model(model, deviceName, {{configKey, configValue}}),
                 ov::Exception);

    ov::CompiledModel compiled_model2;

    OV_ASSERT_NO_THROW(compiled_model2 = ie.compile_model(model, deviceName));

    ASSERT_THROW(compiled_model2.set_property({{configKey, configValue}}),
                 ov::Exception);  // Expect to throw due to unimplemented method

    ASSERT_THROW(auto property2 = compiled_model2.get_property(configKey),
                 ov::Exception);  // Expect to throw due to unsupported config
}

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests_ClassExecutableNetworkGetPropertiesTestNPU,
    ClassPluginPropertiesTestSuite4NPU,
    ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                       ::testing::ValuesIn({std::make_pair<std::string, ov::Any>("THISCONFIGKEYNOTEXIST",
                                                                                 ov::Any("THISCONFIGVALUENOTEXIST"))})),
    ClassPluginPropertiesTestSuite4NPU::getTestCaseName);

}  // namespace
