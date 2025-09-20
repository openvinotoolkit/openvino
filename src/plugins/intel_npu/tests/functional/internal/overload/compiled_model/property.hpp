// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/intel_npu/properties.hpp>
#include <vector>

#include "behavior/compiled_model/properties.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/util/log.hpp"

using namespace ov::test::behavior;

namespace {

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
        static uint8_t testCounter = 0;
        result << "_testCounter=" << std::to_string(testCounter++)
               << "_";  // used to avoid same names for different tests
        result << "targetDevice=" << ov::test::utils::getDeviceNameTestCase(targetDevice) << "_";
        result << "config=(" << configuration.first << "=" << configuration.second.as<std::string>() << ")";
        result << "_targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        return result.str();
    }
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
        static uint8_t testCounter = 0;
        result << "_testCounter="
               << std::to_string(testCounter++) + "_";  // used to avoid same names for different tests
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

using ClassPluginPropertiesTestSuite1NPU = ClassPluginPropertiesTestNPU;

TEST_P(ClassPluginPropertiesTestSuite1NPU, CanSetGetInternalMutableProperty) {
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, {{configKey, configValue}}));

    ov::Any retrieved_value;
    OV_ASSERT_NO_THROW(retrieved_value = ie.get_property(deviceName, configKey));

    ASSERT_EQ(retrieved_value.as<std::string>(), configValue.as<std::string>());
}

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

using ClassExecutableNetworkInvalidDeviceIDTestSuite = ClassExecutableNetworkGetPropertiesTestNPU;

TEST_P(ClassExecutableNetworkInvalidDeviceIDTestSuite, InvalidNPUdeviceIDTest) {
    deviceName = configValue.as<std::string>();
    OV_EXPECT_THROW_HAS_SUBSTRING(ov::CompiledModel compiled_model = ie.compile_model(model, deviceName),
                                  ov::Exception,
                                  "Could not find available NPU device");
}

}  // namespace
