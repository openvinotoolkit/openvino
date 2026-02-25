// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/intel_npu/properties.hpp>
#include <vector>

#include "behavior/compiled_model/properties.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/core/log.hpp"

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
                                  "Could not find a valid NPU device for the provided configuration.");
}

using CheckCompilerTypeProperty = ClassExecutableNetworkGetPropertiesTestNPU;

TEST_P(CheckCompilerTypeProperty, CheckCompilerTypePropertyFromCompiledModel) {
    std::string platform = ov::test::utils::getTestsPlatformFromEnvironmentOr(deviceName);
    size_t pos0 = platform.find("5010");
    size_t pos1 = platform.find("4000");
    ov::Core core;

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, deviceName));
    auto compiler_type = compiled_model.get_property(ov::intel_npu::compiler_type);

    if (pos0 != std::string::npos || pos1 != std::string::npos) {
        ASSERT_TRUE(compiler_type == ov::intel_npu::CompilerType::PLUGIN);
    } else {
        ASSERT_TRUE(compiler_type == ov::intel_npu::CompilerType::DRIVER);
    }

    OV_ASSERT_NO_THROW(
        compiled_model =
            core.compile_model(model, deviceName, {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}));
    compiler_type = compiled_model.get_property(ov::intel_npu::compiler_type);
    ASSERT_TRUE(compiler_type == ov::intel_npu::CompilerType::DRIVER);

    if (pos0 != std::string::npos || pos1 != std::string::npos) {
        OV_ASSERT_NO_THROW(compiled_model =
                               core.compile_model(model,
                                                  deviceName,
                                                  {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}));
        compiler_type = compiled_model.get_property(ov::intel_npu::compiler_type);
        ASSERT_TRUE(compiler_type == ov::intel_npu::CompilerType::PLUGIN);
    }
}

TEST_P(CheckCompilerTypeProperty, CheckCompilerTypePropertyAfterSettingExtraConfigToGetProperty) {
    std::string platform = ov::test::utils::getTestsPlatformFromEnvironmentOr(deviceName);
    size_t pos0 = platform.find("5010");
    size_t pos1 = platform.find("4000");
    ov::Core core;

    auto test_custom_compiler_type =
        core.get_property(deviceName,
                          ov::intel_npu::compiler_type,
                          {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)});
    ASSERT_TRUE(test_custom_compiler_type == ov::intel_npu::CompilerType::DRIVER);

    test_custom_compiler_type = core.get_property(deviceName, ov::intel_npu::compiler_type);
    if (pos0 != std::string::npos || pos1 != std::string::npos) {
        ASSERT_TRUE(test_custom_compiler_type == ov::intel_npu::CompilerType::PREFER_PLUGIN);
    } else {
        ASSERT_TRUE(test_custom_compiler_type == ov::intel_npu::CompilerType::DRIVER);
    }

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, deviceName));
    auto compiler_type = compiled_model.get_property(ov::intel_npu::compiler_type);

    if (pos0 != std::string::npos || pos1 != std::string::npos) {
        ASSERT_TRUE(compiler_type == ov::intel_npu::CompilerType::PLUGIN);
    } else {
        ASSERT_TRUE(compiler_type == ov::intel_npu::CompilerType::DRIVER);
    }
}

TEST_P(CheckCompilerTypeProperty, CheckLogAfterSettingExtraConfigToGetProperty) {
    std::string logs;
    std::mutex logs_mutex;
    ov::Core core;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    core.set_property(deviceName, ov::log::level(ov::log::Level::INFO));
    core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN));

    ov::util::set_log_callback(log_cb);
    auto compiler_type = core.get_property(
        deviceName,
        ov::intel_npu::compiler_type,
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER), ov::intel_npu::qdq_optimization(true)});
    ov::util::reset_log_callback();

    ASSERT_TRUE(compiler_type == ov::intel_npu::CompilerType::DRIVER);
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);

    compiler_type = core.get_property(deviceName, ov::intel_npu::compiler_type);
    ASSERT_TRUE(compiler_type == ov::intel_npu::CompilerType::PLUGIN);
}

TEST_P(CheckCompilerTypeProperty, CheckLogAfterGettingPropertyWithExtraConfig) {
    std::string logs;
    std::mutex logs_mutex;
    ov::Core core;

    core.set_property(deviceName, ov::log::level(ov::log::Level::INFO));

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(core.get_property(deviceName,
                                         ov::intel_npu::defer_weights_load,
                                         {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}));
    ov::util::reset_log_callback();

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);

    logs.clear();

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(core.get_property(
        deviceName,
        ov::intel_npu::defer_weights_load,
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER), ov::intel_npu::qdq_optimization(true)}));
    ov::util::reset_log_callback();

    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
}

TEST_P(CheckCompilerTypeProperty, SetRuntimeProperty) {
    std::string logs;
    std::mutex logs_mutex;
    ov::Core core;

    core.set_property(deviceName, ov::log::level(ov::log::Level::INFO));

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(
        core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)));
    OV_ASSERT_NO_THROW(core.get_property(deviceName, ov::intel_npu::defer_weights_load));
    ov::util::reset_log_callback();

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);

    logs.clear();

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::intel_npu::qdq_optimization(true)));
    ov::util::reset_log_callback();

    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
}

TEST_P(CheckCompilerTypeProperty, SetCompilerPropertyForDifferentCompiler) {
    std::string logs;
    std::mutex logs_mutex;
    ov::Core core;

    core.set_property(deviceName, ov::log::level(ov::log::Level::INFO));

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(
        core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)));
    OV_ASSERT_NO_THROW(core.get_property(deviceName, ov::intel_npu::defer_weights_load));
    ov::util::reset_log_callback();
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);

    logs.clear();

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::intel_npu::qdq_optimization(true)));
    ov::util::reset_log_callback();
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);

    logs.clear();

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(
        core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)));
    ov::util::reset_log_callback();
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);

    logs.clear();

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::intel_npu::qdq_optimization(true)));
    ov::util::reset_log_callback();
    ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CheckCompilerTypeProperty, GetCompilerVersion) {
    std::string logs;
    std::mutex logs_mutex;
    ov::Core core;

    core.set_property(deviceName, ov::log::level(ov::log::Level::INFO));

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    OV_ASSERT_NO_THROW(
        core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)));
    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(core.get_property(deviceName, ov::intel_npu::compiler_version));
    ov::util::reset_log_callback();
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);

    logs.clear();

    OV_ASSERT_NO_THROW(
        core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)));
    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(core.get_property(deviceName, ov::intel_npu::compiler_version));
    ov::util::reset_log_callback();
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);

    logs.clear();

    ov::util::set_log_callback(log_cb);
    OV_ASSERT_NO_THROW(core.get_property(deviceName,
                                         ov::intel_npu::compiler_version,
                                         {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}));
    ov::util::reset_log_callback();
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);

    logs.clear();
}

}  // namespace
