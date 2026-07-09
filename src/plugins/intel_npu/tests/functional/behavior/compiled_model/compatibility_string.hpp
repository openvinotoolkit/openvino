// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <mutex>
#include <openvino/runtime/intel_npu/properties.hpp>

#include "behavior/compiled_model/properties.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/pass/serialize.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using namespace ov::test::behavior;

namespace {

// Tests specific for RUNTIME_REQUIREMENTS and COMPATIBILITY_CHECK properties
class ClassCompatibilityStringTestNPU : public OVCompiledModelPropertiesBase,
                                        public ::testing::WithParamInterface<std::string> {
protected:
    std::string deviceName;
    ov::Core core;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVCompiledModelPropertiesBase::SetUp();
        deviceName = GetParam();
    }
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        auto targetDevice = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        std::ostringstream result;
        static uint8_t testCounter = 0;
        result << "_testCounter="
               << std::to_string(testCounter++) + "_";  // used to avoid same names for different tests
        result << "targetDevice=" << ov::test::utils::getDeviceNameTestCase(targetDevice) << "_";
        result << "_targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
        return result.str();
    }
};

using ClassCompatibilityStringTestSuite = ClassCompatibilityStringTestNPU;

TEST_P(ClassCompatibilityStringTestSuite, CompatibilityCheckIsSupported) {
    std::vector<ov::PropertyName> properties;

    // Forcing CIP as the current compiler type
    core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN));

    {
        OV_ASSERT_NO_THROW(properties = core.get_property(deviceName, ov::supported_properties));
        auto it = find(properties.cbegin(), properties.cend(), ov::compatibility_check);
        ASSERT_TRUE(it != properties.cend());
        ASSERT_FALSE(it->is_mutable());
    }

    // Forcing CID as the current compiler type
    core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER));

    // Test that COMPATIBILITY_CHECK is still present in supported properties when CID is used as the current compiler
    // type Even if CID does not support the option, the property should be marked as supported since the plugin will
    // fallback to CIP
    {
        OV_ASSERT_NO_THROW(properties = core.get_property(deviceName, ov::supported_properties));
        auto it = find(properties.cbegin(), properties.cend(), ov::compatibility_check);
        ASSERT_TRUE(it != properties.cend());
    }
}

TEST_P(ClassCompatibilityStringTestSuite, CompatibilityCheckInvalidArgument) {
    // Forcing CIP as the current compiler type
    ov::CompatibilityCheck result = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(result = core.get_property(deviceName, ov::compatibility_check));
    ASSERT_TRUE(result == ov::CompatibilityCheck::NOT_APPLICABLE);

    // Provide an argument without runtime_requirements
    OV_ASSERT_NO_THROW(result =
                           core.get_property(deviceName, ov::compatibility_check, ov::log::level(ov::log::Level::ERR)));
    ASSERT_TRUE(result == ov::CompatibilityCheck::NOT_APPLICABLE);

    // An incorrect runtime_requirements argument should return UNSUPPORTED
    OV_ASSERT_NO_THROW(result = core.get_property(deviceName,
                                                  ov::compatibility_check,
                                                  std::make_pair(ov::runtime_requirements.name(), "invalid_string")));
    ASSERT_TRUE(result == ov::CompatibilityCheck::UNSUPPORTED);
}

TEST_P(ClassCompatibilityStringTestSuite, RuntimeRequirementsIsSupported) {
    // Forcing CIP as the current compiler type
    auto model = ov::test::utils::make_conv_pool_relu();
    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(
                           model,
                           deviceName,
                           {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                            ov::intel_npu::platform(ov::intel_npu::Platform::standardize(
                                ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)))}));

    std::vector<ov::PropertyName> properties;
    // Test that RUNTIME_REQUIREMENTS is supported for a model compiled with CIP
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    {
        auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
        ASSERT_TRUE(it != properties.cend());
        ASSERT_FALSE(it->is_mutable());
    }
    OV_ASSERT_NO_THROW(auto requirements = compiledModel.get_property(ov::runtime_requirements));

    OV_ASSERT_NO_THROW(
        compiledModel =
            core.compile_model(model, deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)));
    // Test that RUNTIME_REQUIREMENTS is not supported for a model compiled with CID
    // This check should be conditioned by the compiler/driver version once support is added in L0
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    {
        auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
        ASSERT_TRUE(it == properties.cend());
    }
    OV_EXPECT_THROW(auto requirements = compiledModel.get_property(ov::runtime_requirements),
                    ov::Exception,
                    testing::HasSubstr("Unsupported configuration key: RUNTIME_REQUIREMENTS"));
}

TEST_P(ClassCompatibilityStringTestSuite, RuntimeRequirementsIsNotSupportedForWS) {
    // Preparing the model for the test
    std::stringstream model_xml, model_bin;
    {
        // Serialize generated model into stringstream to later populate `WeightlessCacheAttribute` runtime information
        // of constant nodes
        auto model = ov::test::utils::make_conv_pool_relu();
        ov::pass::Serialize serializer(model_xml, model_bin);
        serializer.run_on_model(model);
    }
    auto model_bin_str = model_bin.str();
    ov::Tensor model_weights(ov::element::u8, ov::Shape{model_bin_str.size()});
    std::memcpy(model_weights.data<char>(), model_bin_str.data(), model_bin_str.size());
    auto model = core.read_model(model_xml.str(), model_weights);

    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(
                           model,
                           deviceName,
                           {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                            ov::intel_npu::platform(ov::intel_npu::Platform::standardize(
                                ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU))),
                            ov::enable_weightless(true)}));

    std::vector<ov::PropertyName> properties;
    // Test that RUNTIME_REQUIREMENTS is not supported for a weightless model
    OV_EXPECT_THROW(auto requirements = compiledModel.get_property(ov::runtime_requirements),
                    ov::Exception,
                    testing::HasSubstr("Unsupported configuration key: RUNTIME_REQUIREMENTS"));

    // Test that RUNTIME_REQUIREMENTS is not in the list of supported properties either
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it == properties.cend());
}

TEST_P(ClassCompatibilityStringTestSuite, RuntimeRequirementsExportImport) {
    // Forcing CIP as the current compiler type
    auto model = ov::test::utils::make_conv_pool_relu();
    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(
                           model,
                           deviceName,
                           {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                            ov::intel_npu::platform(ov::intel_npu::Platform::standardize(
                                ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)))}));
    std::string reference_requirements;
    OV_ASSERT_NO_THROW(reference_requirements = compiledModel.get_property(ov::runtime_requirements));

    std::stringstream compiled_blob;
    OV_ASSERT_NO_THROW(compiledModel.export_model(compiled_blob));

    OV_ASSERT_NO_THROW(compiledModel = {});
    OV_ASSERT_NO_THROW(compiledModel = core.import_model(compiled_blob, deviceName));

    std::vector<ov::PropertyName> properties;
    // Test that RUNTIME_REQUIREMENTS is supported for an imported model as well
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it != properties.cend());
    std::string imported_requirements;
    OV_ASSERT_NO_THROW(imported_requirements = compiledModel.get_property(ov::runtime_requirements));

    // The equality must be guaranteed for a given openvino version
    // If the blob was exported with a different OV version, requirements might differ
    ASSERT_EQ(reference_requirements, imported_requirements);
}

TEST_P(ClassCompatibilityStringTestSuite, CompatibilityStringGenerateAndCheck) {
    // Forcing CIP as the current compiler type
    auto model = ov::test::utils::make_conv_pool_relu();
    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(
                           model,
                           deviceName,
                           {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                            ov::intel_npu::platform(ov::intel_npu::Platform::standardize(
                                ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)))}));

    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiledModel.get_property(ov::runtime_requirements));
    ov::CompatibilityCheck result = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(result = core.get_property(deviceName,
                                                  ov::compatibility_check,
                                                  std::make_pair(ov::runtime_requirements.name(), requirements)));
    ASSERT_TRUE(result == ov::CompatibilityCheck::SUPPORTED);
}

using CompatibilityCheckFallbackTestSuite = ClassCompatibilityStringTestNPU;

TEST_P(CompatibilityCheckFallbackTestSuite, CompatibilityCheckIsReadOnly) {
    std::string logs;
    std::mutex logs_mutex;
    ov::AnyMap compatibilityCheckProperty = {{ov::compatibility_check.name(), ov::Any(ov::AnyMap{})}};

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    core.get_property(deviceName, ov::intel_npu::compiler_type);  // initialize plugin with runtime property

    OV_ASSERT_NO_THROW(
        core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)));

    // Determine at runtime whether the driver version is sufficient to handle the
    // compatibility check without falling back to PluginCompilerAdapter.
    const auto initStructs = ::intel_npu::ZeroInitStructsHolder::getInstance();
    const bool driverHandlesCompatibilityCheck =
        initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16);

    auto original_level = core.get_property(deviceName, ov::log::level);
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::log::level(ov::log::Level::INFO)));
    {
        ov::test::utils::LogCallbackGuard log_callback_guard(log_cb);
        OV_EXPECT_THROW_HAS_SUBSTRING(core.set_property(deviceName, compatibilityCheckProperty),
                                      ov::Exception,
                                      "READ-ONLY");
    }
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::log::level(original_level)));

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);

    if (driverHandlesCompatibilityCheck) {
        ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    } else {
        ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    }
}

TEST_P(CompatibilityCheckFallbackTestSuite, CompatibilityCheckUsesPluginCompilerFallbackForOlderDriver) {
    std::string logs;
    std::mutex logs_mutex;

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    OV_ASSERT_NO_THROW(
        core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)));

    // Determine at runtime whether the driver version is sufficient to handle the
    // compatibility check without falling back to PluginCompilerAdapter.
    const auto initStructs = ::intel_npu::ZeroInitStructsHolder::getInstance();
    const bool driverHandlesCompatibilityCheck =
        initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16);

    auto original_level = core.get_property(deviceName, ov::log::level);
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::log::level(ov::log::Level::INFO)));
    {
        ov::test::utils::LogCallbackGuard log_callback_guard(log_cb);
        OV_ASSERT_NO_THROW((void)core.get_property(deviceName, ov::compatibility_check));
    }
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::log::level(original_level)));

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);

    if (driverHandlesCompatibilityCheck) {
        ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    } else {
        ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    }
}

TEST_P(CompatibilityCheckFallbackTestSuite, CompatibilityCheckSupportedPropertiesLoadsPluginCompiler) {
    std::string logs;
    std::mutex logs_mutex;

    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    OV_ASSERT_NO_THROW(
        core.set_property(deviceName, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)));

    // Determine at runtime whether the driver version is sufficient to handle the
    // compatibility check without falling back to PluginCompilerAdapter.
    const auto initStructs = ::intel_npu::ZeroInitStructsHolder::getInstance();
    const bool driverHandlesCompatibilityCheck =
        initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16);

    auto original_level = core.get_property(deviceName, ov::log::level);
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::log::level(ov::log::Level::INFO)));
    {
        ov::test::utils::LogCallbackGuard log_callback_guard(log_cb);
        auto supported_props = core.get_property(deviceName, ov::supported_properties);
        auto it = std::find(supported_props.begin(), supported_props.end(), ov::compatibility_check.name());
        ASSERT_NE(it, supported_props.end());
    }
    OV_ASSERT_NO_THROW(core.set_property(deviceName, ov::log::level(original_level)));

    if (driverHandlesCompatibilityCheck) {
        ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    } else {
        ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    }
}

}  // namespace
