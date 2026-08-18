// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

namespace ov::test::behavior {

// Tests specific for RUNTIME_REQUIREMENTS and COMPATIBILITY_CHECK properties
class ClassCompatibilityStringTestNPU : public OVCompiledModelPropertiesBase,
                                        public ::testing::WithParamInterface<std::string> {
protected:
    std::string deviceName;
    ov::Core core;

private:
    std::optional<utils::LoggerLevelGuard> _logGuard;

public:
    void SetUp() override {
        _logGuard.emplace(::intel_npu::Logger::global().level());
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVCompiledModelPropertiesBase::SetUp();
        deviceName = GetParam();
    }

    void TearDown() override {
        _logGuard.reset();
        OVCompiledModelPropertiesBase::TearDown();
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
    auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());
    OV_ASSERT_NO_THROW(auto requirements = compiledModel.get_property(ov::runtime_requirements));

    OV_ASSERT_NO_THROW(compiledModel =
                           core.compile_model(model,
                                              deviceName,
                                              {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                                               ov::intel_npu::bypass_umd_caching(true)}));
    // Test that RUNTIME_REQUIREMENTS is supported for CID when the L0 graph extension version >= 1.16,
    // and unsupported for earlier driver versions. CIP always supports it.
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    const auto initStructs = ::intel_npu::ZeroInitStructsHolder::getInstance();
    const bool driverHandlesCompatibilityCheck =
        initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16);
    if (driverHandlesCompatibilityCheck) {
        ASSERT_TRUE(it != properties.cend());
    } else {
        ASSERT_TRUE(it == properties.cend());
    }
    // #E224500 workaround until driver fix
    compiledModel = {};
}

TEST_P(ClassCompatibilityStringTestSuite, RuntimeRequirementsValueIsReadableWhenSupported) {
    auto model = ov::test::utils::make_conv_pool_relu();
    ov::CompiledModel compiledModel;

    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(
                           model,
                           deviceName,
                           {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                            ov::intel_npu::platform(ov::intel_npu::Platform::standardize(
                                ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)))}));

    std::vector<ov::PropertyName> properties;
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it != properties.cend());

    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiledModel.get_property(ov::runtime_requirements));
    ASSERT_FALSE(requirements.empty());

    OV_ASSERT_NO_THROW(compiledModel =
                           core.compile_model(model,
                                              deviceName,
                                              {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                                               ov::intel_npu::bypass_umd_caching(true)}));

    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    const auto initStructs = ::intel_npu::ZeroInitStructsHolder::getInstance();
    const bool driverHandlesCompatibilityCheck =
        initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16);
    if (driverHandlesCompatibilityCheck) {
        OV_ASSERT_NO_THROW(auto requirements = compiledModel.get_property(ov::runtime_requirements));
    } else {
        OV_EXPECT_THROW(auto requirements = compiledModel.get_property(ov::runtime_requirements),
                        ov::Exception,
                        testing::HasSubstr("Unsupported configuration key: RUNTIME_REQUIREMENTS"));
    }
    // #E224500 workaround until driver fix
    compiledModel = {};
}

TEST_P(ClassCompatibilityStringTestSuite, RuntimeRequirementsIsSupportedForWS) {
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
                            ov::enable_weightless(true),
                            ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ONE_SHOT)}));

    std::vector<ov::PropertyName> properties;
    // Test that RUNTIME_REQUIREMENTS is supported for a weightless model
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());

    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiledModel.get_property(ov::runtime_requirements));
    ASSERT_FALSE(requirements.empty());
}

TEST_P(ClassCompatibilityStringTestSuite, RuntimeRequirementsIsSupportedForWSIterative) {
    // Preparing the model for the test
    std::stringstream model_xml, model_bin;
    {
        // Serialize generated model into stringstream to later populate `WeightlessCacheAttribute` runtime
        // information of constant nodes
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
                            ov::enable_weightless(true),
                            ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE)}));

    std::vector<ov::PropertyName> properties;
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it != properties.cend());

    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiledModel.get_property(ov::runtime_requirements));
    ASSERT_FALSE(requirements.empty());
}

TEST_P(ClassCompatibilityStringTestSuite, RuntimeRequirementsExportImportForWSIterative) {
    // Preparing the model for the test
    std::stringstream model_xml, model_bin;
    {
        // Serialize generated model into stringstream to later populate `WeightlessCacheAttribute` runtime
        // information of constant nodes
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
                            ov::enable_weightless(true),
                            ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE)}));

    std::vector<ov::PropertyName> properties;
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it != properties.cend());

    std::string reference_requirements;
    OV_ASSERT_NO_THROW(reference_requirements = compiledModel.get_property(ov::runtime_requirements));
    ASSERT_FALSE(reference_requirements.empty());

    std::stringstream compiled_blob;
    OV_ASSERT_NO_THROW(compiledModel.export_model(compiled_blob));

    OV_ASSERT_NO_THROW(compiledModel = {});
    // A weightless blob does not embed the weights, so the original model must be provided on import
    OV_ASSERT_NO_THROW(compiledModel = core.import_model(compiled_blob, deviceName, {ov::hint::model(model)}));

    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it != properties.cend());

    std::string imported_requirements;
    OV_ASSERT_NO_THROW(imported_requirements = compiledModel.get_property(ov::runtime_requirements));
    ASSERT_FALSE(imported_requirements.empty());

    // The equality must be guaranteed for a given openvino version
    // If the blob was exported with a different OV version, requirements might differ
    ASSERT_EQ(reference_requirements, imported_requirements);
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

TEST_P(ClassCompatibilityStringTestSuite, RuntimeRequirementsExportImportForWSIterativeCID) {
    // The compatibility descriptor on the compiler-in-driver path is only available when the driver
    // graph API version is high enough. On older drivers it is expected to be absent.
    const auto initStructs = ::intel_npu::ZeroInitStructsHolder::getInstance();
    const bool driverHandlesCompatibilityCheck =
        initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16);

    // Preparing the model for the test
    std::stringstream model_xml, model_bin;
    {
        // Serialize generated model into stringstream to later populate `WeightlessCacheAttribute` runtime
        // information of constant nodes
        auto model = ov::test::utils::make_conv_pool_relu();
        ov::pass::Serialize serializer(model_xml, model_bin);
        serializer.run_on_model(model);
    }
    auto model_bin_str = model_bin.str();
    ov::Tensor model_weights(ov::element::u8, ov::Shape{model_bin_str.size()});
    std::memcpy(model_weights.data<char>(), model_bin_str.data(), model_bin_str.size());
    auto model = core.read_model(model_xml.str(), model_weights);

    // Forcing CID as the current compiler type
    // Check for the property in the supported properties list, since the driver may not support it.
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(supportedProperties =
                           core.get_property(deviceName,
                                             ov::supported_properties,
                                             {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}));
    if (find(supportedProperties.cbegin(), supportedProperties.cend(), ov::enable_weightless.name()) ==
        supportedProperties.cend()) {
        GTEST_SKIP() << "current driver does not support ENABLE_WEIGHTLESS";
    }

    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(
                           model,
                           deviceName,
                           {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                            ov::intel_npu::bypass_umd_caching(true),
                            ov::enable_weightless(true),
                            ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE)}));

    std::vector<ov::PropertyName> properties;
    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    auto it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    if (!driverHandlesCompatibilityCheck) {
        ASSERT_TRUE(it == properties.cend());
        return;
    }
    ASSERT_TRUE(it != properties.cend());

    std::string reference_requirements;
    OV_ASSERT_NO_THROW(reference_requirements = compiledModel.get_property(ov::runtime_requirements));
    ASSERT_FALSE(reference_requirements.empty());

    std::stringstream compiled_blob;
    OV_ASSERT_NO_THROW(compiledModel.export_model(compiled_blob));

    OV_ASSERT_NO_THROW(compiledModel = {});
    // A weightless blob does not embed the weights, so the original model must be provided on import
    OV_ASSERT_NO_THROW(compiledModel = core.import_model(compiled_blob, deviceName, {ov::hint::model(model)}));

    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));
    it = find(properties.cbegin(), properties.cend(), ov::runtime_requirements);
    ASSERT_TRUE(it != properties.cend());

    std::string imported_requirements;
    OV_ASSERT_NO_THROW(imported_requirements = compiledModel.get_property(ov::runtime_requirements));
    ASSERT_FALSE(imported_requirements.empty());

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

    if (driverHandlesCompatibilityCheck) {
        ASSERT_EQ(logs.find("Option COMPATIBILITY_CHECK with value `null` is supported by PluginCompilerAdapter"),
                  std::string::npos);
    } else {
        ASSERT_NE(logs.find("Option COMPATIBILITY_CHECK with value `null` is supported by PluginCompilerAdapter"),
                  std::string::npos);
    }

    // DriverCompilerAdapter should not be initialized at all in such a scenario, checking that the corresponding
    // message is not present in the log
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
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

    if (driverHandlesCompatibilityCheck) {
        ASSERT_EQ(logs.find("Option COMPATIBILITY_CHECK with value `null` is supported by PluginCompilerAdapter"),
                  std::string::npos);
    } else {
        ASSERT_NE(logs.find("Option COMPATIBILITY_CHECK with value `null` is supported by PluginCompilerAdapter"),
                  std::string::npos);
    }

    // DriverCompilerAdapter should not be initialized at all in such a scenario, checking that the corresponding
    // message is not present in the log
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
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
        ASSERT_EQ(logs.find("Option COMPATIBILITY_CHECK with value `null` is supported by PluginCompilerAdapter"),
                  std::string::npos);
    } else {
        ASSERT_NE(logs.find("Option COMPATIBILITY_CHECK with value `null` is supported by PluginCompilerAdapter"),
                  std::string::npos);
    }
}

TEST_P(CompatibilityCheckFallbackTestSuite, CompatibilityCheckAcceptsEmptyString) {
    // Empty descriptor means that there are no runtime requirements
    // No E2E test reaches this branch because compilation never produces an empty descriptor
    ov::CompatibilityCheck result = ov::CompatibilityCheck::SUPPORTED;
    OV_ASSERT_NO_THROW(result = core.get_property(deviceName,
                                                  ov::compatibility_check,
                                                  std::make_pair(ov::runtime_requirements.name(), "")));
    ASSERT_EQ(result, ov::CompatibilityCheck::NOT_APPLICABLE);
}

}  // namespace ov::test::behavior

using namespace ov::test::behavior;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         ClassCompatibilityStringTestSuite,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ClassCompatibilityStringTestSuite::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorCompatibilityFallbackTests,
                         CompatibilityCheckFallbackTestSuite,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         CompatibilityCheckFallbackTestSuite::getTestCaseName);
