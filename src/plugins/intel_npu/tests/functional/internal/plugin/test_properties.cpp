// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "compiler_option_support_helper.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/log.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "plugin_property_manager.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_backend.hpp"

using ::testing::AllOf;
using ::testing::HasSubstr;

using ConfigParams = std::tuple<std::string,   // Device name
                                std::string>;  // Config name

namespace ov {
namespace test {
namespace behavior {
class PropertiesManagerTests : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<ConfigParams> {
protected:
    ov::SoPtr<::intel_npu::IEngineBackend> backend;
    std::unique_ptr<::intel_npu::PluginPropertyManager> propertiesManager;

    std::string configuration;
    std::string targetDevice;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConfigParams>& obj) {
        std::string targetDevice;
        std::string configuration;
        std::tie(targetDevice, configuration) = obj.param;

        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "config=" << configuration << "_";

        return result.str();
    }

    void SetUp() override {
        using namespace ::intel_npu;

        std::tie(targetDevice, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();

        backend = ov::SoPtr<IEngineBackend>(std::make_shared<ZeroEngineBackend>());

        propertiesManager = std::make_unique<PluginPropertyManager>(
            backend,
            std::make_shared<::intel_npu::CompilerOptionSupportHelper>(backend, CompilerAdapterFactory()),
            ::intel_npu::Logger::global());
    }

    void TearDown() override {
        OVPluginTestBase::TearDown();
    }
};

TEST_P(PropertiesManagerTests, ExpectRunTimeSpecialBothPropertyIsSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool isSupported = false;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::log::level(ov::log::Level::INFO)}});
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported(configuration);
    }

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    ASSERT_TRUE(isSupported);
}

using CompileLogLevelPropertyTests = PropertiesManagerTests;

TEST_P(CompileLogLevelPropertyTests, InheritsPluginLogLevelWhenUnset) {
    propertiesManager->setProperty({{ov::log::level(ov::log::Level::DEBUG)}});

    ov::Any retrieved;
    OV_ASSERT_NO_THROW(retrieved = propertiesManager->getProperty(ov::intel_npu::compile_log_level.name()));
    ASSERT_EQ(retrieved.as<ov::log::Level>(), ov::log::Level::DEBUG);
}

TEST_P(CompileLogLevelPropertyTests, IsIndependentFromLogLevel) {
    propertiesManager->setProperty(
        {{ov::log::level(ov::log::Level::DEBUG)}, {ov::intel_npu::compile_log_level(ov::log::Level::ERR)}});

    ov::Any compilerLevel;
    OV_ASSERT_NO_THROW(compilerLevel = propertiesManager->getProperty(ov::intel_npu::compile_log_level.name()));
    ASSERT_EQ(compilerLevel.as<ov::log::Level>(), ov::log::Level::ERR);

    ov::Any pluginLevel;
    OV_ASSERT_NO_THROW(pluginLevel = propertiesManager->getProperty(ov::log::level.name()));
    ASSERT_EQ(pluginLevel.as<ov::log::Level>(), ov::log::Level::DEBUG);
}

using CompatibilityCheckTests = PropertiesManagerTests;

TEST_P(CompatibilityCheckTests, ExpectArgumentIsNotSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool isSupported = true;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    ov::AnyMap arguments = {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                            {"DUMMY_PROPERTY", "DUMMY_VALUE"}};

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);

        try {
            propertiesManager->setProperty(arguments);
            isSupported = true;
        } catch (...) {
            isSupported = false;
        }
    }

    ASSERT_FALSE(isSupported);
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CompatibilityCheckUsesPluginCompilerAdapterOnlyWhenDriverVersionIsInsufficient) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    // Determine at runtime whether the driver version is sufficient to handle the
    // compatibility check without falling back to PluginCompilerAdapter.
    const auto initStructs = backend ? backend->getInitStructs() : nullptr;
    const bool driverHandlesCompatibilityCheck =
        initStructs != nullptr && initStructs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 16);

    bool isSupported = false;
    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        isSupported = propertiesManager->isPropertySupported(ov::compatibility_check.name());
    }

    if (driverHandlesCompatibilityCheck) {
        // Driver version >= 1.16: Property must be reported as supported.
        ASSERT_EQ(logs.find("initialize PluginCompilerAdapter complete"), std::string::npos);
        ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
        ASSERT_TRUE(isSupported);
    } else {
        if (logs.find("initialize PluginCompilerAdapter complete") == std::string::npos) {
            // Driver version < 1.16: Because CiP can not be loaded on this path in CI, the property must be reported as
            // unsupported.
            ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
            ASSERT_FALSE(isSupported);
        } else {
            // Driver version < 1.16: Because CiP can be loaded on this path in CI, the property must be reported as
            // supported.
            ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
            ASSERT_TRUE(isSupported);
        }
    }
}

TEST_P(CompatibilityCheckTests, ExpectTurboPropertyAndCompatibilityCheckAreSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool turboSupported = false;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    const bool turboSupportedByDevice = backend && backend->isCommandQueueExtSupported();

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        turboSupported = propertiesManager->isPropertySupported(ov::intel_npu::turbo.name());
    }

    if (turboSupportedByDevice) {
        // Turbo is supported by device, so checking support must not trigger compiler adapters.
        ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
        ASSERT_TRUE(turboSupported);
    } else {
        ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
        ASSERT_FALSE(turboSupported);
    }
}

TEST_P(CompatibilityCheckTests, ExpectCompilerPropertyIsNotSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool isSupported = true;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported("DUMMY_PROPERTY");
    }

    ASSERT_FALSE(isSupported);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);

    logs.clear();

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}});
        isSupported = propertiesManager->isPropertySupported("DUMMY_PROPERTY");
    }

    ASSERT_FALSE(isSupported);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

using ExpectLoadingCompilerPropertySupported = PropertiesManagerTests;

TEST_P(ExpectLoadingCompilerPropertySupported, ExpectCompilerPropertyIsSupported) {
    std::string logs;
    std::mutex logs_mutex;
    bool isSupported = false;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported(configuration);
    }

    ASSERT_TRUE(isSupported);
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

using namespace ov::test::behavior;

const std::vector<std::string> supported_configs = {{ov::hint::performance_mode.name()},
                                                    {ov::cache_dir.name()},
                                                    {ov::intel_npu::driver_version.name()}};
const std::vector<std::string> supported_compiler_configs = {{ov::intel_npu::qdq_optimization.name()}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         PropertiesManagerTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(supported_configs)),
                         PropertiesManagerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         CompatibilityCheckTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::Values(std::string{})),
                         PropertiesManagerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ExpectLoadingCompilerPropertySupported,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(supported_compiler_configs)),
                         PropertiesManagerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         CompileLogLevelPropertyTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::Values(std::string{})),
                         PropertiesManagerTests::getTestCaseName);
