// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <random>
#include <thread>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
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

template <typename OptionType>
void register_option(::intel_npu::OptionsDesc& options, ::intel_npu::FilteredConfig& config) {
    options.add<OptionType>();
    const auto option_mode = OptionType::mode();
    const bool is_enabled_by_default = option_mode == OptionMode::RunTime || option_mode == OptionMode::Both;
    config.enable(OptionType::key(), is_enabled_by_default);
}

template <typename... OptionTypes>
void register_options(::intel_npu::OptionsDesc& options, ::intel_npu::FilteredConfig& config) {
    (register_option<OptionTypes>(options, config), ...);
}

class PropertiesManagerTests : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<ConfigParams> {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::FilteredConfig npu_config = ::intel_npu::FilteredConfig(options);
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

        options->reset();
        register_options<LOG_LEVEL,
                         CACHE_DIR,
                         CACHE_MODE,
                         COMPILED_BLOB,
                         DEVICE_ID,
                         NUM_STREAMS,
                         PERF_COUNT,
                         LOADED_FROM_CACHE,
                         COMPILATION_NUM_THREADS,
                         PERFORMANCE_HINT,
                         EXECUTION_MODE_HINT,
                         PERFORMANCE_HINT_NUM_REQUESTS,
                         INFERENCE_PRECISION_HINT,
                         MODEL_PRIORITY,
                         COMPILATION_MODE_PARAMS,
                         DMA_ENGINES,
                         TILES,
                         COMPILATION_MODE,
                         COMPILER_TYPE,
                         COMPILER_VERSION,
                         PLATFORM,
                         CREATE_EXECUTOR,
                         DYNAMIC_SHAPE_TO_STATIC,
                         PROFILING_TYPE,
                         BACKEND_COMPILATION_PARAMS,
                         BATCH_MODE,
                         BYPASS_UMD_CACHING,
                         DEFER_WEIGHTS_LOAD,
                         WEIGHTS_PATH,
                         RUN_INFERENCES_SEQUENTIALLY,
                         COMPILER_DYNAMIC_QUANTIZATION,
                         QDQ_OPTIMIZATION,
                         QDQ_OPTIMIZATION_AGGRESSIVE,
                         STEPPING,
                         DISABLE_VERSION_CHECK,
                         EXPORT_RAW_BLOB,
                         IMPORT_RAW_BLOB,
                         BATCH_COMPILER_MODE_SETTINGS,
                         TURBO,
                         ENABLE_WEIGHTLESS,
                         SEPARATE_WEIGHTS_VERSION,
                         WS_COMPILE_CALL_NUMBER,
                         MODEL_SERIALIZER_VERSION,
                         ENABLE_STRIDES_FOR,
                         SHARED_COMMON_QUEUE,
                         CACHE_ENCRYPTION_CALLBACKS,
                         RUNTIME_REQUIREMENTS,
                         COMPATIBILITY_CHECK,
                         MAX_TILES,
                         WORKLOAD_TYPE,
                         DISABLE_IDLE_MEMORY_PRUNING>(*options, npu_config);

        OPENVINO_SUPPRESS_DEPRECATED_START
        register_option<ENABLE_CPU_PINNING>(*options, npu_config);
        OPENVINO_SUPPRESS_DEPRECATED_END

        // parse again env_variables to update registered configs which have env vars set
        npu_config.parseEnvVars();

        for_each_exposed_npuw_option([&](auto tag) {
            using Opt = typename decltype(tag)::type;
            register_option<Opt>(*options, npu_config);
        });

        // Special cases
        // Disable NPU_TURBO in case driver is not present or it does not support the extension.
        npu_config.enable(ov::intel_npu::turbo.name(), backend != nullptr && backend->isCommandQueueExtSupported());
        // Disable workload type in case driver is not present or it does not support the extension.
        npu_config.enable(ov::workload_type.name(), backend != nullptr && backend->isCommandQueueExtSupported());
        // Disable max tiles in case we don't have a device.
        npu_config.enable(ov::intel_npu::max_tiles.name(), backend != nullptr && backend->getDevice() != nullptr);
        // Disable idle memory pruning in case driver is not present or it does not support the extension.
        npu_config.enable(ov::intel_npu::disable_idle_memory_prunning.name(),
                          backend != nullptr && backend->isContextExtSupported());

        if (npu_config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::PREFER_PLUGIN && backend != nullptr) {
            auto device = backend->getDevice();
            if (device) {
                auto platformName = device->getName();
                CompilerAdapterFactory compilerFactory;
                auto compileType = compilerFactory.determineAppropriateCompilerTypeBasedOnPlatform(platformName);
                if (compileType == ov::intel_npu::CompilerType::DRIVER) {
                    npu_config.update({{ov::intel_npu::compiler_type.name(), COMPILER_TYPE::toString(compileType)}});
                }
            }
        }

        propertiesManager = std::make_unique<PluginPropertyManager>(npu_config, backend, ::intel_npu::Logger::global());
    }

    void TearDown() override {
        APIBaseTest::TearDown();
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
