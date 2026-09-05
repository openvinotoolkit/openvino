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
#include <optional>
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
#include "openvino/util/codec_xor.hpp"
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
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
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
#define REGISTER_OPTION(OPT_TYPE) \
    do {                          \
        options->add<OPT_TYPE>(); \
    } while (0)

        REGISTER_OPTION(LOG_LEVEL);
        REGISTER_OPTION(COMPILE_LOG_LEVEL);
        REGISTER_OPTION(CACHE_DIR);
        REGISTER_OPTION(CACHE_MODE);
        REGISTER_OPTION(COMPILED_BLOB);
        REGISTER_OPTION(DEVICE_ID);
        REGISTER_OPTION(NUM_STREAMS);
        REGISTER_OPTION(PERF_COUNT);
        REGISTER_OPTION(LOADED_FROM_CACHE);
        REGISTER_OPTION(COMPILATION_NUM_THREADS);
        REGISTER_OPTION(PERFORMANCE_HINT);
        REGISTER_OPTION(EXECUTION_MODE_HINT);
        REGISTER_OPTION(PERFORMANCE_HINT_NUM_REQUESTS);
        REGISTER_OPTION(INFERENCE_PRECISION_HINT);
        REGISTER_OPTION(COMPILATION_MODE_PARAMS);
        REGISTER_OPTION(DMA_ENGINES);
        REGISTER_OPTION(TILES);
        REGISTER_OPTION(COMPILATION_MODE);
        REGISTER_OPTION(COMPILER_TYPE);
        REGISTER_OPTION(COMPILER_VERSION);
        REGISTER_OPTION(PLATFORM);
        REGISTER_OPTION(CREATE_EXECUTOR);
        REGISTER_OPTION(DYNAMIC_SHAPE_TO_STATIC);
        REGISTER_OPTION(PROFILING_TYPE);
        REGISTER_OPTION(BACKEND_COMPILATION_PARAMS);
        REGISTER_OPTION(BATCH_MODE);
        REGISTER_OPTION(BYPASS_UMD_CACHING);
        REGISTER_OPTION(DEFER_WEIGHTS_LOAD);
        REGISTER_OPTION(WEIGHTS_PATH);
        REGISTER_OPTION(RUN_INFERENCES_SEQUENTIALLY);
        REGISTER_OPTION(COMPILER_DYNAMIC_QUANTIZATION);
        REGISTER_OPTION(QDQ_OPTIMIZATION);
        REGISTER_OPTION(QDQ_OPTIMIZATION_AGGRESSIVE);
        REGISTER_OPTION(STEPPING);
        REGISTER_OPTION(DISABLE_VERSION_CHECK);
        REGISTER_OPTION(EXPORT_RAW_BLOB);
        REGISTER_OPTION(IMPORT_RAW_BLOB);
        REGISTER_OPTION(BATCH_COMPILER_MODE_SETTINGS);
        REGISTER_OPTION(TURBO);
        REGISTER_OPTION(ENABLE_WEIGHTLESS);
        REGISTER_OPTION(SEPARATE_WEIGHTS_VERSION);
        REGISTER_OPTION(WS_COMPILE_CALL_NUMBER);
        REGISTER_OPTION(MODEL_SERIALIZER_VERSION);
        REGISTER_OPTION(ENABLE_STRIDES_FOR);
        REGISTER_OPTION(SHARED_COMMON_QUEUE);
        REGISTER_OPTION(CACHE_ENCRYPTION_CALLBACKS);
        REGISTER_OPTION(MAX_TILES);
        REGISTER_OPTION(MODEL_PTR);
        REGISTER_OPTION(DISABLE_IDLE_MEMORY_PRUNING);

        if (backend) {
            REGISTER_OPTION(MODEL_PRIORITY);

            if (backend->isCommandQueueExtSupported()) {
                REGISTER_OPTION(WORKLOAD_TYPE);
            }
        }

        OPENVINO_SUPPRESS_DEPRECATED_START
        REGISTER_OPTION(ENABLE_CPU_PINNING);
        OPENVINO_SUPPRESS_DEPRECATED_END

        // NPUW properties are requested by OV Core during caching and
        // have no effect on the NPU plugin. But we still need to enable
        // those for OV Core to query. Note: do this last to not filter
        // them out. register npuw caching properties
        for_each_exposed_npuw_option([&](auto tag) {
            using Opt = typename decltype(tag)::type;
            REGISTER_OPTION(Opt);
        });

        propertiesManager = std::make_unique<PluginPropertyManager>(
            options,
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

TEST_P(CompatibilityCheckTests, ExpectTurboPropertyIsSupported) {
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
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}});
        turboSupported = propertiesManager->isPropertySupported(ov::intel_npu::turbo.name());
    }

    if (turboSupportedByDevice) {
        // Turbo is supported by device, so checking support must not trigger compiler adapters.
        ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
        ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    } else {
        ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
        ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    }

    ASSERT_TRUE(turboSupported);
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

TEST_P(CompatibilityCheckTests, CheckTurboWithGetMergedConfigAndUnknownPropertiesOnImport) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    if (backend->isCommandQueueExtSupported()) {
        auto [filteredConfig, unknownProperties] = [&]() {
            utils::LogCallbackGuard log_callback_guard(log_cb);
            utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
            return propertiesManager->getMergedConfigAndUnknownProperties({{ov::intel_npu::turbo(true)}},
                                                                          ::intel_npu::ConfigMergeMode::Import);
        }();

        ASSERT_TRUE(filteredConfig.has<::intel_npu::TURBO>());
        ASSERT_TRUE(filteredConfig.get<::intel_npu::TURBO>());
        ASSERT_TRUE(unknownProperties.empty());
    } else {
        {
            auto [filteredConfig, unknownProperties] = [&]() {
                utils::LogCallbackGuard log_callback_guard(log_cb);
                utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
                return propertiesManager->getMergedConfigAndUnknownProperties({{ov::intel_npu::turbo(true)}},
                                                                              ::intel_npu::ConfigMergeMode::Import);
            }();

            ASSERT_NE(logs.find("Property 'NPU_TURBO' is recognized as a compiler option, will not be used for current "
                                "configuration."),
                      std::string::npos);
            ASSERT_FALSE(filteredConfig.has<::intel_npu::TURBO>());
            ASSERT_TRUE(unknownProperties.empty());
        }
    }

    ASSERT_EQ(
        logs.find("Property 'NPU_COMPILER_TYPE' is used to specify the compiler type, will not be used for current "
                  "configuration."),
        std::string::npos);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckTurboWithGetMergedConfigAndUnknownPropertiesOnCompile) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}});

    auto [filteredConfig, unknownProperties] = [&]() {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        return propertiesManager->getMergedConfigAndUnknownProperties({{ov::intel_npu::turbo(true)}},
                                                                      ::intel_npu::ConfigMergeMode::Compile);
    }();

    ASSERT_TRUE(filteredConfig.has<::intel_npu::TURBO>());
    ASSERT_TRUE(filteredConfig.get<::intel_npu::TURBO>());
    ASSERT_TRUE(unknownProperties.empty());

    if (backend->isCommandQueueExtSupported()) {
        ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    } else {
        ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    }

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(
        logs.find("Property 'NPU_COMPILER_TYPE' is used to specify the compiler type, will not be used for current "
                  "configuration."),
        std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckCompilerPropertyWithGetMergedConfigAndUnknownPropertiesOnCompile) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto [filteredConfig, unknownProperties] = [&]() {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        return propertiesManager->getMergedConfigAndUnknownProperties(
            {{ov::intel_npu::platform("NPU3720"), ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}},
            ::intel_npu::ConfigMergeMode::Compile);
    }();

    ASSERT_EQ(
        logs.find("Property 'NPU_COMPILER_TYPE' is used to specify the compiler type, will not be used for current "
                  "configuration."),
        std::string::npos);
    ASSERT_TRUE(filteredConfig.has<::intel_npu::PLATFORM>());
    ASSERT_EQ(filteredConfig.get<::intel_npu::PLATFORM>(), "NPU3720");
    ASSERT_TRUE(unknownProperties.empty());

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckCompilerPropertyWithGetMergedConfigAndUnknownPropertiesOnImport) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto [filteredConfig, unknownProperties] = [&]() {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        return propertiesManager->getMergedConfigAndUnknownProperties(
            {{ov::intel_npu::platform("NPU3720"), ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}},
            ::intel_npu::ConfigMergeMode::Import);
    }();

    ASSERT_FALSE(filteredConfig.has<::intel_npu::COMPILER_TYPE>());
    ASSERT_FALSE(filteredConfig.has<::intel_npu::PLATFORM>());
    ASSERT_FALSE(filteredConfig.has<::intel_npu::COMPILER_TYPE>());
    ASSERT_TRUE(unknownProperties.empty());

    ASSERT_NE(
        logs.find("Property 'NPU_COMPILER_TYPE' is used to specify the compiler type, will not be used for current "
                  "configuration."),
        std::string::npos);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    ASSERT_NE(logs.find("Property 'NPU_PLATFORM' is recognized as a compiler option, will not be used for current "
                        "configuration."),
              std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckLoadedFromCacheWithGetMergedConfigAndUnknownPropertiesOnImport) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto [filteredConfig, unknownProperties] = [&]() {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        return propertiesManager->getMergedConfigAndUnknownProperties(
            {{{ov::loaded_from_cache.name(), ov::Any(true)},
              ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}},
            ::intel_npu::ConfigMergeMode::Import);
    }();

    ASSERT_FALSE(filteredConfig.has<::intel_npu::COMPILER_TYPE>());
    ASSERT_TRUE(filteredConfig.has<::intel_npu::LOADED_FROM_CACHE>());
    ASSERT_TRUE(filteredConfig.get<::intel_npu::LOADED_FROM_CACHE>());
    ASSERT_TRUE(unknownProperties.empty());

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckDummyPropertyWithGetMergedConfigAndUnknownPropertiesOnCompile) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto [filteredConfig, unknownProperties] = [&]() {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        return propertiesManager->getMergedConfigAndUnknownProperties(
            {{{"Dummy_Property", ov::Any(true)}, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}},
            ::intel_npu::ConfigMergeMode::Compile);
    }();

    ASSERT_TRUE(filteredConfig.has<::intel_npu::COMPILER_TYPE>());
    ASSERT_EQ(filteredConfig.get<::intel_npu::COMPILER_TYPE>(), ov::intel_npu::CompilerType::PLUGIN);
    ASSERT_FALSE(unknownProperties.empty());
    ASSERT_EQ(unknownProperties.size(), 1);

    ASSERT_EQ(
        logs.find("Property 'NPU_COMPILER_TYPE' is used to specify the compiler type, will not be used for current "
                  "configuration."),
        std::string::npos);
    ASSERT_NE(
        logs.find(
            "Property 'Dummy_Property' is unknown to the plugin property manager, will be sent to the compiled model."),
        std::string::npos);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckDummyPropertyWithGetMergedConfigAndUnknownPropertiesOnImport) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto [filteredConfig, unknownProperties] = [&]() {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        return propertiesManager->getMergedConfigAndUnknownProperties(
            {{{"Dummy_Property", ov::Any(true)}, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}},
            ::intel_npu::ConfigMergeMode::Import);
    }();

    ASSERT_FALSE(filteredConfig.has<::intel_npu::COMPILER_TYPE>());
    ASSERT_FALSE(unknownProperties.empty());
    ASSERT_EQ(unknownProperties.size(), 1);

    ASSERT_NE(
        logs.find("Property 'NPU_COMPILER_TYPE' is used to specify the compiler type, will not be used for current "
                  "configuration."),
        std::string::npos);
    ASSERT_NE(
        logs.find(
            "Property 'Dummy_Property' is unknown to the plugin property manager, will be sent to the compiled model."),
        std::string::npos);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckDummyPropertyWithGetMergedConfigAndUnknownPropertiesOnQuery) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);

        OV_EXPECT_THROW(propertiesManager->getMergedConfigAndUnknownProperties(
                            {{{"Dummy_Property", ov::Any(true)},
                              ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}},
                            ::intel_npu::ConfigMergeMode::Query),
                        ov::Exception,
                        testing::HasSubstr("Unsupported configuration key: Dummy_Property"));
    }

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_NE(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckROPropertyWithGetMergedConfigAndUnknownProperties) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);

        OV_EXPECT_THROW(propertiesManager->getMergedConfigAndUnknownProperties(
                            {{{ov::intel_npu::compiler_version.name(), ov::Any(true)}}},
                            ::intel_npu::ConfigMergeMode::Compile),
                        ov::Exception,
                        testing::HasSubstr("READ-ONLY configuration key: NPU_COMPILER_VERSION"));
    }

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckUnsupportedConfigWithGetMergedConfigAndUnknownProperties) {
    std::string logs;
    std::mutex logs_mutex;

    // Keep this std::function alive while logging is active.
    std::function<void(std::string_view)> log_cb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logs_mutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    std::shared_ptr<::intel_npu::OptionsDesc> localOptions = std::make_shared<::intel_npu::OptionsDesc>();
    localOptions->add<::intel_npu::COMPILER_TYPE>();
    auto localPropertiesManager = std::make_unique<::intel_npu::PluginPropertyManager>(
        localOptions,
        backend,
        std::make_shared<::intel_npu::CompilerOptionSupportHelper>(backend, ::intel_npu::CompilerAdapterFactory()),
        ::intel_npu::Logger::global());

    {
        utils::LogCallbackGuard log_callback_guard(log_cb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);

        OV_EXPECT_THROW(
            localPropertiesManager->getMergedConfigAndUnknownProperties({{ov::device::id("3720")}},
                                                                        ::intel_npu::ConfigMergeMode::Compile),
            ov::Exception,
            testing::HasSubstr("[ NOT_FOUND ] Option 'DEVICE_ID' is not supported for current configuration"));
    }

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

TEST_P(CompatibilityCheckTests, CheckModelPtrWithGetMergedConfigAndUnknownProperties) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto result = std::make_shared<ov::op::v0::Result>(param);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    auto constModel = std::const_pointer_cast<const ov::Model>(model);

    std::optional<::intel_npu::FilteredConfig> filteredConfig;
    for (const auto& modelProperty : {ov::hint::model(model), ov::hint::model(constModel)}) {
        auto [config, unknownProperties] =
            propertiesManager->getMergedConfigAndUnknownProperties({{modelProperty}},
                                                                   ::intel_npu::ConfigMergeMode::Import);
        filteredConfig = std::move(config);

        ASSERT_TRUE(filteredConfig->has<::intel_npu::MODEL_PTR>());
        ASSERT_EQ(filteredConfig->get<::intel_npu::MODEL_PTR>().lock(), model);
        ASSERT_TRUE(unknownProperties.empty());
    }

    model.reset();
    constModel.reset();
    ASSERT_EQ(filteredConfig->get<::intel_npu::MODEL_PTR>().lock(), nullptr);
}

TEST_P(CompatibilityCheckTests, CheckCacheEncryptionCallbacksWithGetMergedConfigAndUnknownProperties) {
    auto [filteredConfig, unknownProperties] = [&]() {
        return propertiesManager->getMergedConfigAndUnknownProperties(
            {{ov::cache_encryption_callbacks(ov::EncryptionCallbacks{ov::util::codec_xor, nullptr})}},
            ::intel_npu::ConfigMergeMode::Import);
    }();

    ASSERT_TRUE(filteredConfig.has<::intel_npu::CACHE_ENCRYPTION_CALLBACKS>());
    ASSERT_TRUE(unknownProperties.empty());
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
