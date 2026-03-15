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
#include "metrics.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/log.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_backend.hpp"

using ::testing::AllOf;
using ::testing::HasSubstr;

using ConfigParams = std::tuple<std::string,   // Device name
                                std::string>;  // Config name

namespace {
class LogCallbackGuard {
public:
    explicit LogCallbackGuard(const std::function<void(std::string_view)>& callback) {
        ov::util::set_log_callback(callback);
    }

    ~LogCallbackGuard() {
        ov::util::reset_log_callback();
    }

    LogCallbackGuard(const LogCallbackGuard&) = delete;
    LogCallbackGuard& operator=(const LogCallbackGuard&) = delete;
};

class LoggerLevelGuard {
public:
    explicit LoggerLevelGuard(ov::log::Level level) : _previousLevel(::intel_npu::Logger::global().level()) {
        ::intel_npu::Logger::global().setLevel(level);
    }

    ~LoggerLevelGuard() {
        ::intel_npu::Logger::global().setLevel(_previousLevel);
    }

    LoggerLevelGuard(const LoggerLevelGuard&) = delete;
    LoggerLevelGuard& operator=(const LoggerLevelGuard&) = delete;

private:
    ov::log::Level _previousLevel;
};
}  // namespace

namespace ov {
namespace test {
namespace behavior {
class PropertiesManagerTests : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<ConfigParams> {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::FilteredConfig npu_config = ::intel_npu::FilteredConfig(options);
    ov::SoPtr<::intel_npu::IEngineBackend> backend;
    std::unique_ptr<::intel_npu::Properties> propertiesManager;

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
        auto metrics = std::make_shared<Metrics>(backend);

        options->reset();

#define REGISTER_OPTION(OPT_TYPE)                             \
    do {                                                      \
        auto dummyopt = details::makeOptionModel<OPT_TYPE>(); \
        std::string o_name = dummyopt.key().data();           \
        options->add<OPT_TYPE>();                             \
        npu_config.enable(std::move(o_name), false);          \
    } while (0)

        REGISTER_OPTION(LOG_LEVEL);
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
        REGISTER_OPTION(ENABLE_CPU_PINNING);
        REGISTER_OPTION(INFERENCE_PRECISION_HINT);
        REGISTER_OPTION(MODEL_PRIORITY);
        REGISTER_OPTION(EXCLUSIVE_ASYNC_REQUESTS);
        REGISTER_OPTION(COMPILATION_MODE_PARAMS);
        REGISTER_OPTION(DMA_ENGINES);
        REGISTER_OPTION(TILES);
        REGISTER_OPTION(COMPILATION_MODE);
        REGISTER_OPTION(COMPILER_TYPE);
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
        REGISTER_OPTION(MAX_TILES);
        REGISTER_OPTION(DISABLE_VERSION_CHECK);
        REGISTER_OPTION(EXPORT_RAW_BLOB);
        REGISTER_OPTION(IMPORT_RAW_BLOB);
        REGISTER_OPTION(BATCH_COMPILER_MODE_SETTINGS);
        REGISTER_OPTION(TURBO);
        REGISTER_OPTION(WEIGHTLESS_BLOB);
        REGISTER_OPTION(SEPARATE_WEIGHTS_VERSION);
        REGISTER_OPTION(WS_COMPILE_CALL_NUMBER);
        REGISTER_OPTION(USE_BASE_MODEL_SERIALIZER);
        REGISTER_OPTION(MODEL_SERIALIZER_VERSION);
        REGISTER_OPTION(ENABLE_STRIDES_FOR);

        if (backend) {
            if (backend->isCommandQueueExtSupported()) {
                REGISTER_OPTION(WORKLOAD_TYPE);
            }
            if (backend->isContextExtSupported()) {
                REGISTER_OPTION(DISABLE_IDLE_MEMORY_PRUNING);
            }
        }

        REGISTER_OPTION(NPU_USE_NPUW);
        REGISTER_OPTION(NPUW_DEVICES);
        REGISTER_OPTION(NPUW_SUBMODEL_DEVICE);
        REGISTER_OPTION(NPUW_WEIGHTS_BANK);
        REGISTER_OPTION(NPUW_WEIGHTS_BANK_ALLOC);
        REGISTER_OPTION(NPUW_ONLINE_PIPELINE);
        REGISTER_OPTION(NPUW_ONLINE_AVOID);
        REGISTER_OPTION(NPUW_ONLINE_ISOLATE);
        REGISTER_OPTION(NPUW_ONLINE_NO_FOLD);
        REGISTER_OPTION(NPUW_ONLINE_MIN_SIZE);
        REGISTER_OPTION(NPUW_ONLINE_KEEP_BLOCKS);
        REGISTER_OPTION(NPUW_ONLINE_KEEP_BLOCK_SIZE);
        REGISTER_OPTION(NPUW_ATTN);
        REGISTER_OPTION(NPUW_ATTN_HFA_FUSED);
        REGISTER_OPTION(NPUW_FOLD);
        REGISTER_OPTION(NPUW_CWAI);
        REGISTER_OPTION(NPUW_DQ);
        REGISTER_OPTION(NPUW_DQ_FULL);
        REGISTER_OPTION(NPUW_PMM);
        REGISTER_OPTION(NPUW_SLICE_OUT);
        REGISTER_OPTION(NPUW_SPATIAL);
        REGISTER_OPTION(NPUW_SPATIAL_NWAY);
        REGISTER_OPTION(NPUW_SPATIAL_DYN);
        REGISTER_OPTION(NPUW_F16IC);
        REGISTER_OPTION(NPUW_HOST_GATHER);
        REGISTER_OPTION(NPUW_DCOFF_TYPE);
        REGISTER_OPTION(NPUW_DCOFF_SCALE);
        REGISTER_OPTION(NPUW_FUNCALL_FOR_ALL);
        REGISTER_OPTION(NPUW_FUNCALL_ASYNC);
        REGISTER_OPTION(NPUW_UNFOLD_IREQS);
        REGISTER_OPTION(NPUW_FALLBACK_EXEC);
        REGISTER_OPTION(NPUW_LLM);
        REGISTER_OPTION(NPUW_LLM_BATCH_DIM);
        REGISTER_OPTION(NPUW_LLM_SEQ_LEN_DIM);
        REGISTER_OPTION(NPUW_LLM_MAX_PROMPT_LEN);
        REGISTER_OPTION(NPUW_LLM_MAX_GENERATION_TOKEN_LEN);
        REGISTER_OPTION(NPUW_LLM_MIN_RESPONSE_LEN);
        REGISTER_OPTION(NPUW_LLM_OPTIMIZE_V_TENSORS);
        REGISTER_OPTION(NPUW_LLM_OPTIMIZE_FP8);
        REGISTER_OPTION(NPUW_LLM_CACHE_ROPE);
        REGISTER_OPTION(NPUW_LLM_PREFILL_MOE_HINT);
        REGISTER_OPTION(NPUW_LLM_GENERATE_MOE_HINT);
        REGISTER_OPTION(NPUW_LLM_GENERATE_PYRAMID);
        REGISTER_OPTION(NPUW_LLM_PREFILL_CHUNK_SIZE);
        REGISTER_OPTION(NPUW_LLM_SHARED_HEAD);
        REGISTER_OPTION(NPUW_LLM_MAX_LORA_RANK);
        REGISTER_OPTION(NPUW_LLM_ENABLE_PREFIX_CACHING);
        REGISTER_OPTION(NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE);
        REGISTER_OPTION(NPUW_LLM_PREFIX_CACHING_MAX_NUM_BLOCKS);
        REGISTER_OPTION(NPUW_WHISPER);
        REGISTER_OPTION(NPUW_WHISPER_EOS_TOKEN);
        REGISTER_OPTION(NPUW_EAGLE);
        REGISTER_OPTION(NPUW_TEXT_EMBED);
        REGISTER_OPTION(NPUW_LLM_PREFILL_HINT);
        REGISTER_OPTION(NPUW_LLM_PREFILL_CONFIG);
        REGISTER_OPTION(NPUW_LLM_ADDITIONAL_PREFILL_CONFIG);
        REGISTER_OPTION(NPUW_LLM_PREFILL_ATTENTION_HINT);
        REGISTER_OPTION(NPUW_LLM_GENERATE_HINT);
        REGISTER_OPTION(NPUW_LLM_GENERATE_CONFIG);
        REGISTER_OPTION(NPUW_LLM_ADDITIONAL_GENERATE_CONFIG);
        REGISTER_OPTION(NPUW_LLM_GENERATE_ATTENTION_HINT);
        REGISTER_OPTION(NPUW_LLM_SHARED_LM_HEAD_CONFIG);
        REGISTER_OPTION(NPUW_LLM_ADDITIONAL_SHARED_LM_HEAD_CONFIG);
        REGISTER_OPTION(NPUW_KOKORO);
        REGISTER_OPTION(NPUW_KOKORO_BLOCK_SIZE);
        REGISTER_OPTION(NPUW_KOKORO_OVERLAP_SIZE);
        REGISTER_OPTION(NPUW_MOE_TOKEN_CHUNK_SIZE);
        REGISTER_OPTION(NPUW_MOE_POOL_SIZE);

        npu_config.enableRuntimeOptions();

        // Special cases - options with OptionMode::Both must be enabled for the plugin even if the compiler does not
        // support them, because they may be used by the plugin itself or by the driver.
        // We still check compiler support to decide whether these options should be removed from the config string.

        // NPU_TURBO might be supported by the driver
        if (backend && backend->isCommandQueueExtSupported()) {
            npu_config.enable(ov::intel_npu::turbo.name(), true);
        }

        // LOG_LEVEL, PERFORMANCE_HINT and PERF_COUNT are needed by runtime options
        npu_config.enable(ov::log::level.name(), true);
        npu_config.enable(ov::hint::performance_mode.name(), true);
        npu_config.enable(ov::enable_profiling.name(), true);

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

        propertiesManager = std::make_unique<Properties>(PropertiesType::PLUGIN, npu_config, metrics, backend);
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
        LogCallbackGuard log_callback_guard(log_cb);
        LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::log::level(ov::log::Level::INFO)}});
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported(configuration);
    }

    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
    ASSERT_TRUE(isSupported);
}

TEST_P(PropertiesManagerTests, ExpectArgumentIsNotSupported) {
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
        LogCallbackGuard log_callback_guard(log_cb);
        LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);

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
        LogCallbackGuard log_callback_guard(log_cb);
        LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported(configuration);
    }

    ASSERT_TRUE(isSupported);
    ASSERT_NE(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

using ExpectLoadingCompilerPropertyNotSupported = PropertiesManagerTests;

TEST_P(ExpectLoadingCompilerPropertyNotSupported, ExpectCompilerPropertyIsNotSupported) {
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
        LogCallbackGuard log_callback_guard(log_cb);
        LoggerLevelGuard logger_level_guard(ov::log::Level::INFO);
        propertiesManager->setProperty({{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}});
        isSupported = propertiesManager->isPropertySupported(configuration);
    }

    ASSERT_FALSE(isSupported);
    ASSERT_EQ(logs.find("initialize DriverCompilerAdapter start"), std::string::npos);
    ASSERT_EQ(logs.find("initialize PluginCompilerAdapter start"), std::string::npos);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
