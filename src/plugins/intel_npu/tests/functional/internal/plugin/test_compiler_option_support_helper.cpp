// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "compiler_option_support_helper.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_backend.hpp"

namespace ov {
namespace test {
namespace behavior {

namespace {
// Log markers emitted once per adapter, when its constructor runs (see driver_compiler_adapter.cpp /
// plugin_compiler_adapter.cpp). Used to detect whether a given compiler type had to be (re)initialized.
std::string startMarkerFor(ov::intel_npu::CompilerType compilerType) {
    return compilerType == ov::intel_npu::CompilerType::DRIVER ? "initialize DriverCompilerAdapter start"
                                                               : "initialize PluginCompilerAdapter start";
}

size_t countOccurrences(const std::string& haystack, const std::string& needle) {
    size_t count = 0;
    for (size_t pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

// Only PluginCompilerAdapter's successful bulk load logs a distinctive, identifiable marker (see
// "VCLCompilerImpl return supported_options" in plugin_compiler_adapter.cpp). DriverCompilerAdapter has
// no equivalent marker on its success path, so "exactly once" bulk-load checks below only apply to PLUGIN.
std::string bulkLoadMarkerFor(ov::intel_npu::CompilerType compilerType) {
    return compilerType == ov::intel_npu::CompilerType::PLUGIN ? "VCLCompilerImpl return supported_options" : "";
}
}  // namespace

// Exercises CompilerOptionSupportHelper against a real backend/compiler, complementing the
// backend-agnostic unit tests in tests/unit/npu/compiler_option_support_helper_test.cpp.
class CompilerOptionSupportHelperFunctionalTests : public ov::test::behavior::OVPluginTestBase,
                                                   public testing::WithParamInterface<ov::intel_npu::CompilerType> {
protected:
    ov::SoPtr<::intel_npu::IEngineBackend> backend;
    std::unique_ptr<::intel_npu::CompilerOptionSupportHelper> helper;
    ov::intel_npu::CompilerType compilerType;

    std::string targetDevice;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_NPU;

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();

        compilerType = GetParam();
        backend = ov::SoPtr<::intel_npu::IEngineBackend>(std::make_shared<::intel_npu::ZeroEngineBackend>());
        helper =
            std::make_unique<::intel_npu::CompilerOptionSupportHelper>(backend, ::intel_npu::CompilerAdapterFactory());
    }

    void TearDown() override {
        OVPluginTestBase::TearDown();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::intel_npu::CompilerType>& obj) {
        std::ostringstream result;
        result << "compilerType=" << obj.param;
        return result.str();
    }
};

TEST_P(CompilerOptionSupportHelperFunctionalTests, RejectsCompilerTypeOutsideSupportedList) {
    EXPECT_THROW(helper->isOptionSupported(ov::intel_npu::CompilerType::PREFER_PLUGIN, "ANY_OPTION"), ov::Exception);
}

// Properties from DriverCompilerAdapter's legacy fallback list marked as supported since compiler
// version 0.0 (see "_supportedPropertiesWithVersions" in driver_compiler_adapter.cpp). These are safe,
// version-independent choices to exercise real (uncached) compiler lookups for both compiler types.
const std::vector<std::string> knownSupportedOptions = {ov::enable_profiling.name(),
                                                        ov::hint::performance_mode.name(),
                                                        ov::log::level.name(),
                                                        ov::intel_npu::platform.name()};

TEST_P(CompilerOptionSupportHelperFunctionalTests, GetsSupportFromCompilerWhenNotCached) {
    std::string logs;
    std::mutex logsMutex;

    std::function<void(std::string_view)> logCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logsMutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard logCallbackGuard(logCb);
        utils::LoggerLevelGuard loggerLevelGuard(ov::log::Level::INFO);

        for (const auto& optionName : knownSupportedOptions) {
            bool isSupported = false;
            OV_ASSERT_NO_THROW(isSupported = helper->isOptionSupported(compilerType, optionName));
            EXPECT_TRUE(isSupported) << "Option: " << optionName;
        }
    }

    // The cache was empty for these options, so the real compiler adapter must have been created.
    ASSERT_NE(logs.find(startMarkerFor(compilerType)), std::string::npos);
}

TEST_P(CompilerOptionSupportHelperFunctionalTests, CachesResultAfterRealCompilerLookup) {
    for (const auto& optionName : knownSupportedOptions) {
        OV_ASSERT_NO_THROW(helper->isOptionSupported(compilerType, optionName));
    }

    std::string logs;
    std::mutex logsMutex;
    std::function<void(std::string_view)> logCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logsMutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard logCallbackGuard(logCb);
        utils::LoggerLevelGuard loggerLevelGuard(ov::log::Level::INFO);

        // Second lookup for every option: each must be served from the cache populated by the
        // previous real compiler lookup, without creating the adapter again.
        for (const auto& optionName : knownSupportedOptions) {
            bool isSupported = false;
            OV_ASSERT_NO_THROW(isSupported = helper->isOptionSupported(compilerType, optionName));
            EXPECT_TRUE(isSupported) << "Option: " << optionName;
        }
    }

    ASSERT_EQ(logs.find(startMarkerFor(compilerType)), std::string::npos);
}

TEST_P(CompilerOptionSupportHelperFunctionalTests, AddsNewOptionToCacheAndReturnsItWithoutCompilerLookup) {
    const auto cacheKey = static_cast<::intel_npu::OptionSupportCache::CacheKey>(compilerType);
    helper->getOptionSupportCache()->addSupportedOption(cacheKey, "MANUALLY_CACHED_OPTION", true);

    std::string logs;
    std::mutex logsMutex;
    std::function<void(std::string_view)> logCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logsMutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    bool isSupported = false;
    {
        utils::LogCallbackGuard logCallbackGuard(logCb);
        utils::LoggerLevelGuard loggerLevelGuard(ov::log::Level::INFO);

        OV_ASSERT_NO_THROW(isSupported = helper->isOptionSupported(compilerType, "MANUALLY_CACHED_OPTION"));
    }

    EXPECT_TRUE(isSupported);
    // The answer came straight from the cache entry added above; no adapter should have been created.
    ASSERT_EQ(logs.find(startMarkerFor(compilerType)), std::string::npos);
}

// "DUMMY_OPTION_NOT_IN_BULK_LIST" is not part of any real compiler's bulk "get_supported_options()" dump,
// so unlike "knownSupportedOptions" above, the first lookup cannot be answered from the bulk-populated
// cache and must fall through to the individual "is_option_supported" probe on the real compiler adapter.
TEST_P(CompilerOptionSupportHelperFunctionalTests, ProbesCompilerDirectlyForOptionOutsideBulkList) {
    const std::string unlistedOption = "DUMMY_OPTION_NOT_IN_BULK_LIST";

    std::string firstCallLogs;
    std::mutex firstCallLogsMutex;
    std::function<void(std::string_view)> firstCallLogCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(firstCallLogsMutex);
        firstCallLogs.append(msg);
        firstCallLogs.push_back('\n');
    };

    bool firstResult = false;
    {
        utils::LogCallbackGuard logCallbackGuard(firstCallLogCb);
        utils::LoggerLevelGuard loggerLevelGuard(ov::log::Level::INFO);

        OV_ASSERT_NO_THROW(firstResult = helper->isOptionSupported(compilerType, unlistedOption));
    }
    // The bulk list did not contain the option, so the individual probe must have created the adapter.
    ASSERT_NE(firstCallLogs.find(startMarkerFor(compilerType)), std::string::npos);

    std::string secondCallLogs;
    std::mutex secondCallLogsMutex;
    std::function<void(std::string_view)> secondCallLogCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(secondCallLogsMutex);
        secondCallLogs.append(msg);
        secondCallLogs.push_back('\n');
    };

    bool secondResult = false;
    {
        utils::LogCallbackGuard logCallbackGuard(secondCallLogCb);
        utils::LoggerLevelGuard loggerLevelGuard(ov::log::Level::INFO);

        OV_ASSERT_NO_THROW(secondResult = helper->isOptionSupported(compilerType, unlistedOption));
    }
    EXPECT_EQ(firstResult, secondResult);
    // The individual probe above populated the shared cache, so the repeat lookup must not recreate
    // the adapter.
    ASSERT_EQ(secondCallLogs.find(startMarkerFor(compilerType)), std::string::npos);
}

// Cross-checks BATCH_MODE support: first against a freshly created compiler adapter directly (ground
// truth, bypassing the helper's cache), then through CompilerOptionSupportHelper, and asserts both agree.
// Finally, repeats the helper lookup to confirm the second call is served from cache.
TEST_P(CompilerOptionSupportHelperFunctionalTests, BatchModeSupportMatchesDirectCompilerProbe) {
    const std::string batchModeOption = ov::intel_npu::batch_mode.name();

    ::intel_npu::CompilerAdapterFactory factory;
    auto adapterCompilerType = compilerType;
    std::unique_ptr<::intel_npu::ICompilerAdapter> compilerAdapter;
    OV_ASSERT_NO_THROW(compilerAdapter = factory.getCompiler(backend, adapterCompilerType, ""));

    bool supportedByCompiler = false;
    OV_ASSERT_NO_THROW(supportedByCompiler = compilerAdapter->is_option_supported(batchModeOption));

    bool supportedByHelper = false;
    OV_ASSERT_NO_THROW(supportedByHelper = helper->isOptionSupported(compilerType, batchModeOption));

    EXPECT_EQ(supportedByCompiler, supportedByHelper);

    std::string logs;
    std::mutex logsMutex;
    std::function<void(std::string_view)> logCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logsMutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    bool cachedResult = false;
    {
        utils::LogCallbackGuard logCallbackGuard(logCb);
        utils::LoggerLevelGuard loggerLevelGuard(ov::log::Level::INFO);

        // Repeat lookup for the same option: it must be served from the cache populated above,
        // without recreating the adapter.
        OV_ASSERT_NO_THROW(cachedResult = helper->isOptionSupported(compilerType, batchModeOption));
    }

    EXPECT_EQ(supportedByHelper, cachedResult);
    ASSERT_EQ(logs.find(startMarkerFor(compilerType)), std::string::npos);
}

// Stresses the shared, per-compiler-type once_flag/cache guard in CompilerOptionSupportHelper::isOptionSupported
// with concurrent lookups for the same key. Bulk loading ("get_supported_options") must still run exactly once.
TEST_P(CompilerOptionSupportHelperFunctionalTests, ConcurrentLookupsForSameKeyLoadOptionsOnceWithoutRaces) {
    constexpr int kThreadCount = 8;
    std::vector<std::thread> threads;
    std::vector<bool> results(kThreadCount, false);
    std::vector<std::exception_ptr> failures(kThreadCount);

    std::string logs;
    std::mutex logsMutex;
    std::function<void(std::string_view)> logCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logsMutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard logCallbackGuard(logCb);
        utils::LoggerLevelGuard loggerLevelGuard(ov::log::Level::DEBUG);

        for (int i = 0; i < kThreadCount; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    // Mixing options already known to be part of the bulk list, queried concurrently
                    // for the same compiler type/key.
                    const auto& optionName = knownSupportedOptions[i % knownSupportedOptions.size()];
                    results[i] = helper->isOptionSupported(compilerType, optionName);
                } catch (...) {
                    failures[i] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    for (int i = 0; i < kThreadCount; ++i) {
        ASSERT_FALSE(failures[i]) << "Thread " << i << " threw an exception";
        EXPECT_TRUE(results[i]) << "Thread " << i;
    }

    const auto marker = bulkLoadMarkerFor(compilerType);
    if (!marker.empty()) {
        // Regardless of how many threads raced past the cache-miss check, the expensive bulk
        // "get_supported_options" retrieval, guarded by the per-key once_flag, must run exactly once.
        EXPECT_EQ(countOccurrences(logs, marker), 1u);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         CompilerOptionSupportHelperFunctionalTests,
                         ::testing::Values(ov::intel_npu::CompilerType::DRIVER, ov::intel_npu::CompilerType::PLUGIN),
                         CompilerOptionSupportHelperFunctionalTests::getTestCaseName);

// Single helper instance shared by concurrent lookups against two independent compiler-type keys
// (DRIVER and PLUGIN), to verify the once_flag/cache map has no cross-key races or deadlocks.
class CompilerOptionSupportHelperConcurrencyTests : public ov::test::behavior::OVPluginTestBase {
protected:
    ov::SoPtr<::intel_npu::IEngineBackend> backend;
    std::unique_ptr<::intel_npu::CompilerOptionSupportHelper> helper;

    std::string targetDevice;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_NPU;

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();

        backend = ov::SoPtr<::intel_npu::IEngineBackend>(std::make_shared<::intel_npu::ZeroEngineBackend>());
        helper =
            std::make_unique<::intel_npu::CompilerOptionSupportHelper>(backend, ::intel_npu::CompilerAdapterFactory());
    }

    void TearDown() override {
        OVPluginTestBase::TearDown();
    }
};

TEST_F(CompilerOptionSupportHelperConcurrencyTests,
      ConcurrentLookupsForDifferentKeysLoadOptionsOncePerKeyWithoutDeadlock) {
    constexpr int kThreadsPerType = 4;
    std::vector<std::thread> threads;
    std::vector<bool> results(kThreadsPerType * 2, false);
    std::vector<std::exception_ptr> failures(kThreadsPerType * 2);

    std::string logs;
    std::mutex logsMutex;
    std::function<void(std::string_view)> logCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logsMutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    {
        utils::LogCallbackGuard logCallbackGuard(logCb);
        utils::LoggerLevelGuard loggerLevelGuard(ov::log::Level::DEBUG);

        for (int i = 0; i < kThreadsPerType; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    results[i] = helper->isOptionSupported(ov::intel_npu::CompilerType::DRIVER,
                                                           knownSupportedOptions[i % knownSupportedOptions.size()]);
                } catch (...) {
                    failures[i] = std::current_exception();
                }
            });
            threads.emplace_back([&, i]() {
                const int idx = kThreadsPerType + i;
                try {
                    results[idx] = helper->isOptionSupported(ov::intel_npu::CompilerType::PLUGIN,
                                                             knownSupportedOptions[i % knownSupportedOptions.size()]);
                } catch (...) {
                    failures[idx] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    for (size_t i = 0; i < results.size(); ++i) {
        ASSERT_FALSE(failures[i]) << "Thread " << i << " threw an exception";
        EXPECT_TRUE(results[i]) << "Thread " << i;
    }

    // Concurrent traffic on the DRIVER key must not block or interfere with the independent
    // once_flag/cache entry for the PLUGIN key (and vice versa): PLUGIN must still bulk-load exactly once.
    EXPECT_EQ(countOccurrences(logs, bulkLoadMarkerFor(ov::intel_npu::CompilerType::PLUGIN)), 1u);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
