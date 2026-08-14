// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "compiler_option_support_helper.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
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

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         CompilerOptionSupportHelperFunctionalTests,
                         ::testing::Values(ov::intel_npu::CompilerType::DRIVER, ov::intel_npu::CompilerType::PLUGIN),
                         CompilerOptionSupportHelperFunctionalTests::getTestCaseName);

}  // namespace behavior
}  // namespace test
}  // namespace ov
