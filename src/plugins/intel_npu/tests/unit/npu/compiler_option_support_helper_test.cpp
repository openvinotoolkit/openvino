// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_option_support_helper.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "vcl_provider_utils.hpp"

namespace {

using namespace ::intel_npu;

using CompilerType = ov::intel_npu::CompilerType;

// No case below reaches the compiler-in-plugin branch of the factory. Each is answered from the
// cache, rejected before any lookup, or uses the driver path without a device.
VCLFunctionTableProvider noVclLibrary() {
    return vcl_provider::forbidden("The VCL function table provider must not be invoked by this test");
}

// Minimal IEngineBackend double: exercises the real CompilerAdapterFactory::getCompiler() logic
// (no device available) without requiring an actual NPU driver/compiler in the test environment.
class MockEngineBackendNoDevice : public IEngineBackend {
public:
    const std::shared_ptr<IDevice> getDevice() const override {
        return nullptr;
    }
    const std::string getName() const override {
        return "MOCK";
    }
    bool isCommandQueueExtSupported() const override {
        return false;
    }
    bool isLUIDExtSupported() const override {
        return false;
    }
    bool isContextExtSupported() const override {
        return false;
    }
    void updateInfo(const ov::AnyMap&) override {}
};

TEST(CompilerOptionSupportHelperTests, ReturnsCachedSupportedOptionWithoutBackend) {
    CompilerOptionSupportHelper helper({nullptr}, CompilerAdapterFactory{noVclLibrary()});
    const auto key = static_cast<OptionSupportCache::CacheKey>(CompilerType::PLUGIN);
    helper.getOptionSupportCache()->addSupportedOption(key, "CACHED_OPTION", true);

    EXPECT_TRUE(helper.isOptionSupported(CompilerType::PLUGIN, "CACHED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, ReturnsCachedUnsupportedOptionWithoutBackend) {
    CompilerOptionSupportHelper helper({nullptr}, CompilerAdapterFactory{noVclLibrary()});
    const auto key = static_cast<OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(key, "CACHED_OPTION", false);

    EXPECT_FALSE(helper.isOptionSupported(CompilerType::DRIVER, "CACHED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, KeepsCompilerTypeCacheEntriesIndependent) {
    CompilerOptionSupportHelper helper({nullptr}, CompilerAdapterFactory{noVclLibrary()});
    const auto pluginKey = static_cast<OptionSupportCache::CacheKey>(CompilerType::PLUGIN);
    const auto driverKey = static_cast<OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(pluginKey, "SHARED_OPTION", true);
    helper.getOptionSupportCache()->addSupportedOption(driverKey, "SHARED_OPTION", false);

    EXPECT_TRUE(helper.isOptionSupported(CompilerType::PLUGIN, "SHARED_OPTION"));
    EXPECT_FALSE(helper.isOptionSupported(CompilerType::DRIVER, "SHARED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, RejectsPreferPluginBeforeCompilerLookup) {
    CompilerOptionSupportHelper helper({nullptr}, CompilerAdapterFactory{noVclLibrary()});

    EXPECT_THROW(helper.isOptionSupported(CompilerType::PREFER_PLUGIN, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, RejectsCompilerTypeOutsideSupportedList) {
    CompilerOptionSupportHelper helper({nullptr}, CompilerAdapterFactory{noVclLibrary()});

    EXPECT_THROW(helper.isOptionSupported(static_cast<CompilerType>(999), "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, ExposesSharedOptionSupportCache) {
    CompilerOptionSupportHelper helper({nullptr}, CompilerAdapterFactory{noVclLibrary()});

    ASSERT_NE(helper.getOptionSupportCache(), nullptr);
}

TEST(CompilerOptionSupportHelperTests, DriverLookupWithoutDeviceThrows) {
    auto backend = std::make_shared<MockEngineBackendNoDevice>();
    ov::SoPtr<IEngineBackend> backendPtr(backend);
    CompilerOptionSupportHelper helper(backendPtr, CompilerAdapterFactory{noVclLibrary()});

    EXPECT_THROW(helper.isOptionSupported(CompilerType::DRIVER, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, OptionValueBypassesCacheAndReachesCompilerLookup) {
    auto backend = std::make_shared<MockEngineBackendNoDevice>();
    ov::SoPtr<IEngineBackend> backendPtr(backend);
    CompilerOptionSupportHelper helper(backendPtr, CompilerAdapterFactory{noVclLibrary()});
    const auto key = static_cast<OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(key, "ANY_OPTION", true);

    // Even though "ANY_OPTION" is cached as supported, providing a concrete value must skip the
    // cache and fall through to the real compiler lookup, which throws without a device.
    EXPECT_THROW(helper.isOptionSupported(CompilerType::DRIVER, "ANY_OPTION", std::string("1")), ov::Exception);
}

}  // namespace
