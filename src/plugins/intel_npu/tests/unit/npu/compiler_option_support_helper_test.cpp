// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_option_support_helper.hpp"

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"

namespace {

using CompilerType = ov::intel_npu::CompilerType;

// None of the cases below reach the compiler-in-plugin branch of the factory: they are answered
// from the cache, rejected before any lookup, or exercise the driver path without a device. So the
// provider must never be invoked, and throwing is how that is enforced rather than assumed.
::intel_npu::VCLFunctionTableProvider noVclLibrary() {
    return []() -> std::shared_ptr<const ::intel_npu::VCLFunctionTable> {
        OPENVINO_THROW("The VCL function table provider must not be invoked by this test");
    };
}

// Minimal IEngineBackend double: exercises the real CompilerAdapterFactory::getCompiler() logic
// (no device available) without requiring an actual NPU driver/compiler in the test environment.
class MockEngineBackendNoDevice : public ::intel_npu::IEngineBackend {
public:
    const std::shared_ptr<::intel_npu::IDevice> getDevice() const override {
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
    ::intel_npu::CompilerOptionSupportHelper helper({nullptr}, ::intel_npu::CompilerAdapterFactory{noVclLibrary()});
    const auto key = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::PLUGIN);
    helper.getOptionSupportCache()->addSupportedOption(key, "CACHED_OPTION", true);

    EXPECT_TRUE(helper.isOptionSupported(CompilerType::PLUGIN, "CACHED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, ReturnsCachedUnsupportedOptionWithoutBackend) {
    ::intel_npu::CompilerOptionSupportHelper helper({nullptr}, ::intel_npu::CompilerAdapterFactory{noVclLibrary()});
    const auto key = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(key, "CACHED_OPTION", false);

    EXPECT_FALSE(helper.isOptionSupported(CompilerType::DRIVER, "CACHED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, KeepsCompilerTypeCacheEntriesIndependent) {
    ::intel_npu::CompilerOptionSupportHelper helper({nullptr}, ::intel_npu::CompilerAdapterFactory{noVclLibrary()});
    const auto pluginKey = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::PLUGIN);
    const auto driverKey = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(pluginKey, "SHARED_OPTION", true);
    helper.getOptionSupportCache()->addSupportedOption(driverKey, "SHARED_OPTION", false);

    EXPECT_TRUE(helper.isOptionSupported(CompilerType::PLUGIN, "SHARED_OPTION"));
    EXPECT_FALSE(helper.isOptionSupported(CompilerType::DRIVER, "SHARED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, RejectsPreferPluginBeforeCompilerLookup) {
    ::intel_npu::CompilerOptionSupportHelper helper({nullptr}, ::intel_npu::CompilerAdapterFactory{noVclLibrary()});

    EXPECT_THROW(helper.isOptionSupported(CompilerType::PREFER_PLUGIN, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, RejectsCompilerTypeOutsideSupportedList) {
    ::intel_npu::CompilerOptionSupportHelper helper({nullptr}, ::intel_npu::CompilerAdapterFactory{noVclLibrary()});

    EXPECT_THROW(helper.isOptionSupported(static_cast<CompilerType>(999), "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, ExposesSharedOptionSupportCache) {
    ::intel_npu::CompilerOptionSupportHelper helper({nullptr}, ::intel_npu::CompilerAdapterFactory{noVclLibrary()});

    ASSERT_NE(helper.getOptionSupportCache(), nullptr);
}

TEST(CompilerOptionSupportHelperTests, DriverLookupWithoutDeviceThrows) {
    auto backend = std::make_shared<MockEngineBackendNoDevice>();
    ov::SoPtr<::intel_npu::IEngineBackend> backendPtr(backend);
    ::intel_npu::CompilerOptionSupportHelper helper(backendPtr, ::intel_npu::CompilerAdapterFactory{noVclLibrary()});

    EXPECT_THROW(helper.isOptionSupported(CompilerType::DRIVER, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, OptionValueBypassesCacheAndReachesCompilerLookup) {
    auto backend = std::make_shared<MockEngineBackendNoDevice>();
    ov::SoPtr<::intel_npu::IEngineBackend> backendPtr(backend);
    ::intel_npu::CompilerOptionSupportHelper helper(backendPtr, ::intel_npu::CompilerAdapterFactory{noVclLibrary()});
    const auto key = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(key, "ANY_OPTION", true);

    // Even though "ANY_OPTION" is cached as supported, providing a concrete value must skip the
    // cache and fall through to the real compiler lookup, which throws without a device.
    EXPECT_THROW(helper.isOptionSupported(CompilerType::DRIVER, "ANY_OPTION", std::string("1")), ov::Exception);
}

}  // namespace
