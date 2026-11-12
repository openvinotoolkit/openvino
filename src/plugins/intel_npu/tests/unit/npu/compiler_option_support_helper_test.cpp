// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_option_support_helper.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "openvino/core/except.hpp"

namespace {

using CompilerType = ov::intel_npu::CompilerType;

struct CompilerOptionSupportTestContext {
    std::shared_ptr<::intel_npu::OptionSupportCache> cache;
    std::shared_ptr<const ::intel_npu::CompilerAdapterFactory> factory;
    std::unique_ptr<::intel_npu::CompilerOptionSupportHelper> helper;
};

CompilerOptionSupportTestContext makeContext(
    const ov::SoPtr<::intel_npu::IEngineBackend>& backend = ov::SoPtr<::intel_npu::IEngineBackend>{nullptr}) {
    auto cache = std::make_shared<::intel_npu::OptionSupportCache>();
    auto factory = std::make_shared<const ::intel_npu::CompilerAdapterFactory>(backend,
                                                                               ::intel_npu::VCLFunctionTableProvider{},
                                                                               cache);
    auto helper = std::make_unique<::intel_npu::CompilerOptionSupportHelper>(backend, factory, cache);
    return {std::move(cache), std::move(factory), std::move(helper)};
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
    auto context = makeContext();
    const auto key = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::PLUGIN);
    context.cache->addSupportedOption(key, "CACHED_OPTION", true);

    EXPECT_TRUE(context.helper->isOptionSupported(CompilerType::PLUGIN, "CACHED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, ReturnsCachedUnsupportedOptionWithoutBackend) {
    auto context = makeContext();
    const auto key = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    context.cache->addSupportedOption(key, "CACHED_OPTION", false);

    EXPECT_FALSE(context.helper->isOptionSupported(CompilerType::DRIVER, "CACHED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, KeepsCompilerTypeCacheEntriesIndependent) {
    auto context = makeContext();
    const auto pluginKey = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::PLUGIN);
    const auto driverKey = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    context.cache->addSupportedOption(pluginKey, "SHARED_OPTION", true);
    context.cache->addSupportedOption(driverKey, "SHARED_OPTION", false);

    EXPECT_TRUE(context.helper->isOptionSupported(CompilerType::PLUGIN, "SHARED_OPTION"));
    EXPECT_FALSE(context.helper->isOptionSupported(CompilerType::DRIVER, "SHARED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, RejectsPreferPluginBeforeCompilerLookup) {
    auto context = makeContext();

    EXPECT_THROW(context.helper->isOptionSupported(CompilerType::PREFER_PLUGIN, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, RejectsCompilerTypeOutsideSupportedList) {
    auto context = makeContext();

    EXPECT_THROW(context.helper->isOptionSupported(static_cast<CompilerType>(999), "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, ContextSharesCacheWithFactoryAndHelper) {
    const auto context = makeContext();
    ASSERT_NE(context.cache, nullptr);
    ASSERT_NE(context.factory, nullptr);
    ASSERT_NE(context.helper, nullptr);
}

TEST(CompilerOptionSupportHelperTests, DriverLookupWithoutDeviceThrows) {
    auto backend = std::make_shared<MockEngineBackendNoDevice>();
    ov::SoPtr<::intel_npu::IEngineBackend> backendPtr(backend);
    auto context = makeContext(backendPtr);

    EXPECT_THROW(context.helper->isOptionSupported(CompilerType::DRIVER, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, OptionValueBypassesCacheAndReachesCompilerLookup) {
    auto backend = std::make_shared<MockEngineBackendNoDevice>();
    ov::SoPtr<::intel_npu::IEngineBackend> backendPtr(backend);
    auto context = makeContext(backendPtr);
    const auto key = static_cast<::intel_npu::OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    context.cache->addSupportedOption(key, "ANY_OPTION", true);

    // Even though "ANY_OPTION" is cached as supported, providing a concrete value must skip the
    // cache and fall through to the real compiler lookup, which throws without a device.
    EXPECT_THROW(context.helper->isOptionSupported(CompilerType::DRIVER, "ANY_OPTION", std::string("1")),
                 ov::Exception);
}

}  // namespace
