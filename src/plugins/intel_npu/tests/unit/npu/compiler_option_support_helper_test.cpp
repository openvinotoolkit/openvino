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
#include "npu/compiler_adapter/fake_vcl.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "vcl_provider_utils.hpp"

namespace {

using namespace ::intel_npu;
namespace vcl_provider = ov::test::npu::vcl_provider;

using CompilerType = ov::intel_npu::CompilerType;

// No case below reaches the compiler-in-plugin branch of the factory, unless it brings its own VCL. Each is answered
// from the cache, rejected before any lookup, or uses the driver path without a backend.
std::shared_ptr<const CompilerAdapterFactory> factoryWithoutVclLibrary() {
    return std::make_shared<const CompilerAdapterFactory>(
        vcl_provider::forbidden("The VCL function table provider must not be invoked by this test"));
}

TEST(CompilerOptionSupportHelperTests, Construction_WithoutAdapterFactory_Throws) {
    EXPECT_THROW(CompilerOptionSupportHelper({nullptr}, nullptr), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, ReturnsCachedSupportedOptionWithoutBackend) {
    CompilerOptionSupportHelper helper({nullptr}, factoryWithoutVclLibrary());
    const auto key = static_cast<OptionSupportCache::CacheKey>(CompilerType::PLUGIN);
    helper.getOptionSupportCache()->addSupportedOption(key, "CACHED_OPTION", true);

    EXPECT_TRUE(helper.isOptionSupported(CompilerType::PLUGIN, "CACHED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, ReturnsCachedUnsupportedOptionWithoutBackend) {
    CompilerOptionSupportHelper helper({nullptr}, factoryWithoutVclLibrary());
    const auto key = static_cast<OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(key, "CACHED_OPTION", false);

    EXPECT_FALSE(helper.isOptionSupported(CompilerType::DRIVER, "CACHED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, KeepsCompilerTypeCacheEntriesIndependent) {
    CompilerOptionSupportHelper helper({nullptr}, factoryWithoutVclLibrary());
    const auto pluginKey = static_cast<OptionSupportCache::CacheKey>(CompilerType::PLUGIN);
    const auto driverKey = static_cast<OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(pluginKey, "SHARED_OPTION", true);
    helper.getOptionSupportCache()->addSupportedOption(driverKey, "SHARED_OPTION", false);

    EXPECT_TRUE(helper.isOptionSupported(CompilerType::PLUGIN, "SHARED_OPTION"));
    EXPECT_FALSE(helper.isOptionSupported(CompilerType::DRIVER, "SHARED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, RejectsPreferPluginBeforeCompilerLookup) {
    CompilerOptionSupportHelper helper({nullptr}, factoryWithoutVclLibrary());

    EXPECT_THROW(helper.isOptionSupported(CompilerType::PREFER_PLUGIN, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, RejectsCompilerTypeOutsideSupportedList) {
    CompilerOptionSupportHelper helper({nullptr}, factoryWithoutVclLibrary());

    EXPECT_THROW(helper.isOptionSupported(static_cast<CompilerType>(999), "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, ExposesSharedOptionSupportCache) {
    CompilerOptionSupportHelper helper({nullptr}, factoryWithoutVclLibrary());

    ASSERT_NE(helper.getOptionSupportCache(), nullptr);
}

TEST(CompilerOptionSupportHelperTests, IsOptionSupported_DriverWithoutBackend_Throws) {
    CompilerOptionSupportHelper helper({nullptr}, factoryWithoutVclLibrary());

    EXPECT_THROW(helper.isOptionSupported(CompilerType::DRIVER, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, OptionValueBypassesCacheAndReachesCompilerLookup) {
    CompilerOptionSupportHelper helper({nullptr}, factoryWithoutVclLibrary());
    const auto key = static_cast<OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(key, "ANY_OPTION", true);

    // Even though "ANY_OPTION" is cached as supported, providing a concrete value must skip the
    // cache and fall through to the real compiler lookup, which throws without a backend.
    EXPECT_THROW(helper.isOptionSupported(CompilerType::DRIVER, "ANY_OPTION", std::string("1")), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, IsOptionSupported_PluginWithoutBackend_FollowsTheCompiler) {
    fake_vcl::FakeVcl fake;
    fake.unsupportedOptions.insert("UNSUPPORTED_OPTION");
    CompilerOptionSupportHelper helper({nullptr}, std::make_shared<const CompilerAdapterFactory>(fake.provider()));

    EXPECT_TRUE(helper.isOptionSupported(CompilerType::PLUGIN, "SUPPORTED_OPTION"));
    EXPECT_FALSE(helper.isOptionSupported(CompilerType::PLUGIN, "UNSUPPORTED_OPTION"));
}

TEST(CompilerOptionSupportHelperTests, IsOptionSupported_PluginWithoutBackend_LoadsTheCompilerOptionsOnce) {
    fake_vcl::FakeVcl fake;
    CompilerOptionSupportHelper helper({nullptr}, std::make_shared<const CompilerAdapterFactory>(fake.provider()));

    helper.isOptionSupported(CompilerType::PLUGIN, "OPTION_A");
    const auto loadCalls = fake.callCount(fake_vcl::kGetCompilerSupportedOptions);
    helper.isOptionSupported(CompilerType::PLUGIN, "OPTION_B");
    helper.isOptionSupported(CompilerType::PLUGIN, "OPTION_A");

    EXPECT_GT(loadCalls, 0u);
    EXPECT_EQ(fake.callCount(fake_vcl::kGetCompilerSupportedOptions), loadCalls);
}

}  // namespace
