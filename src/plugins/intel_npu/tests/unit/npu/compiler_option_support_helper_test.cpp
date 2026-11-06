// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_option_support_helper.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "npu/mock_backend_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "vcl_provider_utils.hpp"

namespace {

using namespace ::intel_npu;

using CompilerType = ov::intel_npu::CompilerType;

// No case below reaches the compiler-in-plugin branch of the factory. Each is answered from the
// cache, rejected before any lookup, or uses the driver path without a device.
std::shared_ptr<const CompilerAdapterFactory> factoryWithoutVclLibrary() {
    return std::make_shared<const CompilerAdapterFactory>(
        vcl_provider::forbidden("The VCL function table provider must not be invoked by this test"));
}

// A backend without devices, like in offline compilation. It exercises the real CompilerAdapterFactory::getCompiler()
// logic without requiring an actual NPU driver/compiler in the test environment.
ov::SoPtr<IEngineBackend> makeBackendWithoutDevice() {
    return ov::SoPtr<IEngineBackend>(std::make_shared<npu_mock::MockEngineBackend>(std::vector<std::string>{}));
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

TEST(CompilerOptionSupportHelperTests, DriverLookupWithoutDeviceThrows) {
    CompilerOptionSupportHelper helper(makeBackendWithoutDevice(), factoryWithoutVclLibrary());

    EXPECT_THROW(helper.isOptionSupported(CompilerType::DRIVER, "ANY_OPTION"), ov::Exception);
}

TEST(CompilerOptionSupportHelperTests, OptionValueBypassesCacheAndReachesCompilerLookup) {
    CompilerOptionSupportHelper helper(makeBackendWithoutDevice(), factoryWithoutVclLibrary());
    const auto key = static_cast<OptionSupportCache::CacheKey>(CompilerType::DRIVER);
    helper.getOptionSupportCache()->addSupportedOption(key, "ANY_OPTION", true);

    // Even though "ANY_OPTION" is cached as supported, providing a concrete value must skip the
    // cache and fall through to the real compiler lookup, which throws without a device.
    EXPECT_THROW(helper.isOptionSupported(CompilerType::DRIVER, "ANY_OPTION", std::string("1")), ov::Exception);
}

}  // namespace
