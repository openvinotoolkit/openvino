// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_supported_options_cache.hpp"

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <optional>
#include <string>
#include <vector>

namespace {

TEST(CompilerSupportedOptionsCacheTests, UnknownOptionsAreReportedAsUnsupportedByDefault) {
    ::intel_npu::CompilerSupportedOptionsCache cache;
    EXPECT_FALSE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "CACHE_TEST_UNKNOWN_OPTION"));
    EXPECT_FALSE(cache.isOptionSupported(ov::intel_npu::CompilerType::PLUGIN, "CACHE_TEST_UNKNOWN_OPTION"));
}

TEST(CompilerSupportedOptionsCacheTests, RejectsNonExplicitCompilerType) {
    ::intel_npu::CompilerSupportedOptionsCache cache;
    EXPECT_THROW(cache.isOptionSupported(ov::intel_npu::CompilerType::PREFER_PLUGIN, "CACHE_TEST_OPT"), ov::Exception);
}

TEST(CompilerSupportedOptionsCacheTests, AddedOptionIsFound) {
    ::intel_npu::CompilerSupportedOptionsCache cache;
    cache.addSupportedOption(ov::intel_npu::CompilerType::DRIVER, "OPT_A");
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "OPT_A"));
    EXPECT_FALSE(cache.isOptionSupported(ov::intel_npu::CompilerType::PLUGIN, "OPT_A"));
}

TEST(CompilerSupportedOptionsCacheTests, SetOptionsAreFound) {
    ::intel_npu::CompilerSupportedOptionsCache cache;
    cache.setSupportedOptions(ov::intel_npu::CompilerType::PLUGIN, {"OPT_X", "OPT_Y"});
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::PLUGIN, "OPT_X"));
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::PLUGIN, "OPT_Y"));
    EXPECT_FALSE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "OPT_X"));
}

TEST(CompilerSupportedOptionsCacheTests, SetOptionsMergesWithExisting) {
    ::intel_npu::CompilerSupportedOptionsCache cache;
    cache.addSupportedOption(ov::intel_npu::CompilerType::DRIVER, "OPT_EARLY");
    cache.setSupportedOptions(ov::intel_npu::CompilerType::DRIVER, {"OPT_BULK"});
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "OPT_EARLY"));
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "OPT_BULK"));
}

TEST(CompilerSupportedOptionsCacheTests, OptionWithValueIsFound) {
    ::intel_npu::CompilerSupportedOptionsCache cache;
    cache.addSupportedOption(ov::intel_npu::CompilerType::DRIVER, "OPT_VAL", std::string{"VALUE_A"});
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "OPT_VAL", std::string{"VALUE_A"}));
    EXPECT_FALSE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "OPT_VAL", std::string{"VALUE_B"}));
    EXPECT_FALSE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "OPT_VAL"));
}

TEST(CompilerSupportedOptionsCacheTests, DriverAndPluginCachesAreIndependent) {
    ::intel_npu::CompilerSupportedOptionsCache cache;
    cache.setSupportedOptions(ov::intel_npu::CompilerType::DRIVER, {"DRIVER_ONLY"});
    cache.setSupportedOptions(ov::intel_npu::CompilerType::PLUGIN, {"PLUGIN_ONLY"});
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "DRIVER_ONLY"));
    EXPECT_FALSE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "PLUGIN_ONLY"));
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::PLUGIN, "PLUGIN_ONLY"));
    EXPECT_FALSE(cache.isOptionSupported(ov::intel_npu::CompilerType::PLUGIN, "DRIVER_ONLY"));
}

TEST(CompilerSupportedOptionsCacheTests, AddDoesNotCreateDuplicates) {
    ::intel_npu::CompilerSupportedOptionsCache cache;
    cache.addSupportedOption(ov::intel_npu::CompilerType::DRIVER, "OPT_DUP");
    cache.addSupportedOption(ov::intel_npu::CompilerType::DRIVER, "OPT_DUP");
    EXPECT_TRUE(cache.isOptionSupported(ov::intel_npu::CompilerType::DRIVER, "OPT_DUP"));
}

}  // namespace
