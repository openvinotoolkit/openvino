// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/option_support_cache.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

namespace {

using CacheKey = ::intel_npu::OptionSupportCache::CacheKey;

constexpr CacheKey kFirstKey = 1u;
constexpr CacheKey kSecondKey = 2u;
constexpr CacheKey kThirdKey = 3u;

TEST(OptionSupportCacheTests, UnknownOptionsAreReportedAsUnsupportedByDefault) {
    ::intel_npu::OptionSupportCache cache;
    EXPECT_FALSE(cache.isOptionSupported(kFirstKey, "CACHE_TEST_UNKNOWN_OPTION"));
    EXPECT_FALSE(cache.isOptionSupported(kSecondKey, "CACHE_TEST_UNKNOWN_OPTION"));
}

TEST(OptionSupportCacheTests, ArbitraryKeysAreSupported) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kThirdKey, "CACHE_TEST_OPT");
    EXPECT_TRUE(cache.isOptionSupported(kThirdKey, "CACHE_TEST_OPT"));
}

TEST(OptionSupportCacheTests, AddedOptionIsFound) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_A");
    EXPECT_TRUE(cache.isOptionSupported(kFirstKey, "OPT_A"));
    EXPECT_FALSE(cache.isOptionSupported(kSecondKey, "OPT_A"));
}

TEST(OptionSupportCacheTests, SetOptionsAreFound) {
    ::intel_npu::OptionSupportCache cache;
    cache.setSupportedOptions(kSecondKey, {"OPT_X", "OPT_Y"});
    EXPECT_TRUE(cache.isOptionSupported(kSecondKey, "OPT_X"));
    EXPECT_TRUE(cache.isOptionSupported(kSecondKey, "OPT_Y"));
    EXPECT_FALSE(cache.isOptionSupported(kFirstKey, "OPT_X"));
}

TEST(OptionSupportCacheTests, SetOptionsMergesWithExisting) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_EARLY");
    cache.setSupportedOptions(kFirstKey, {"OPT_BULK"});
    EXPECT_TRUE(cache.isOptionSupported(kFirstKey, "OPT_EARLY"));
    EXPECT_TRUE(cache.isOptionSupported(kFirstKey, "OPT_BULK"));
}

TEST(OptionSupportCacheTests, OptionWithValueIsFound) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_VAL", std::string{"VALUE_A"});
    EXPECT_TRUE(cache.isOptionSupported(kFirstKey, "OPT_VAL", std::string{"VALUE_A"}));
    EXPECT_FALSE(cache.isOptionSupported(kFirstKey, "OPT_VAL", std::string{"VALUE_B"}));
    EXPECT_FALSE(cache.isOptionSupported(kFirstKey, "OPT_VAL"));
}

TEST(OptionSupportCacheTests, DifferentKeysAreIndependent) {
    ::intel_npu::OptionSupportCache cache;
    cache.setSupportedOptions(kFirstKey, {"FIRST_ONLY"});
    cache.setSupportedOptions(kSecondKey, {"SECOND_ONLY"});
    EXPECT_TRUE(cache.isOptionSupported(kFirstKey, "FIRST_ONLY"));
    EXPECT_FALSE(cache.isOptionSupported(kFirstKey, "SECOND_ONLY"));
    EXPECT_TRUE(cache.isOptionSupported(kSecondKey, "SECOND_ONLY"));
    EXPECT_FALSE(cache.isOptionSupported(kSecondKey, "FIRST_ONLY"));
}

TEST(OptionSupportCacheTests, AddDoesNotCreateDuplicates) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_DUP");
    cache.addSupportedOption(kFirstKey, "OPT_DUP");
    EXPECT_TRUE(cache.isOptionSupported(kFirstKey, "OPT_DUP"));
}

}  // namespace
