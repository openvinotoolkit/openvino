// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/option_support_cache.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"

namespace {

using CacheKey = ::intel_npu::OptionSupportCache::CacheKey;

constexpr CacheKey kFirstKey = 1u;
constexpr CacheKey kSecondKey = 2u;
constexpr CacheKey kThirdKey = 3u;

TEST(OptionSupportCacheTests, UnknownOptionsAreReportedAsUnsupportedByDefault) {
    ::intel_npu::OptionSupportCache cache;
    EXPECT_FALSE(cache.isOptionSupported(kFirstKey, "CACHE_TEST_UNKNOWN_OPTION").has_value());
    EXPECT_FALSE(cache.isOptionSupported(kSecondKey, "CACHE_TEST_UNKNOWN_OPTION").has_value());
}

TEST(OptionSupportCacheTests, ArbitraryKeysAreSupported) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kThirdKey, "CACHE_TEST_OPT");
    EXPECT_EQ(cache.isOptionSupported(kThirdKey, "CACHE_TEST_OPT"), std::make_optional(true));
}

TEST(OptionSupportCacheTests, AddedOptionIsFound) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_A");
    EXPECT_EQ(cache.isOptionSupported(kFirstKey, "OPT_A"), std::make_optional(true));
    EXPECT_FALSE(cache.isOptionSupported(kSecondKey, "OPT_A").has_value());
}

TEST(OptionSupportCacheTests, SetOptionsAreFound) {
    ::intel_npu::OptionSupportCache cache;
    cache.setSupportedOptions(kSecondKey, {"OPT_X", "OPT_Y"});
    EXPECT_EQ(cache.isOptionSupported(kSecondKey, "OPT_X"), std::make_optional(true));
    EXPECT_EQ(cache.isOptionSupported(kSecondKey, "OPT_Y"), std::make_optional(true));
    EXPECT_FALSE(cache.isOptionSupported(kFirstKey, "OPT_X").has_value());
}

TEST(OptionSupportCacheTests, SetOptionsMergesWithExisting) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_EARLY");
    cache.setSupportedOptions(kFirstKey, {"OPT_BULK"});
    EXPECT_EQ(cache.isOptionSupported(kFirstKey, "OPT_EARLY"), std::make_optional(true));
    EXPECT_EQ(cache.isOptionSupported(kFirstKey, "OPT_BULK"), std::make_optional(true));
}

TEST(OptionSupportCacheTests, DifferentKeysAreIndependent) {
    ::intel_npu::OptionSupportCache cache;
    cache.setSupportedOptions(kFirstKey, {"FIRST_ONLY"});
    cache.setSupportedOptions(kSecondKey, {"SECOND_ONLY"});
    EXPECT_EQ(cache.isOptionSupported(kFirstKey, "FIRST_ONLY"), std::make_optional(true));
    EXPECT_FALSE(cache.isOptionSupported(kFirstKey, "SECOND_ONLY").has_value());
    EXPECT_EQ(cache.isOptionSupported(kSecondKey, "SECOND_ONLY"), std::make_optional(true));
    EXPECT_FALSE(cache.isOptionSupported(kSecondKey, "FIRST_ONLY").has_value());
}

TEST(OptionSupportCacheTests, AddDoesNotCreateDuplicates) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_DUP");
    cache.addSupportedOption(kFirstKey, "OPT_DUP");
    EXPECT_EQ(cache.isOptionSupported(kFirstKey, "OPT_DUP"), std::make_optional(true));
}

TEST(OptionSupportCacheTests, UnsupportedOptionCanBeCachedExplicitly) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_UNSUPPORTED", false);
    EXPECT_EQ(cache.isOptionSupported(kFirstKey, "OPT_UNSUPPORTED"), std::make_optional(false));
}

TEST(OptionSupportCacheTests, RepeatedUnsupportedOptionInsertIsIdempotent) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_UNSUPPORTED", false);
    cache.addSupportedOption(kFirstKey, "OPT_UNSUPPORTED", false);
    EXPECT_EQ(cache.isOptionSupported(kFirstKey, "OPT_UNSUPPORTED"), std::make_optional(false));
}

TEST(OptionSupportCacheTests, ConflictingSupportStateFromTrueToFalseThrows) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_FLIP", true);
    EXPECT_THROW(cache.addSupportedOption(kFirstKey, "OPT_FLIP", false), ov::Exception);
}

TEST(OptionSupportCacheTests, ConflictingSupportStateFromFalseToTrueThrows) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_FLIP", false);
    EXPECT_THROW(cache.addSupportedOption(kFirstKey, "OPT_FLIP", true), ov::Exception);
}

TEST(OptionSupportCacheTests, SetSupportedOptionsConflictsWithExistingUnsupportedState) {
    ::intel_npu::OptionSupportCache cache;
    cache.addSupportedOption(kFirstKey, "OPT_CONFLICT", false);
    EXPECT_THROW(cache.setSupportedOptions(kFirstKey, {"OPT_CONFLICT"}), ov::Exception);
}

}  // namespace
