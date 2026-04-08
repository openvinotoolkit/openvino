// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "util.hpp"

namespace {
// ===================== matchLinCacheString tests =====================

TEST(MatchLinCacheStringTest, MatchesPastConvCache) {
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.past.conv.0", "past"));
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.past.conv.99", "past"));
}

TEST(MatchLinCacheStringTest, MatchesPresentConvCache) {
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.present.conv.0", "present"));
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.present.conv.99", "present"));
}

TEST(MatchLinCacheStringTest, MatchesPastSsmCache) {
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.past.ssm.0", "past"));
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.past.ssm.5", "past"));
}

TEST(MatchLinCacheStringTest, MatchesPresentSsmCache) {
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.present.ssm.0", "present"));
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.present.ssm.5", "present"));
}

TEST(MatchLinCacheStringTest, DoesNotMatchKVCacheNames) {
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("past_key_values.0.key", "past"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("past_key_values.0.value", "past"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("present.0.key", "present"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("present.0.value", "present"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.past.key.0", "past"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.past.value.0", "past"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.present.key.0", "present"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.present.value.0", "present"));
}

TEST(MatchLinCacheStringTest, PastDoesNotMatchPresentAndViceVersa) {
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.present.conv.0", "past"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.past.conv.0", "present"));
}

TEST(MatchLinCacheStringTest, DoesNotMatchInvalidStrings) {
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("", "past"));
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.past.conv", "past"));  // no index
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.conv.0", "past"));  // missing past/present
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("random_string", "past"));
}

TEST(MatchLinCacheStringTest, DefaultParameterIsPast) {
    EXPECT_TRUE(ov::npuw::util::matchLinCacheString("cache_params.past.conv.0"));  // default = "past"
    EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.present.conv.0"));  // default = "past"
}

// ===================== starts_with_past_lincache tests =====================

TEST(StartsWithPastLincacheTest, DetectsConvCacheNames) {
    EXPECT_TRUE(ov::npuw::util::starts_with_past_lincache("cache_params.past.conv.0"));
    EXPECT_TRUE(ov::npuw::util::starts_with_past_lincache("cache_params.past.conv.15"));
}

TEST(StartsWithPastLincacheTest, DetectsSsmCacheNames) {
    EXPECT_TRUE(ov::npuw::util::starts_with_past_lincache("cache_params.past.ssm.0"));
    EXPECT_TRUE(ov::npuw::util::starts_with_past_lincache("cache_params.past.ssm.7"));
}

TEST(StartsWithPastLincacheTest, KVCacheNotDetected) {
    EXPECT_FALSE(ov::npuw::util::starts_with_past_lincache("past_key_values.0.key"));
    EXPECT_FALSE(ov::npuw::util::starts_with_past_lincache("past_key_values.0.value"));
}

TEST(StartsWithPastLincacheTest, PresentNamesNotDetected) {
    EXPECT_FALSE(ov::npuw::util::starts_with_past_lincache("cache_params.present.conv.0"));
    EXPECT_FALSE(ov::npuw::util::starts_with_past_lincache("cache_params.present.ssm.0"));
}

TEST(StartsWithPastLincacheTest, NonCacheNamesNotDetected) {
    EXPECT_FALSE(ov::npuw::util::starts_with_past_lincache("input_ids"));
    EXPECT_FALSE(ov::npuw::util::starts_with_past_lincache("attention_mask"));
    EXPECT_FALSE(ov::npuw::util::starts_with_past_lincache(""));
}

TEST(StartsWithPastLincacheTest, ClassifiesInputNamesCorrectly) {
    std::vector<std::string> kvcache_names;
    std::vector<std::string> lincache_names;

    std::vector<std::string> all_inputs = {
        "past_key_values.0.key",
        "past_key_values.0.value",
        "past_key_values.1.key",
        "past_key_values.1.value",
        "cache_params.past.conv.0",
        "cache_params.past.conv.1",
        "cache_params.past.ssm.0",
        "cache_params.past.ssm.1",
        "input_ids",
        "attention_mask"
    };

    for (const auto& name : all_inputs) {
        if (ov::npuw::util::starts_with(name, "past_key_values")) {
            kvcache_names.push_back(name);
        } else if (ov::npuw::util::starts_with_past_lincache(name)) {
            lincache_names.push_back(name);
        }
    }

    EXPECT_EQ(kvcache_names.size(), 4u);  // 2 keys + 2 values from layer 1
    EXPECT_EQ(lincache_names.size(), 4u);  // conv.0/1, ssm.0/1
}
}  // namespace
