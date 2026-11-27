// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "util.hpp"

namespace {

// ===================== get_qsa_state_kinds / qsa_state_kinds_alternation =====================

TEST(QSAStateKindsTest, ListsTheThreeKinds) {
    EXPECT_EQ(ov::npuw::util::get_qsa_state_kinds(),
              (std::vector<std::string>{"block_index_key", "incomplete_index_key", "incomplete_index_rope"}));
    EXPECT_EQ(ov::npuw::util::qsa_state_kinds_alternation(),
              "block_index_key|incomplete_index_key|incomplete_index_rope");
}

// ===================== matchQSAStateString =====================

TEST(MatchQSAStateStringTest, MatchesEveryKind) {
    for (const auto& kind : ov::npuw::util::get_qsa_state_kinds()) {
        SCOPED_TRACE(kind);
        EXPECT_TRUE(ov::npuw::util::matchQSAStateString("cache_params.past." + kind + ".0", "past"));
        EXPECT_TRUE(ov::npuw::util::matchQSAStateString("cache_params.past." + kind + ".31", "past"));
        EXPECT_TRUE(ov::npuw::util::matchQSAStateString("cache_params.present." + kind + ".0", "present"));
        EXPECT_FALSE(ov::npuw::util::matchQSAStateString("cache_params.present." + kind + ".0", "past"));
        EXPECT_FALSE(ov::npuw::util::matchQSAStateString("cache_params.past." + kind + ".0", "present"));
        EXPECT_FALSE(ov::npuw::util::matchQSAStateString("cache_params.past." + kind));  // no index
    }
}

TEST(MatchQSAStateStringTest, DefaultParameterIsPast) {
    EXPECT_TRUE(ov::npuw::util::matchQSAStateString("cache_params.past.block_index_key.0"));
    EXPECT_FALSE(ov::npuw::util::matchQSAStateString("cache_params.present.block_index_key.0"));
}

TEST(MatchQSAStateStringTest, DoesNotMatchOtherNames) {
    EXPECT_FALSE(ov::npuw::util::matchQSAStateString("", "past"));
    EXPECT_FALSE(ov::npuw::util::matchQSAStateString("past_key_values.0.key", "past"));
    EXPECT_FALSE(ov::npuw::util::matchQSAStateString("cache_params.past.conv.0", "past"));
    EXPECT_FALSE(ov::npuw::util::matchQSAStateString("cache_params.past.ssm.0", "past"));
    EXPECT_FALSE(ov::npuw::util::matchQSAStateString("cache_params.past.index_key.0", "past"));
    // the StatefulToStateless names are renamed by ConvertSparseAttentionToDense before any lookup
    EXPECT_FALSE(ov::npuw::util::matchQSAStateString(
        "input_restored.cache_params.past.block_index_key.0cache_params.present.block_index_key.0",
        "past"));
}

// ===================== starts_with_past_qsa_state =====================

TEST(StartsWithPastQSAStateTest, DetectsPastNamesOnly) {
    for (const auto& kind : ov::npuw::util::get_qsa_state_kinds()) {
        SCOPED_TRACE(kind);
        EXPECT_TRUE(ov::npuw::util::starts_with_past_qsa_state("cache_params.past." + kind + ".0"));
        EXPECT_FALSE(ov::npuw::util::starts_with_past_qsa_state("cache_params.present." + kind + ".0"));
    }
    EXPECT_FALSE(ov::npuw::util::starts_with_past_qsa_state("cache_params.past.conv.0"));
    EXPECT_FALSE(ov::npuw::util::starts_with_past_qsa_state("cache_params.past.ssm.0"));
    EXPECT_FALSE(ov::npuw::util::starts_with_past_qsa_state("past_key_values.0.key"));
    EXPECT_FALSE(ov::npuw::util::starts_with_past_qsa_state(""));
}

// The QSA states share the lincache carrying contract but are not linear-attention states:
// the lincache predicates must not pick them up.
TEST(StartsWithPastQSAStateTest, NotDetectedAsLinCache) {
    for (const auto& kind : ov::npuw::util::get_qsa_state_kinds()) {
        SCOPED_TRACE(kind);
        EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.past." + kind + ".0", "past"));
        EXPECT_FALSE(ov::npuw::util::matchLinCacheString("cache_params.present." + kind + ".0", "present"));
        EXPECT_FALSE(ov::npuw::util::starts_with_past_lincache("cache_params.past." + kind + ".0"));
    }
}

}  // namespace
