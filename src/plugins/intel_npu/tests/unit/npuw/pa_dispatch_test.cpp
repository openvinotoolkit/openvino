// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the PA dispatch-contract validation (ov::npuw::pa). The
// contract is a pure function over the parsed control tensors, so the tests
// construct Dispatch values directly and assert on the specific violation
// each malformed dispatch must report.

#include "pa_dispatch.hpp"

#include <gtest/gtest.h>

#include <string>

#include "openvino/core/except.hpp"

namespace {

using ov::npuw::pa::Dispatch;
using ov::npuw::pa::plan_calls;
using ov::npuw::pa::validate_dispatch;

// A well-formed two-subsequence dispatch: 4 + 2 scheduled tokens on top of
// 0 / 6 past tokens, a covering block table (block_size 4) and one sampled
// token per subsequence.
Dispatch make_valid_dispatch() {
    Dispatch d;
    d.past_lens = {0, 6};
    d.subsequence_begins = {0, 4, 6};
    d.block_indices = {0, 1, 2};
    d.block_indices_begins = {0, 1, 3};
    d.sampled_tokens_indices = {3, 5};
    d.max_context_len = 8;
    d.input_ids_size = 6;
    d.position_ids_token_count = 6;
    d.has_block_table = true;
    d.has_sampled_tokens = true;
    return d;
}

constexpr std::size_t kBlockSize = 4u;

void expect_violation(const Dispatch& d, const std::string& what, std::size_t block_size = kBlockSize) {
    try {
        validate_dispatch(d, block_size, 0u);
        FAIL() << "expected a dispatch-contract violation: " << what;
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find(what), std::string::npos) << ex.what();
    }
}

TEST(PADispatchContract, ValidDispatchPasses) {
    EXPECT_NO_THROW(validate_dispatch(make_valid_dispatch(), kBlockSize, 0u));
}

TEST(PADispatchContract, TokenAndSequenceCountsDerive) {
    const auto d = make_valid_dispatch();
    EXPECT_EQ(d.tokens(), 6);
    EXPECT_EQ(d.sequences(), 2);
}

// input_ids is absent on embedding-input models; -1 skips the cross-check.
TEST(PADispatchContract, AbsentInputIdsIsLegal) {
    auto d = make_valid_dispatch();
    d.input_ids_size = -1;
    EXPECT_NO_THROW(validate_dispatch(d, kBlockSize, 0u));
}

TEST(PADispatchContract, InputIdsSizeMismatchRejected) {
    auto d = make_valid_dispatch();
    d.input_ids_size = 5;
    expect_violation(d, "input_ids size != subsequence_begins token count");
}

TEST(PADispatchContract, PositionIdsCountMismatchRejected) {
    auto d = make_valid_dispatch();
    d.position_ids_token_count = 7;
    expect_violation(d, "position_ids last dim != subsequence_begins token count");
}

TEST(PADispatchContract, SubsequenceBeginsSizeMismatchRejected) {
    auto d = make_valid_dispatch();
    d.past_lens = {0};  // now sub.size() != past.size() + 1
    expect_violation(d, "subsequence_begins size != past_lens size + 1");
}

TEST(PADispatchContract, SubsequenceBeginsMustStartAtZero) {
    auto d = make_valid_dispatch();
    d.subsequence_begins = {1, 4, 6};
    expect_violation(d, "subsequence_begins does not start at 0");
}

TEST(PADispatchContract, SubsequenceBeginsMustStrictlyIncrease) {
    auto d = make_valid_dispatch();
    d.subsequence_begins = {0, 4, 4};
    d.input_ids_size = 4;
    d.position_ids_token_count = 4;
    d.sampled_tokens_indices = {3, 3};
    expect_violation(d, "subsequence_begins is not strictly increasing");
}

TEST(PADispatchContract, NegativePastLenRejected) {
    auto d = make_valid_dispatch();
    d.past_lens = {0, -1};
    expect_violation(d, "negative past_lens entry");
}

TEST(PADispatchContract, MaxContextLenMustBoundEveryContext) {
    auto d = make_valid_dispatch();
    d.max_context_len = 7;  // subsequence 1 context is 6 past + 2 scheduled = 8
    expect_violation(d, "max_context_len < a subsequence's context length");
}

TEST(PADispatchContract, BlockIndicesBeginsMustBePrefixSum) {
    auto d = make_valid_dispatch();
    d.block_indices_begins = {0, 1, 2};  // back() != block_indices.size()
    expect_violation(d, "block_indices_begins is not a prefix-sum over block_indices");
}

TEST(PADispatchContract, BlocksMustCoverEachContext) {
    auto d = make_valid_dispatch();
    d.block_indices = {0, 1};
    d.block_indices_begins = {0, 1, 2};  // one block (4 slots) for a context of 8
    expect_violation(d, "block_indices do not cover a subsequence's context");
}

// A block_size of 0 means the cache geometry is still dynamic; coverage is
// not checked, everything else still is.
TEST(PADispatchContract, DynamicBlockSizeSkipsCoverageOnly) {
    auto d = make_valid_dispatch();
    d.block_indices = {0};
    d.block_indices_begins = {0, 0, 1};
    EXPECT_NO_THROW(validate_dispatch(d, 0u, 0u));
    d.max_context_len = 7;
    expect_violation(d, "max_context_len < a subsequence's context length", 0u);
}

TEST(PADispatchContract, AbsentBlockTableSkipsBlockChecks) {
    auto d = make_valid_dispatch();
    d.has_block_table = false;
    d.block_indices.clear();
    d.block_indices_begins.clear();
    EXPECT_NO_THROW(validate_dispatch(d, kBlockSize, 0u));
}

// An empty selection is legal (intermediate prefill chunks gather nothing).
TEST(PADispatchContract, EmptySampledTokensIsLegal) {
    auto d = make_valid_dispatch();
    d.sampled_tokens_indices.clear();
    EXPECT_NO_THROW(validate_dispatch(d, kBlockSize, 0u));
}

TEST(PADispatchContract, SampledTokensOutOfRangeRejected) {
    auto d = make_valid_dispatch();
    d.sampled_tokens_indices = {6};
    expect_violation(d, "sampled_tokens_indices out of token range");
}

TEST(PADispatchContract, ViolationNamesTheDispatch) {
    auto d = make_valid_dispatch();
    d.past_lens = {0, -1};
    try {
        validate_dispatch(d, kBlockSize, 7u);
        FAIL() << "expected a dispatch-contract violation";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("PA dispatch #7"), std::string::npos) << ex.what();
    }
}

// A dispatch shaped as N subsequences of the given scheduled lengths; only the
// fields plan_calls reads are populated.
Dispatch make_dispatch_of(const std::vector<int32_t>& seq_lens) {
    Dispatch d;
    d.subsequence_begins = {0};
    for (const auto len : seq_lens) {
        d.past_lens.push_back(0);
        d.subsequence_begins.push_back(d.subsequence_begins.back() + len);
    }
    return d;
}

// The plan as text: "128[s0:0+128] 4[s1:0+1,s2:0+1,s3:0+1,s4:0+1] dyn[s0:128+2]".
std::string plan_of(const std::vector<int32_t>& seq_lens, const std::vector<std::size_t>& chunk_sizes) {
    std::string text;
    for (const auto& call : plan_calls(make_dispatch_of(seq_lens), chunk_sizes)) {
        text += (text.empty() ? "" : " ") + (call.chunk_size ? std::to_string(call.chunk_size) : "dyn") + "[";
        for (const auto& seg : call.segments) {
            text += (text.back() == '[' ? "s" : ",s") + std::to_string(seg.seq) + ":" + std::to_string(seg.offset) +
                    "+" + std::to_string(seg.tokens);
        }
        text += "]";
    }
    return text;
}

const std::vector<std::size_t> kChunkSizes = {1024u, 128u, 8u, 4u, 1u};

TEST(PADispatchPlan, SingleSequenceDecodeTakesTheOneTokenVariant) {
    EXPECT_EQ(plan_of({1}, kChunkSizes), "1[s0:0+1]");
    EXPECT_EQ(plan_of({1}, {1024u, 128u}), "");
}

TEST(PADispatchPlan, DecodeBatchFillsTheMultiTokenVariants) {
    EXPECT_EQ(plan_of({1, 1, 1, 1}, kChunkSizes), "4[s0:0+1,s1:0+1,s2:0+1,s3:0+1]");
    EXPECT_EQ(plan_of({1, 1, 1, 1, 1}, kChunkSizes), "4[s0:0+1,s1:0+1,s2:0+1,s3:0+1] 1[s4:0+1]");
    EXPECT_EQ(plan_of({1, 1, 1, 1, 1, 1}, kChunkSizes), "4[s0:0+1,s1:0+1,s2:0+1,s3:0+1] dyn[s4:0+1,s5:0+1]");
    EXPECT_EQ(plan_of(std::vector<int32_t>(13, 1), kChunkSizes),
              "8[s0:0+1,s1:0+1,s2:0+1,s3:0+1,s4:0+1,s5:0+1,s6:0+1,s7:0+1] "
              "4[s8:0+1,s9:0+1,s10:0+1,s11:0+1] 1[s12:0+1]");
}

TEST(PADispatchPlan, DecodeBatchTooSmallForAVariantRunsOneToOne) {
    EXPECT_EQ(plan_of({1, 1}, kChunkSizes), "");
    EXPECT_EQ(plan_of({1, 1, 1}, kChunkSizes), "");
    EXPECT_EQ(plan_of({1, 1, 1, 1}, {1024u, 128u, 1u}), "");
}

TEST(PADispatchPlan, PrefillTooShortForAVariantRunsOneToOne) {
    EXPECT_EQ(plan_of({3}, kChunkSizes), "");
    EXPECT_EQ(plan_of({2, 3}, kChunkSizes), "");
    EXPECT_EQ(plan_of({127}, {1024u, 128u, 1u}), "");
}

TEST(PADispatchPlan, PrefillIsCutIntoTheLargestVariantsThatFit) {
    EXPECT_EQ(plan_of({128}, kChunkSizes), "128[s0:0+128]");
    EXPECT_EQ(plan_of({1300}, kChunkSizes),
              "1024[s0:0+1024] 128[s0:1024+128] 128[s0:1152+128] 8[s0:1280+8] 8[s0:1288+8] 4[s0:1296+4]");
    EXPECT_EQ(plan_of({130}, kChunkSizes), "128[s0:0+128] dyn[s0:128+2]");
    EXPECT_EQ(plan_of({129}, kChunkSizes), "128[s0:0+128] 1[s0:128+1]");
}

TEST(PADispatchPlan, LeftoversOfAllSubsequencesShareOneDynamicCall) {
    EXPECT_EQ(plan_of({6, 1, 130}, kChunkSizes), "4[s0:0+4] 128[s2:0+128] 1[s1:0+1] dyn[s0:4+2,s2:128+2]");
    EXPECT_EQ(plan_of({129, 2, 1}, kChunkSizes), "128[s0:0+128] dyn[s0:128+1,s1:0+2,s2:0+1]");
}

TEST(PADispatchPlan, LastTokensOfPrefillsJoinTheDecodeBatch) {
    EXPECT_EQ(plan_of({129, 1, 5, 1}, kChunkSizes), "128[s0:0+128] 4[s2:0+4] 4[s0:128+1,s1:0+1,s2:4+1,s3:0+1]");
}

TEST(PADispatchPlan, NoVariantsOrNoTokensRunOneToOne) {
    EXPECT_EQ(plan_of({7638}, {}), "");
    EXPECT_EQ(plan_of({}, kChunkSizes), "");
}

}  // namespace
