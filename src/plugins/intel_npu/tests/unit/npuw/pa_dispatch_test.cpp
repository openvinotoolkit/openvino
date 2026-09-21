// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the PA dispatch contract and planner (ov::npuw::pa). Both
// are pure functions over the parsed control tensors, so the tests construct
// Dispatch values directly: the validation tests assert on the specific
// violation each malformed dispatch must report, the planner tests on the
// variant infers a dispatch is split into.

#include "pa_dispatch.hpp"

#include <gtest/gtest.h>

#include <string>

#include "openvino/core/except.hpp"

namespace {

using ov::npuw::pa::Chunk;
using ov::npuw::pa::Dispatch;
using ov::npuw::pa::plan_dispatch;
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

// A dispatch shaped as N subsequences of the given scheduled lengths, each
// sampling its last token; only the fields the planner reads are populated.
Dispatch make_dispatch_of(const std::vector<int64_t>& seq_lens) {
    Dispatch d;
    d.subsequence_begins = {0};
    for (const auto len : seq_lens) {
        d.past_lens.push_back(0);
        d.subsequence_begins.push_back(d.subsequence_begins.back() + len);
        d.sampled_tokens_indices.push_back(d.subsequence_begins.back() - 1);
    }
    return d;
}

const std::vector<std::size_t> kVariantDims = {1024u, 128u, 16u, 1u};
constexpr std::size_t kMaxSampled = 16u;

std::string plan_of(const std::vector<int64_t>& seq_lens) {
    return ov::npuw::pa::to_string(plan_dispatch(make_dispatch_of(seq_lens), kVariantDims, kMaxSampled));
}

TEST(PADispatchPlan, SingleDecodeTakesTheOneTokenVariant) {
    EXPECT_EQ(plan_of({1}), "1/1[0+0:1]");
}

TEST(PADispatchPlan, DecodeBatchIsOneInfer) {
    EXPECT_EQ(plan_of({1, 1, 1}), "3/16[0+0:1,1+0:1,2+0:1]");
}

TEST(PADispatchPlan, DecodeBatchSplitsAtMaxSampled) {
    const auto plan = plan_dispatch(make_dispatch_of(std::vector<int64_t>(20, 1)), kVariantDims, kMaxSampled);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].pieces.size(), 16u);
    EXPECT_EQ(plan[0].token_dim, 16u);
    EXPECT_EQ(plan[1].pieces.size(), 4u);
    EXPECT_EQ(plan[1].token_dim, 16u);
}

TEST(PADispatchPlan, PrefillTakesTheLargestVariantFirst) {
    EXPECT_EQ(plan_of({2048}), "1024/1024[0+0:1024] 1024/1024[0+1024:1024]");
}

TEST(PADispatchPlan, RemainderIsPaddedWhenPaddingIsCheap) {
    EXPECT_EQ(plan_of({86}), "86/128[0+0:86]");
    EXPECT_EQ(plan_of({1029}), "1024/1024[0+0:1024] 5/16[0+1024:5]");
}

TEST(PADispatchPlan, RemainderIsSplitWhenPaddingIsTooMuch) {
    // 176 would pad to 1024, over four times the tokens: take 128, then 48
    // pads to 128. 17 would pad to 128: 16 and 1.
    EXPECT_EQ(plan_of({1200}), "1024/1024[0+0:1024] 128/128[0+1024:128] 48/128[0+1152:48]");
    EXPECT_EQ(plan_of({17}), "16/16[0+0:16] 1/1[0+16:1]");
    EXPECT_EQ(plan_of({46}), "46/128[0+0:46]");
}

TEST(PADispatchPlan, MixedDispatchKeepsSequencesApartAndDecodesTogether) {
    EXPECT_EQ(plan_of({200, 1, 1}), "128/128[0+0:128] 72/128[0+128:72] 2/16[1+0:1,2+0:1]");
}

TEST(PADispatchPlan, TooManySampledRowsInOneChunkRejected) {
    auto d = make_dispatch_of({1024});
    d.sampled_tokens_indices.clear();
    for (int64_t i = 0; i < 1024; ++i) {
        d.sampled_tokens_indices.push_back(i);  // echo: every prompt token sampled
    }
    EXPECT_THROW(plan_dispatch(d, kVariantDims, kMaxSampled), ov::Exception);
}

TEST(PADispatchPlan, RequiresTheOneTokenVariant) {
    EXPECT_THROW(plan_dispatch(make_dispatch_of({1}), {1024u, 128u}, kMaxSampled), ov::Exception);
}

}  // namespace
