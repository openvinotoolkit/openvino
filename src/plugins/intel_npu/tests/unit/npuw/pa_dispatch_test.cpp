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
using ov::npuw::pa::validate_dispatch;
using ov::npuw::pa::variants_serve;

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
// fields variants_serve reads are populated.
Dispatch make_dispatch_of(const std::vector<int64_t>& seq_lens) {
    Dispatch d;
    d.subsequence_begins = {0};
    for (const auto len : seq_lens) {
        d.past_lens.push_back(0);
        d.subsequence_begins.push_back(d.subsequence_begins.back() + len);
    }
    return d;
}

const std::vector<std::size_t> kVariantDims = {1024u, 128u, 1u};

TEST(PADispatchRouting, SingleSequenceDecodeServedByTheOneTokenVariant) {
    EXPECT_TRUE(variants_serve(make_dispatch_of({1}), kVariantDims));
    EXPECT_FALSE(variants_serve(make_dispatch_of({1}), {1024u, 128u}));
}

TEST(PADispatchRouting, DecodeBatchRunsOneToOne) {
    EXPECT_FALSE(variants_serve(make_dispatch_of({1, 1, 1, 1}), kVariantDims));
}

TEST(PADispatchRouting, ShortPrefillRunsOneToOne) {
    EXPECT_FALSE(variants_serve(make_dispatch_of({32, 32, 32, 32}), kVariantDims));
    EXPECT_FALSE(variants_serve(make_dispatch_of({127}), kVariantDims));
}

TEST(PADispatchRouting, LongPrefillIsChunked) {
    EXPECT_TRUE(variants_serve(make_dispatch_of({128}), kVariantDims));
    EXPECT_TRUE(variants_serve(make_dispatch_of({7638}), kVariantDims));
}

TEST(PADispatchRouting, MixedDispatchWithOneLongSubsequenceIsChunked) {
    EXPECT_TRUE(variants_serve(make_dispatch_of({1, 1, 200, 1}), kVariantDims));
}

TEST(PADispatchRouting, NoVariantsOrNoTokensRunOneToOne) {
    EXPECT_FALSE(variants_serve(make_dispatch_of({7638}), {}));
    EXPECT_FALSE(variants_serve(make_dispatch_of({}), kVariantDims));
}

}  // namespace
