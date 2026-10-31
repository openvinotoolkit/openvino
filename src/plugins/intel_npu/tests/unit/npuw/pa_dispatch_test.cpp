// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the PA dispatch-contract validation (ov::npuw::pa). The
// contract is a pure function over the parsed control tensors, so the tests
// construct Dispatch values directly and assert on the specific violation
// each malformed dispatch must report.

#include "pa_dispatch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"

namespace {

using ov::npuw::pa::Dispatch;
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
// fields the chunk planning reads are populated.
Dispatch make_dispatch_of(const std::vector<int32_t>& seq_lens) {
    Dispatch d;
    d.subsequence_begins = {0};
    for (const auto len : seq_lens) {
        d.past_lens.push_back(0);
        d.subsequence_begins.push_back(d.subsequence_begins.back() + len);
    }
    return d;
}

const std::vector<std::size_t> kChunkSizes = {1024u, 128u, 1u};

using ov::npuw::pa::Chunk;
using ov::npuw::pa::chunk_controls;
using ov::npuw::pa::plan_chunks;

// A dispatch as the CB pipeline builds it: per subsequence a past length, a
// scheduled length, a block table covering both (32-slot blocks, ids
// consecutive across subsequences) and the last row sampled.
Dispatch make_cb_dispatch(const std::vector<int32_t>& past_lens, const std::vector<int32_t>& seq_lens) {
    Dispatch d;
    d.subsequence_begins = {0};
    d.block_indices_begins = {0};
    for (std::size_t s = 0; s < seq_lens.size(); ++s) {
        const auto ctx = past_lens[s] + seq_lens[s];
        d.past_lens.push_back(past_lens[s]);
        d.subsequence_begins.push_back(d.subsequence_begins.back() + seq_lens[s]);
        for (int32_t b = 0; b < (ctx + 31) / 32; ++b) {
            d.block_indices.push_back(static_cast<int32_t>(d.block_indices.size()));
        }
        d.block_indices_begins.push_back(static_cast<int32_t>(d.block_indices.size()));
        d.sampled_tokens_indices.push_back(d.subsequence_begins.back() - 1);
        d.max_context_len = std::max<int64_t>(d.max_context_len, ctx);
    }
    d.input_ids_size = d.tokens();
    d.position_ids_token_count = d.tokens();
    d.has_block_table = true;
    d.has_sampled_tokens = true;
    return d;
}

// A: 300-row prompt, B: decode on 50 past, C: 200-row prompt.
// subsequence_begins {0, 300, 301, 501}, blocks A 0-9, B 10-11, C 12-18,
// sampled rows {299, 300, 500}.
Dispatch make_mixed_dispatch() {
    return make_cb_dispatch({0, 50, 0}, {300, 1, 200});
}

std::vector<int32_t> iota_vec(int32_t from, int32_t to) {
    std::vector<int32_t> v;
    for (auto i = from; i < to; ++i) {
        v.push_back(i);
    }
    return v;
}

void expect_plan(const std::vector<Chunk>& plan, const std::vector<std::pair<int64_t, std::size_t>>& expected) {
    ASSERT_EQ(plan.size(), expected.size());
    int64_t begin = 0;
    for (std::size_t i = 0; i < plan.size(); ++i) {
        EXPECT_EQ(plan[i].begin, begin) << "chunk " << i;
        EXPECT_EQ(plan[i].tokens, expected[i].first) << "chunk " << i;
        EXPECT_EQ(plan[i].chunk_size, expected[i].second) << "chunk " << i;
        begin += plan[i].tokens;
    }
}

TEST(PAChunkPlan, LongPrefillTakesLargestFirstThenDynamicTail) {
    std::vector<std::pair<int64_t, std::size_t>> expected(7, {1024, 1024u});
    expected.insert(expected.end(), 3, {128, 128u});
    expected.push_back({86, 0u});
    expect_plan(plan_chunks(make_dispatch_of({7638}), kChunkSizes), expected);
}

TEST(PAChunkPlan, ChunksSpanSubsequences) {
    expect_plan(plan_chunks(make_mixed_dispatch(), kChunkSizes), {{128, 128u}, {128, 128u}, {128, 128u}, {117, 0u}});
    expect_plan(plan_chunks(make_dispatch_of({100, 100, 100, 100, 100, 12}), kChunkSizes),
                {{128, 128u}, {128, 128u}, {128, 128u}, {128, 128u}});
}

TEST(PAChunkPlan, DecodeBatchIsOneInfer) {
    expect_plan(plan_chunks(make_dispatch_of({1, 1, 1, 1, 1, 1}), kChunkSizes), {{6, 0u}});
}

TEST(PAChunkPlan, OneTokenVariantOnlyForASingleRow) {
    expect_plan(plan_chunks(make_dispatch_of({1}), kChunkSizes), {{1, 1u}});
    expect_plan(plan_chunks(make_dispatch_of({129}), kChunkSizes), {{128, 128u}, {1, 1u}});
    expect_plan(plan_chunks(make_dispatch_of({130}), kChunkSizes), {{128, 128u}, {2, 0u}});
    expect_plan(plan_chunks(make_dispatch_of({1}), {1024u, 128u}), {{1, 0u}});
}

TEST(PAChunkPlan, UnsortedChunkSizesAndEmptyDispatch) {
    expect_plan(plan_chunks(make_dispatch_of({1100}), {1u, 128u, 1024u}), {{1024, 1024u}, {76, 0u}});
    EXPECT_TRUE(plan_chunks(make_dispatch_of({}), kChunkSizes).empty());
}

TEST(PAChunkControls, WindowInsideOneSubsequence) {
    const auto c = chunk_controls(make_mixed_dispatch(), Chunk{128, 128, 128u});
    EXPECT_EQ(c.first_sequence, 0);
    EXPECT_EQ(c.past_lens, (std::vector<int32_t>{128}));
    EXPECT_EQ(c.subsequence_begins, (std::vector<int32_t>{0, 128}));
    EXPECT_EQ(c.block_indices, iota_vec(0, 10));
    EXPECT_EQ(c.block_indices_begins, (std::vector<int32_t>{0, 10}));
    EXPECT_TRUE(c.sampled_tokens_indices.empty());
    EXPECT_TRUE(c.logits_rows.empty());
}

TEST(PAChunkControls, WindowSpansSeveralSubsequences) {
    const auto c = chunk_controls(make_mixed_dispatch(), Chunk{256, 128, 128u});
    EXPECT_EQ(c.first_sequence, 0);
    EXPECT_EQ(c.past_lens, (std::vector<int32_t>{256, 50, 0}));
    EXPECT_EQ(c.subsequence_begins, (std::vector<int32_t>{0, 44, 45, 128}));
    EXPECT_EQ(c.block_indices, iota_vec(0, 19));
    EXPECT_EQ(c.block_indices_begins, (std::vector<int32_t>{0, 10, 12, 19}));
    EXPECT_EQ(c.sampled_tokens_indices, (std::vector<int64_t>{43, 44}));
    EXPECT_EQ(c.logits_rows, (std::vector<std::size_t>{0, 1}));
}

TEST(PAChunkControls, WindowStartsMidSubsequence) {
    const auto c = chunk_controls(make_mixed_dispatch(), Chunk{384, 117, 0u});
    EXPECT_EQ(c.first_sequence, 2);
    EXPECT_EQ(c.past_lens, (std::vector<int32_t>{83}));
    EXPECT_EQ(c.subsequence_begins, (std::vector<int32_t>{0, 117}));
    EXPECT_EQ(c.block_indices, iota_vec(12, 19));
    EXPECT_EQ(c.block_indices_begins, (std::vector<int32_t>{0, 7}));
    EXPECT_EQ(c.sampled_tokens_indices, (std::vector<int64_t>{116}));
    EXPECT_EQ(c.logits_rows, (std::vector<std::size_t>{2}));
}

// Window edges on subsequence boundaries take no row of the neighbours.
TEST(PAChunkControls, FenceAlignedWindows) {
    const auto d = make_mixed_dispatch();
    const auto a = chunk_controls(d, Chunk{0, 300, 0u});
    EXPECT_EQ(a.first_sequence, 0);
    EXPECT_EQ(a.past_lens, (std::vector<int32_t>{0}));
    EXPECT_EQ(a.subsequence_begins, (std::vector<int32_t>{0, 300}));
    EXPECT_EQ(a.sampled_tokens_indices, (std::vector<int64_t>{299}));
    EXPECT_EQ(a.logits_rows, (std::vector<std::size_t>{0}));

    const auto b = chunk_controls(d, Chunk{300, 1, 1u});
    EXPECT_EQ(b.first_sequence, 1);
    EXPECT_EQ(b.past_lens, (std::vector<int32_t>{50}));
    EXPECT_EQ(b.subsequence_begins, (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(b.block_indices, (std::vector<int32_t>{10, 11}));
    EXPECT_EQ(b.block_indices_begins, (std::vector<int32_t>{0, 2}));
    EXPECT_EQ(b.sampled_tokens_indices, (std::vector<int64_t>{0}));
    EXPECT_EQ(b.logits_rows, (std::vector<std::size_t>{1}));
}

// Sampled rows keep the caller's order, not the row order.
TEST(PAChunkControls, SampledRowsMapToTheCallersOrder) {
    auto d = make_mixed_dispatch();
    d.sampled_tokens_indices = {500, 299, 0};
    const auto c = chunk_controls(d, Chunk{0, 501, 0u});
    EXPECT_EQ(c.past_lens, (std::vector<int32_t>{0, 50, 0}));
    EXPECT_EQ(c.sampled_tokens_indices, (std::vector<int64_t>{500, 299, 0}));
    EXPECT_EQ(c.logits_rows, (std::vector<std::size_t>{0, 1, 2}));

    const auto tail = chunk_controls(d, Chunk{256, 245, 0u});
    EXPECT_EQ(tail.sampled_tokens_indices, (std::vector<int64_t>{244, 43}));
    EXPECT_EQ(tail.logits_rows, (std::vector<std::size_t>{0, 1}));
}

// Over a whole plan every row is sent once, each subsequence's past grows by
// the rows sent before, and every sampled row lands in exactly one chunk.
TEST(PAChunkControls, PlanCoversTheDispatchOnce) {
    const auto d = make_cb_dispatch({0, 50, 7, 0, 3}, {300, 1, 1, 700, 140});
    std::vector<int32_t> sent(d.past_lens.size(), 0);
    std::vector<int> sampled_hits(d.sampled_tokens_indices.size(), 0);
    for (const auto& chunk : plan_chunks(d, kChunkSizes)) {
        const auto c = chunk_controls(d, chunk);
        EXPECT_EQ(c.subsequence_begins.back(), chunk.tokens);
        for (std::size_t j = 0; j < c.past_lens.size(); ++j) {
            const auto s = static_cast<std::size_t>(c.first_sequence) + j;
            EXPECT_EQ(c.past_lens[j], d.past_lens[s] + sent[s]);
            sent[s] += c.subsequence_begins[j + 1] - c.subsequence_begins[j];
        }
        for (const auto row : c.logits_rows) {
            ++sampled_hits[row];
        }
    }
    for (std::size_t s = 0; s < sent.size(); ++s) {
        EXPECT_EQ(sent[s], d.subsequence_begins[s + 1] - d.subsequence_begins[s]) << "subsequence " << s;
    }
    EXPECT_EQ(sampled_hits, std::vector<int>(sampled_hits.size(), 1));
}

}  // namespace
