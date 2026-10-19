// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the PA dispatch helpers (ov::npuw::pa): the contract
// validation, the chunk plan and the controls of one chunk. They are pure
// functions over the parsed control tensors, so the tests construct Dispatch
// values directly.

#include "pa_dispatch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"

namespace {

using ov::npuw::pa::Chunk;
using ov::npuw::pa::chunk_controls;
using ov::npuw::pa::Dispatch;
using ov::npuw::pa::plan_chunks;
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

// A dispatch of subsequences with the given scheduled and past lengths, one
// block of 4 per 4 context tokens and, by default, each subsequence's last
// row sampled.
Dispatch make_flat_dispatch(const std::vector<int32_t>& seq_lens,
                            const std::vector<int32_t>& past = {},
                            const std::vector<int64_t>& sampled = {}) {
    Dispatch d;
    d.subsequence_begins = {0};
    d.block_indices_begins = {0};
    for (std::size_t s = 0; s < seq_lens.size(); ++s) {
        d.past_lens.push_back(past.empty() ? 0 : past[s]);
        d.subsequence_begins.push_back(d.subsequence_begins.back() + seq_lens[s]);
        const auto ctx = d.past_lens.back() + seq_lens[s];
        for (int32_t b = 0; b < (ctx + 3) / 4; ++b) {
            d.block_indices.push_back(static_cast<int32_t>(d.block_indices.size()));
        }
        d.block_indices_begins.push_back(static_cast<int32_t>(d.block_indices.size()));
        if (sampled.empty()) {
            d.sampled_tokens_indices.push_back(d.subsequence_begins.back() - 1);
        }
    }
    if (!sampled.empty()) {
        d.sampled_tokens_indices = sampled;
    }
    d.has_block_table = true;
    d.has_sampled_tokens = true;
    return d;
}

const std::vector<std::size_t> kChunkSizes = {1024u, 128u, 1u};
constexpr std::size_t kMaxSampled = 16u;

// The plan as "tokens[/chunk_size]+...", checked to tile the token axis with
// no chunk sampling more rows than its variant produces.
std::string plan(const Dispatch& d) {
    const auto chunks = plan_chunks(d, kChunkSizes, kMaxSampled);
    std::string out;
    int64_t next = 0;
    for (const auto& c : chunks) {
        EXPECT_EQ(c.begin, next);
        EXPECT_GT(c.tokens, 0);
        EXPECT_LE(static_cast<std::size_t>(c.tokens), c.chunk_size);
        const auto sampled =
            std::count_if(d.sampled_tokens_indices.begin(), d.sampled_tokens_indices.end(), [&](int64_t row) {
                return row >= c.begin && row < c.begin + c.tokens;
            });
        EXPECT_LE(static_cast<std::size_t>(sampled), std::min(c.chunk_size, kMaxSampled));
        next = c.begin + c.tokens;
        out += (out.empty() ? "" : "+") + std::to_string(c.tokens);
        if (static_cast<std::size_t>(c.tokens) != c.chunk_size) {
            out += "/" + std::to_string(c.chunk_size);
        }
    }
    EXPECT_EQ(next, d.tokens());
    return out;
}

TEST(PAChunkPlan, LongPrefillTakesTheLargestThatFitsAndPadsTheLast) {
    EXPECT_EQ(plan(make_flat_dispatch({2500})), "1024+1024+128+128+128+68/128");
    EXPECT_EQ(plan(make_flat_dispatch({1024})), "1024");
}

TEST(PAChunkPlan, ChunksCrossSubsequenceBoundaries) {
    EXPECT_EQ(plan(make_flat_dispatch({1300, 48, 192, 192}, {0, 0, 64, 0})), "1024+128+128+128+128+128+68/128");
}

TEST(PAChunkPlan, SingleRowTakesTheOneTokenVariant) {
    EXPECT_EQ(plan(make_flat_dispatch({1})), "1");
    EXPECT_EQ(plan(make_flat_dispatch({1025})), "1024+1");
}

TEST(PAChunkPlan, DecodeBatchIsOneChunk) {
    EXPECT_EQ(plan(make_flat_dispatch({1, 1, 1, 1, 1, 1}, {7, 30, 2, 9, 64, 1})), "6/128");
}

TEST(PAChunkPlan, ChunkEndsBeforeTheSampledRowsItCannotProduce) {
    EXPECT_EQ(plan(make_flat_dispatch(std::vector<int32_t>(40, 1))), "16/128+16/128+8/128");

    // Every row of a prompt sampled (echo, prompt logprobs).
    std::vector<int64_t> all_rows(200);
    std::iota(all_rows.begin(), all_rows.end(), 0);
    std::string expected;
    for (int i = 0; i < 12; ++i) {
        expected += "16/128+";
    }
    EXPECT_EQ(plan(make_flat_dispatch({200}, {}, all_rows)), expected + "8/128");
}

TEST(PAChunkPlan, MixedDecodeAndPrefill) {
    std::vector<int32_t> lens(20, 1);
    lens.push_back(1004);
    EXPECT_EQ(plan(make_flat_dispatch(lens)), "16/128+128+128+128+128+128+128+128+112/128");
}

TEST(PAChunkPlan, IntermediatePrefillSamplesNothing) {
    auto d = make_flat_dispatch({300});
    d.sampled_tokens_indices.clear();
    EXPECT_EQ(plan(d), "128+128+44/128");
}

// Subsequences A (6 rows, no past), B (7 rows, 5 past), C (3 rows, no past):
// blocks A {0,1}, B {2,3,4}, C {5}; the last row of each is sampled.
Dispatch make_three_subsequences() {
    return make_flat_dispatch({6, 7, 3}, {0, 5, 0});
}

TEST(PAChunkControls, ChunkInsideOneSubsequence) {
    const auto c = chunk_controls(make_three_subsequences(), Chunk{8, 3, 4u});
    EXPECT_EQ(c.past_lens, (std::vector<int32_t>{7}));
    EXPECT_EQ(c.subsequence_begins, (std::vector<int32_t>{0, 3}));
    EXPECT_EQ(c.block_indices, (std::vector<int32_t>{2, 3, 4}));
    EXPECT_EQ(c.block_indices_begins, (std::vector<int32_t>{0, 3}));
    EXPECT_TRUE(c.sampled_tokens_indices.empty());
    EXPECT_TRUE(c.logits_rows.empty());
}

TEST(PAChunkControls, ChunkStartingInsideASubsequence) {
    const auto c = chunk_controls(make_three_subsequences(), Chunk{4, 8, 8u});
    EXPECT_EQ(c.past_lens, (std::vector<int32_t>{4, 5}));
    EXPECT_EQ(c.subsequence_begins, (std::vector<int32_t>{0, 2, 8}));
    EXPECT_EQ(c.block_indices, (std::vector<int32_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(c.block_indices_begins, (std::vector<int32_t>{0, 2, 5}));
    EXPECT_EQ(c.sampled_tokens_indices, (std::vector<int64_t>{1}));
    EXPECT_EQ(c.logits_rows, (std::vector<std::size_t>{0}));
}

TEST(PAChunkControls, ChunkStartingAtASubsequence) {
    const auto c = chunk_controls(make_three_subsequences(), Chunk{6, 10, 16u});
    EXPECT_EQ(c.past_lens, (std::vector<int32_t>{5, 0}));
    EXPECT_EQ(c.subsequence_begins, (std::vector<int32_t>{0, 7, 10}));
    EXPECT_EQ(c.block_indices, (std::vector<int32_t>{2, 3, 4, 5}));
    EXPECT_EQ(c.block_indices_begins, (std::vector<int32_t>{0, 3, 4}));
    EXPECT_EQ(c.sampled_tokens_indices, (std::vector<int64_t>{6, 9}));
    EXPECT_EQ(c.logits_rows, (std::vector<std::size_t>{1, 2}));
}

TEST(PAChunkControls, SampledRowsKeepTheCallerOrder) {
    auto d = make_three_subsequences();
    d.sampled_tokens_indices = {15, 5, 12};
    const auto all = chunk_controls(d, Chunk{0, 16, 16u});
    EXPECT_EQ(all.sampled_tokens_indices, (std::vector<int64_t>{15, 5, 12}));
    EXPECT_EQ(all.logits_rows, (std::vector<std::size_t>{0, 1, 2}));
    const auto tail = chunk_controls(d, Chunk{8, 8, 8u});
    EXPECT_EQ(tail.sampled_tokens_indices, (std::vector<int64_t>{7, 4}));
    EXPECT_EQ(tail.logits_rows, (std::vector<std::size_t>{0, 2}));
}

}  // namespace
