// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/paged_cache_manager.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "openvino/reference/paged_attention.hpp"

using namespace ov::reference::paged_attention_cache;

namespace {

// helper to build a zero-filled cache shape [num_blocks, kv_heads, block_size, head_size]
struct CacheLayout {
    std::size_t num_blocks;
    std::size_t kv_heads;
    std::size_t block_size;
    std::size_t head_size;

    ov::Shape shape() const {
        return {num_blocks, kv_heads, block_size, head_size};
    }
    std::size_t elems() const {
        return num_blocks * kv_heads * block_size * head_size;
    }
};

// shorthand node key
constexpr std::uintptr_t NODE = 42;

// register a simple operator state and return the manager
// caller owns the cache data vectors (keep them alive while manager is used)
std::unique_ptr<PagedCacheManager> make_manager(CacheLayout layout,
                                                EvictionPolicy policy,
                                                std::vector<float>& key_data,
                                                std::vector<float>& val_data,
                                                std::size_t seq_count = 1) {
    key_data.assign(layout.elems(), 0.f);
    val_data.assign(layout.elems(), 0.f);

    auto mgr = std::make_unique<PagedCacheManager>(ov::element::f32, policy, /*max_cache_bytes=*/0);

    // initial past_lens = 0 for each sequence
    std::vector<std::int32_t> past(seq_count, 0);
    // no initial block indices
    mgr->ensure_operator(NODE,
                         key_data.data(),
                         val_data.data(),
                         layout.shape(),
                         layout.shape(),
                         nullptr,
                         0,
                         nullptr,
                         0,
                         past.data(),
                         seq_count);
    return mgr;
}

}  // namespace

// -------------------------------------------------------------------
// Basic lifecycle: register, begin_step, ensure_token, resolve_token
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, BasicAllocateAndResolve) {
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    // write a token at position 0
    std::vector<float> krow(layout.kv_heads * layout.head_size, 1.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 2.f);
    mgr->write_token_kv<float>(NODE, 0, 0, krow.data(), vrow.data());

    // resolve it
    PagedCacheManager::TokenAddress addr;
    ASSERT_TRUE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_GE(addr.block, 0);

    // read back key data
    const float* kp = mgr->key_ptr<float>(NODE, addr, 0);
    ASSERT_NE(kp, nullptr);
    EXPECT_FLOAT_EQ(kp[0], 1.f);

    // read back value data
    const float* vp = mgr->value_ptr<float>(NODE, addr, 0);
    ASSERT_NE(vp, nullptr);
    EXPECT_FLOAT_EQ(vp[0], 2.f);
}

// -------------------------------------------------------------------
// FIFO eviction: fills all blocks then forces eviction from oldest
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, FIFOEvictionWorks) {
    // 4 blocks, block_size=2 => can hold 8 tokens total
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd, 2);

    EXPECT_EQ(mgr->eviction_policy(), EvictionPolicy::FIFO);

    // fill seq 0 with 6 tokens (3 blocks)
    std::int32_t past0 = 0;
    std::int32_t past1 = 0;
    std::int32_t pasts[2] = {past0, past1};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 6; t++) {
        krow[0] = static_cast<float>(t + 100);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // fill seq 1 with 2 tokens (1 block) => total 4 blocks used up
    for (int t = 0; t < 2; t++) {
        krow[0] = static_cast<float>(t + 200);
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // now all 4 blocks are used, write one more token for seq 1 which needs a new block
    // this should evict the oldest block from seq 0 (the longest)
    krow[0] = 999.f;
    mgr->write_token_kv<float>(NODE, 1, 2, krow.data(), vrow.data());

    // seq 0 tokens 0 and 1 (the first block) should be evicted
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));

    // seq 0 tokens 2-5 should still be accessible
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 5, addr));

    // seq 1 token 2 should be accessible (the new one)
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));
    const float* kp = mgr->key_ptr<float>(NODE, addr, 0);
    ASSERT_NE(kp, nullptr);
    EXPECT_FLOAT_EQ(kp[0], 999.f);
}

// -------------------------------------------------------------------
// SCORE eviction: picks the sequence whose front block has lowest score
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, ScoreEvictionPicksLowestScoreFrontBlock) {
    // 4 blocks, block_size=2, 1 head, head_size=4
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 2);

    EXPECT_EQ(mgr->eviction_policy(), EvictionPolicy::SCORE);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 4 tokens => 2 blocks
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }
    // seq 1: 4 tokens => 2 blocks, now all 4 blocks used
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t + 10);
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // feed attention scores so that seq 0 front block has LOW score and seq 1 front block has HIGH score
    // layout: [seq0_4_tokens, seq1_4_tokens]
    pasts[0] = 4;
    pasts[1] = 4;
    // seq 0 block 0 tokens: 0.1 each => front block score 0.2
    // seq 0 block 1 tokens: 5.0 each => block score 10.0
    // seq 1 block 0 tokens: 3.0 each => front block score 6.0
    // seq 1 block 1 tokens: 3.0 each => block score 6.0
    std::vector<float> scores = {0.1f, 0.1f, 5.0f, 5.0f, 3.f, 3.f, 3.f, 3.f};
    mgr->update_attention_scores(NODE, scores.data(), scores.size(), pasts, 2);

    // force a new allocation for seq 1 which will trigger eviction
    // should evict seq 0 front block (score 0.2) not seq 1 front block (score 6.0)
    krow[0] = 777.f;
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    // seq 0 tokens 0,1 should be evicted (front block of seq 0 had lowest score)
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));

    // seq 0 tokens 2,3 should survive
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 3, addr));

    // all seq 1 tokens should be intact
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 0, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 3, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 4, addr));
}

// -------------------------------------------------------------------
// SCORE eviction falls back to FIFO when no scores are recorded
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, ScoreEvictionFallsBackToFifo) {
    // 4 blocks, block_size=2, 2 sequences
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 6 tokens => 3 blocks
    for (int t = 0; t < 6; t++) {
        krow[0] = static_cast<float>(t);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }
    // seq 1: 2 tokens => 1 block, total 4 blocks used
    for (int t = 0; t < 2; t++) {
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // no scores fed, force eviction by adding a 5th block for seq 1
    krow[0] = 888.f;
    mgr->write_token_kv<float>(NODE, 1, 2, krow.data(), vrow.data());

    // FIFO fallback should evict oldest block from seq 0 (longest)
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));
}

// -------------------------------------------------------------------
// ADAPTIVE_RKV eviction: evicts from the sequence whose front block is least diverse
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, AdaptiveRKVEvictionPicksLeastDiverse) {
    // 4 blocks, block_size=2
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    EXPECT_EQ(mgr->eviction_policy(), EvictionPolicy::ADAPTIVE_RKV);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 4 tokens => 2 blocks
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }
    // seq 1: 4 tokens => 2 blocks, all 4 blocks used
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t + 10);
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // feed diversity: seq 0 front block has low diversity, seq 1 front block has high diversity
    float div0[2] = {0.1f, 10.f};  // seq 0: block 0 = 0.1, block 1 = 10.0
    float div1[2] = {5.f, 5.f};    // seq 1: block 0 = 5.0, block 1 = 5.0
    mgr->update_diversity_scores(NODE, 0, div0, 2, 0);
    mgr->update_diversity_scores(NODE, 1, div1, 2, 0);

    // force eviction by writing token 4 for seq 1
    krow[0] = 666.f;
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    // seq 0 front block (tokens 0,1) should be evicted (diversity 0.1)
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));  // block 1 survived
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 0, addr));  // seq 1 intact
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 4, addr));  // new token accessible
}

// -------------------------------------------------------------------
// ADAPTIVE_RKV falls back to SCORE then FIFO if no diversity data
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, AdaptiveRKVFallsBackToScoreThenFifo) {
    // 4 blocks, 2 sequences
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 6 tokens => 3 blocks
    for (int t = 0; t < 6; t++) {
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }
    // seq 1: 2 tokens => 1 block, total 4 blocks
    for (int t = 0; t < 2; t++) {
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // no diversity, no scores => should fall all the way back to FIFO
    krow[0] = 111.f;
    mgr->write_token_kv<float>(NODE, 1, 2, krow.data(), vrow.data());

    // FIFO should evict from seq 0 (longest)
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));
}

// -------------------------------------------------------------------
// Sequence reset clears blocks and score/diversity state
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, SequenceResetClearsState) {
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 1);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    std::vector<float> krow(4, 1.f);
    std::vector<float> vrow(4, 1.f);
    for (int t = 0; t < 4; t++) {
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // feed some scores
    past = 4;
    std::vector<float> scores = {1.f, 2.f, 3.f, 4.f};
    mgr->update_attention_scores(NODE, scores.data(), scores.size(), &past, 1);

    // reset the sequence by setting past_lens to 0
    past = 0;
    mgr->begin_step(NODE, &past, 1);

    // all old tokens should be gone
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 3, addr));

    // should be able to write new tokens without issues (blocks were freed)
    mgr->write_token_kv<float>(NODE, 0, 0, krow.data(), vrow.data());
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 0, addr));
}

// -------------------------------------------------------------------
// Multiple evictions in a row dont corrupt memory
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, RepeatedEvictionsNoCorruption) {
    // small pool: 3 blocks of size 2
    CacheLayout layout{3, 2, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd, 1);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // write 20 tokens, which forces many evictions with only 3 blocks (6 token slots)
    for (int t = 0; t < 20; t++) {
        for (std::size_t j = 0; j < krow.size(); j++) {
            krow[j] = static_cast<float>(t * 100 + j);
        }
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // only the last ~6 tokens should be resolvable (3 blocks * 2 tokens/block)
    // earlier ones were evicted
    PagedCacheManager::TokenAddress addr;
    int resolvable = 0;
    for (int t = 0; t < 20; t++) {
        if (mgr->resolve_token(NODE, 0, t, addr)) {
            resolvable++;
            // check that we can read back the key data without crashing
            const float* kp = mgr->key_ptr<float>(NODE, addr, 0);
            ASSERT_NE(kp, nullptr);
            EXPECT_FLOAT_EQ(kp[0], static_cast<float>(t * 100));
        }
    }

    // we expect roughly 6 tokens to be resolvable (3 blocks * 2)
    EXPECT_GE(resolvable, 4);
    EXPECT_LE(resolvable, 8);  // some slack for boundary effects
}

// -------------------------------------------------------------------
// Default constructor uses SCORE policy
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, DefaultConstructorUsesScore) {
    PagedCacheManager mgr(ov::element::f32);
    EXPECT_EQ(mgr.eviction_policy(), EvictionPolicy::SCORE);
}

// -------------------------------------------------------------------
// Reference PA kernel with tiny cache forces eviction, no corruption
// -------------------------------------------------------------------
TEST(PagedCacheManagerTest, ReferenceKernelEvictionNoCorruption) {
    // minimal config: 1 kv head, head_size=4, block_size=2, only 3 blocks
    // with 3 blocks we can hold 6 tokens, so a sequence of 8 forces eviction
    const std::size_t num_blocks = 3;
    const std::size_t kv_heads = 1;
    const std::size_t blk_size = 2;
    const std::size_t head_size = 4;
    const std::size_t q_heads = 1;
    const std::size_t q_features = q_heads * head_size;
    const std::size_t kv_features = kv_heads * head_size;

    // initial (empty) key and value caches
    const std::size_t cache_elems = num_blocks * kv_heads * blk_size * head_size;
    std::vector<float> key_cache(cache_elems, 0.f);
    std::vector<float> val_cache(cache_elems, 0.f);
    ov::Shape cache_shape = {num_blocks, kv_heads, blk_size, head_size};

    // create a manager with SCORE policy
    auto mgr = std::make_unique<PagedCacheManager>(ov::element::f32, EvictionPolicy::SCORE, 0);

    // empty block indices for initial registration
    std::vector<std::int32_t> block_idx = {0, 1, 2};
    std::vector<std::int32_t> block_begins = {0, 3};

    // run two steps: first step with 6 tokens (fills cache), second step with 2 more (forces eviction)
    constexpr std::uintptr_t nk = 99;

    // --- step 1: prompt with 6 tokens ---
    const std::size_t step1_tokens = 6;
    std::vector<float> q1(step1_tokens * q_features);
    std::vector<float> k1(step1_tokens * kv_features);
    std::vector<float> v1(step1_tokens * kv_features);
    for (std::size_t i = 0; i < q1.size(); i++)
        q1[i] = 0.1f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < k1.size(); i++)
        k1[i] = 0.2f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < v1.size(); i++)
        v1[i] = 0.3f * static_cast<float>(i + 1);

    std::int32_t past1 = 0;
    std::int32_t subseq1[2] = {0, static_cast<std::int32_t>(step1_tokens)};
    float scale_val = 1.0f / std::sqrt(static_cast<float>(head_size));
    std::int32_t score_window = 0;

    std::vector<float> out1(step1_tokens * q_features, -999.f);

    ov::reference::paged_attention<float>(nk,
                                          mgr.get(),
                                          out1.data(),
                                          nullptr,
                                          nullptr,
                                          q1.data(),
                                          k1.data(),
                                          v1.data(),
                                          key_cache.data(),
                                          val_cache.data(),
                                          &past1,
                                          subseq1,
                                          block_idx.data(),
                                          block_idx.size(),
                                          block_begins.data(),
                                          block_begins.size(),
                                          &scale_val,
                                          ov::element::f32,
                                          nullptr,
                                          nullptr,
                                          ov::element::Type{},
                                          {},
                                          nullptr,
                                          &score_window,
                                          nullptr,
                                          0,
                                          nullptr,
                                          {},
                                          nullptr,
                                          ov::element::Type{},
                                          {},
                                          nullptr,
                                          ov::element::Type{},
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          ov::element::Type{},
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          {step1_tokens, q_features},
                                          {step1_tokens, kv_features},
                                          {step1_tokens, kv_features},
                                          cache_shape,
                                          cache_shape,
                                          {1},
                                          {2});

    // check no NaN or inf in output
    for (std::size_t i = 0; i < out1.size(); i++) {
        EXPECT_TRUE(std::isfinite(out1[i])) << "step 1 output[" << i << "] is not finite";
    }

    // --- step 2: decode with 2 more tokens, forces eviction ---
    const std::size_t step2_tokens = 2;
    std::vector<float> q2(step2_tokens * q_features);
    std::vector<float> k2(step2_tokens * kv_features);
    std::vector<float> v2(step2_tokens * kv_features);
    for (std::size_t i = 0; i < q2.size(); i++)
        q2[i] = 0.4f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < k2.size(); i++)
        k2[i] = 0.5f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < v2.size(); i++)
        v2[i] = 0.6f * static_cast<float>(i + 1);

    std::int32_t past2 = static_cast<std::int32_t>(step1_tokens);
    std::int32_t subseq2[2] = {0, static_cast<std::int32_t>(step2_tokens)};

    std::vector<float> out2(step2_tokens * q_features, -999.f);

    ov::reference::paged_attention<float>(nk,
                                          mgr.get(),
                                          out2.data(),
                                          nullptr,
                                          nullptr,
                                          q2.data(),
                                          k2.data(),
                                          v2.data(),
                                          key_cache.data(),
                                          val_cache.data(),
                                          &past2,
                                          subseq2,
                                          block_idx.data(),
                                          block_idx.size(),
                                          block_begins.data(),
                                          block_begins.size(),
                                          &scale_val,
                                          ov::element::f32,
                                          nullptr,
                                          nullptr,
                                          ov::element::Type{},
                                          {},
                                          nullptr,
                                          &score_window,
                                          nullptr,
                                          0,
                                          nullptr,
                                          {},
                                          nullptr,
                                          ov::element::Type{},
                                          {},
                                          nullptr,
                                          ov::element::Type{},
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          ov::element::Type{},
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          {step2_tokens, q_features},
                                          {step2_tokens, kv_features},
                                          {step2_tokens, kv_features},
                                          cache_shape,
                                          cache_shape,
                                          {1},
                                          {2});

    // check no NaN, inf, or uninitialized values in output
    for (std::size_t i = 0; i < out2.size(); i++) {
        EXPECT_TRUE(std::isfinite(out2[i])) << "step 2 output[" << i << "] is not finite";
        EXPECT_NE(out2[i], -999.f) << "step 2 output[" << i << "] was never written";
    }
}
